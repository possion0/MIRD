
import torch
import torch.nn as nn
from typing import Optional, Dict, List, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType


class QwenAgent(nn.Module):

    
    def __init__(
        self,
        model_path: str = "",
        lora_config: Optional[Dict] = None,
        device: str = "cuda",
        load_in_8bit: bool = False,
        load_in_4bit: bool = False
    ):

        super(QwenAgent, self).__init__()
        
        self.model_path = model_path
        self.device = device
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            padding_side='left'
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with optional quantization
        load_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.bfloat16,
        }
        
        if load_in_8bit:
            load_kwargs["load_in_8bit"] = True
        elif load_in_4bit:
            load_kwargs["load_in_4bit"] = True
        else:
            load_kwargs["device_map"] = "auto"
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            **load_kwargs
        )
        
        # Apply LoRA if config provided
        if lora_config is not None:
            self._apply_lora(lora_config)
        
        # Get model hidden size for embedding injection
        self.hidden_size = self.model.config.hidden_size
        
        # Special token for visual feature placeholder
        self.feature_placeholder = "<|visual_feature|>"
        
    def _apply_lora(self, lora_config: Dict):
        """Apply LoRA adapter to the model"""
        peft_config = LoraConfig(
            r=lora_config.get('r', 16),
            lora_alpha=lora_config.get('alpha', 32),
            lora_dropout=lora_config.get('dropout', 0.05),
            target_modules=lora_config.get('target_modules', 
                                           ["q_proj", "k_proj", "v_proj", "o_proj"]),
            bias=lora_config.get('bias', "none"),
            task_type=TaskType.CAUSAL_LM
        )
        self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()
    
    def get_input_embeddings(self):
        """Get the model's input embedding layer"""
        return self.model.get_input_embeddings()
    
    def _prepare_inputs_with_features(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        feature_embeddings: torch.Tensor,
        feature_positions: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        batch_size = input_ids.size(0)
        
        # Get token embeddings
        embed_layer = self.get_input_embeddings()
        token_embeddings = embed_layer(input_ids)  # (batch_size, seq_len, hidden_size)
        
        # Cast feature_embeddings to match model dtype (e.g. float16)
        feature_embeddings = feature_embeddings.to(dtype=token_embeddings.dtype)
        
        # Ensure feature embeddings have correct shape
        if feature_embeddings.dim() == 2:
            feature_embeddings = feature_embeddings.unsqueeze(1)  # (batch_size, 1, hidden_size)
        
        num_feature_tokens = feature_embeddings.size(1)
        
        # Default: prepend feature embeddings
        if feature_positions is None:
            # Concatenate: [feature_tokens, text_tokens]
            inputs_embeds = torch.cat([feature_embeddings, token_embeddings], dim=1)
            
            # Update attention mask
            feature_mask = torch.ones(
                batch_size, num_feature_tokens,
                device=attention_mask.device,
                dtype=attention_mask.dtype
            )
            attention_mask = torch.cat([feature_mask, attention_mask], dim=1)
        else:
            # Insert at specified positions
            inputs_embeds = token_embeddings.clone()
            for b in range(batch_size):
                pos = feature_positions[b].item() if feature_positions.dim() > 0 else feature_positions.item()
                inputs_embeds[b, pos:pos+num_feature_tokens] = feature_embeddings[b]
        
        return inputs_embeds, attention_mask
    
    def forward(
        self,
        feature_embeddings: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        text_list: Optional[List[str]] = None,
        **kwargs
    ):

        # If text_list provided, tokenize
        if text_list is not None and input_ids is None:
            prompts = [self._build_prompt(text) for text in text_list]
            encoded = self.tokenizer(
                prompts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(feature_embeddings.device)
            input_ids = encoded.input_ids
            attention_mask = encoded.attention_mask
        
        # Prepare inputs with injected features
        inputs_embeds, attention_mask = self._prepare_inputs_with_features(
            input_ids, attention_mask, feature_embeddings
        )

        # Pad labels if provided to match input length (prepend -100 for feature tokens)
        if labels is not None:
             # inputs_embeds length = num_features + input_ids length
             # labels length = input_ids length
             # We need to prepend num_features of -100 to labels
             num_features = feature_embeddings.size(1) if feature_embeddings.dim() == 3 else 1
             padding_labels = torch.full((labels.size(0), num_features), -100, dtype=labels.dtype, device=labels.device)
             labels = torch.cat([padding_labels, labels], dim=1)
        
        # Forward through model
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs
        )
        
        return outputs
    
    def _build_prompt(self, text: str) -> str:
        """Build analysis prompt for a given text"""
        prompt = f"""Analyze the provided multimodal content (Image + Text) for sarcasm detection.
The image features are provided above.

Text: {text}

Based on the visual and textual features, provide your analysis:
1. Text sentiment analysis
2. Visual content interpretation  
3. Text-image relationship
4. Sarcasm judgment

Analysis:"""
        return prompt
    
    @torch.no_grad()
    def generate(
        self,
        feature_embeddings: torch.Tensor,
        text_list: List[str],
        max_new_tokens: int = 150,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True
    ) -> List[str]:

        self.model.eval()
        
        # Build prompts
        prompts = [self._build_prompt(text) for text in text_list]
        
        # Tokenize
        encoded = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=400,
            return_tensors="pt"
        ).to(feature_embeddings.device)
        
        # Prepare inputs with features
        inputs_embeds, attention_mask = self._prepare_inputs_with_features(
            encoded.input_ids,
            encoded.attention_mask,
            feature_embeddings
        )
        
        # Generate
        outputs = self.model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )
        
        # Decode
        generated_texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        # Extract only the generated part (after the prompt)
        results = []
        for i, full_text in enumerate(generated_texts):
            # Find where the analysis starts
            if "Analysis:" in full_text:
                analysis = full_text.split("Analysis:")[-1].strip()
            else:
                analysis = full_text
            results.append(analysis)
        
        return results
    
    def parse_prediction(self, generated_text: str) -> Tuple[int, str]:

        text_lower = generated_text.lower()
        
        # Check for sarcasm indicators
        sarcasm_keywords = ["yes, sarcasm", "is sarcastic", "sarcasm detected", 
                          "this is sarcastic", "sarcasm: yes"]
        non_sarcasm_keywords = ["no sarcasm", "not sarcastic", "no, not sarcasm",
                               "sarcasm: no", "this is not sarcastic"]
        
        for keyword in sarcasm_keywords:
            if keyword in text_lower:
                return 1, generated_text
        
        for keyword in non_sarcasm_keywords:
            if keyword in text_lower:
                return 0, generated_text
        
        # Default: check for contradiction mentions
        if "contradiction" in text_lower or "ironic" in text_lower:
            return 1, generated_text
        
        return 0, generated_text
    
    def save_lora_adapter(self, save_path: str):
        """Save LoRA adapter weights"""
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
    
    def load_lora_adapter(self, adapter_path: str):
        """Load LoRA adapter weights"""
        from peft import PeftModel
        
        # Prevent double-wrapping: if already PeftModel, unwrap to base model
        if isinstance(self.model, PeftModel):
            # Access the underlying base model
            # Structure: PeftModel -> BaseTuner -> model (base model)
            # Or simply use .get_base_model() if available, but .base_model.model matches train_phase3.py logic
            base_model = self.model.base_model.model
            self.model = PeftModel.from_pretrained(base_model, adapter_path)
            print(f"Reloaded LoRA adapter from {adapter_path} (unwrapped existing PeftModel)")
        else:
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
            print(f"Loaded LoRA adapter from {adapter_path}")


def build_qwen_agent(config) -> QwenAgent:
    """
    Factory function to build QwenAgent from configuration.
    
    Args:
        config: AgentConfig or dict with qwen and lora settings
    
    Returns:
        agent: QwenAgent instance
    """
    qwen_config = config.qwen if hasattr(config, 'qwen') else config
    lora_config = config.lora if hasattr(config, 'lora') else None
    
    lora_dict = None
    if lora_config is not None:
        lora_dict = {
            'r': lora_config.r,
            'alpha': lora_config.alpha,
            'dropout': lora_config.dropout,
            'target_modules': lora_config.target_modules,
            'bias': lora_config.bias
        }
    
    return QwenAgent(
        model_path=qwen_config.model_path,
        lora_config=lora_dict,
        device=config.device if hasattr(config, 'device') else "cuda"
    )
