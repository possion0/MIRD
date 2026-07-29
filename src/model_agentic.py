from typing import Optional, List, Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import copy

from transformers import CLIPModel, BertConfig, BertTokenizer, BertModel
from transformers.models.bert.modeling_bert import BertLayer

from backbone import build_backbone
from projector import FeatureProjector, build_projector
from qwen_agent import QwenAgent, build_qwen_agent


class MultimodalEncoder(nn.Module):
    """Multimodal transformer encoder using BERT layers"""
    def __init__(self, config, layer_number):
        super(MultimodalEncoder, self).__init__()
        layer = BertLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(layer_number)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        all_encoder_layers = []
        all_encoder_attentions = []
        for layer_module in self.layer:
            hidden_states, attention = layer_module(hidden_states, attention_mask, output_attentions=True)
            all_encoder_attentions.append(attention)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers, all_encoder_attentions


class CrossAttention(nn.Module):
    """Cross-attention module for text-image interaction"""
    def __init__(self, feature_dim, dropout_prob=0.1):
        super(CrossAttention, self).__init__()
        self.text_linear = nn.Linear(feature_dim, feature_dim)
        self.extra_linear = nn.Linear(feature_dim, feature_dim)
        self.query_proj = nn.Linear(feature_dim, feature_dim)
        self.key_proj = nn.Linear(feature_dim, feature_dim)
        self.value_proj = nn.Linear(feature_dim, feature_dim)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, query, key, value):
        if query.shape[-1] != 768:
            query = self.text_linear(query)
        if key.shape[-1] != 768:
            key = self.extra_linear(key)
            value = self.extra_linear(value)
        query = self.query_proj(query)
        key = self.key_proj(key)
        value = self.value_proj(value)
        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = attention_scores / torch.sqrt(
            torch.tensor(key.size(-1), dtype=torch.float32, device=key.device)
        )
        attention_weights = F.softmax(attention_scores, dim=-1)
        attended_values = torch.matmul(attention_weights, value)
        attended_values = self.dropout(attended_values)
        return attended_values


class TransformerEncoderLayer(nn.Module):
    """Transformer encoder layer with self-attention"""
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, src, src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class TransformerCrossLayer(nn.Module):
    """Transformer cross-attention layer"""
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, return_attn=False,
                 kdim=None, vdim=None):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, kdim=kdim, vdim=vdim, batch_first=True
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self.return_attn = return_attn

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        tgt2, cross_attn = self.multihead_attn(
            query=self.with_pos_embed(tgt, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory, attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask
        )
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


class AgenticRCLMuFN(nn.Module):

    
    def __init__(self, args, mode='perception', agent_config=None):

        super(AgenticRCLMuFN, self).__init__()
        
        self.mode = mode
        self.args = args
        
        # ============== System 1: Perception Layer ==============
        # CLIP model
        self.model = CLIPModel.from_pretrained("./MMSD2.0-main/openai/clip-vit-base-patch32")
        
        # BERT config and encoder
        self.config = BertConfig.from_pretrained("./MMSD2.0-main/bert-base-uncased")
        self.config.hidden_size = 768
        self.config.num_attention_heads = 8
        self.trans = MultimodalEncoder(self.config, layer_number=args.layers)
        
        # Linear projections
        if args.simple_linear:
            self.text_linear = nn.Linear(args.text_size, args.image_size)
            self.image_linear = nn.Linear(args.image_size, args.image_size)
        else:
            self.text_linear = nn.Sequential(
                nn.Linear(args.text_size, args.image_size),
                nn.Dropout(args.dropout_rate),
                nn.GELU()
            )
            self.image_linear = nn.Sequential(
                nn.Linear(args.image_size, args.image_size),
                nn.Dropout(args.dropout_rate),
                nn.GELU()
            )
        
        # Classifier (for perception mode)
        self.classifier_fuse = nn.Linear(args.image_size, args.label_number)
        self.cross_att = CrossAttention(feature_dim=768, dropout_prob=0.1)
        self.loss_fct = nn.CrossEntropyLoss()
        
        # BERT model
        self.tokenizer = BertTokenizer.from_pretrained("./MMSD2.0-main/bert-base-uncased")
        self.bert_model = BertModel.from_pretrained("./MMSD2.0-main/bert-base-uncased")
        
        # ResNet backbone
        self.backbone = build_backbone(args)
        
        # Transformer components
        self.d_model = 768
        self.nheads = 8
        self.dim_feedforward = 2048
        
        self.txt = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.ReLU(),
            nn.LayerNorm(self.d_model)
        )
        self.txt2 = nn.Sequential(
            nn.Linear(self.d_model * 2, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, self.d_model),
            nn.LayerNorm(self.d_model)
        )
        self.vis2 = nn.Sequential(
            nn.Linear(self.d_model * 2, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, self.d_model),
            nn.LayerNorm(self.d_model)
        )
        
        self.text_self = TransformerEncoderLayer(self.d_model, self.nheads, dim_feedforward=self.dim_feedforward)
        self.text_cross = TransformerCrossLayer(self.d_model, self.nheads, dim_feedforward=self.dim_feedforward)
        self.vis_self = TransformerEncoderLayer(self.d_model, self.nheads, dim_feedforward=self.dim_feedforward)
        self.vis_cross = TransformerCrossLayer(self.d_model, self.nheads, dim_feedforward=self.dim_feedforward)
        self.imtxt_cross = TransformerCrossLayer(768, self.nheads, dim_feedforward=self.dim_feedforward)
        
        self.attetion_block = nn.Sequential(
            nn.Linear(self.d_model * 2, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, self.d_model),
            nn.Sigmoid()
        )
        self.mlp_layer = nn.Sequential(
            nn.Linear(self.d_model * 2, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, self.d_model),
            nn.LayerNorm(self.d_model)
        )
        
        self.input_proj = nn.Conv2d(self.dim_feedforward, 768, kernel_size=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # ============== Bridge: Feature Projector ==============
        self.projector = None
        self.qwen_agent = None
        
        if mode in ['alignment', 'agent'] and agent_config is not None:
            # Initialize projector
            self.projector = FeatureProjector(
                input_dim=agent_config.projector.input_dim,
                hidden_dim=agent_config.projector.hidden_dim,
                output_dim=agent_config.projector.output_dim,
                dropout=agent_config.projector.dropout
            )
            
            # Initialize Qwen agent (only for agent mode or alignment with LM loss)
            if mode == 'agent' or (mode == 'alignment'):
                lora_config = None
                if mode == 'agent':
                    lora_config = {
                        'r': agent_config.lora.r,
                        'alpha': agent_config.lora.alpha,
                        'dropout': agent_config.lora.dropout,
                        'target_modules': agent_config.lora.target_modules,
                        'bias': agent_config.lora.bias
                    }
                
                self.qwen_agent = QwenAgent(
                    model_path=agent_config.qwen.model_path,
                    lora_config=lora_config,
                    device=agent_config.device
                )
    
    def _perception_forward(self, inputs, batch) -> Tensor:

        device = inputs['input_ids'].device
        
        # CLIP forward
        output = self.model(**inputs, output_attentions=True)
        
        # Extract CLIP features
        text_feature = output['text_model_output']['pooler_output']  # (B, 512)
        image_feature = output['vision_model_output']['pooler_output']  # (B, 768)
        text_feature = self.text_linear(text_feature)  # (B, 768)
        image_feature = self.image_linear(image_feature)  # (B, 768)
        
        # Unpack batch (compatible with both MyDataset (5 items) and ReasoningDataset (7 items))
        text_list, image_list, label_list, id_list, samples = batch[:5]
        
        # ResNet features
        features, pos = self.backbone(samples.to(device))
        src, mask = features[-1].decompose()  # (B, 2048, 7, 7)
        src = self.input_proj(src)  # (B, 768, 7, 7)
        pooled_features = self.pool(src)  # (B, 768, 1, 1)
        res_features = pooled_features.view(pooled_features.size(0), -1)  # (B, 768)
        
        # BERT features
        encoded_input = self.tokenizer(text_list, padding=True, truncation=True, return_tensors='pt')
        encoded_input = encoded_input.to(device)
        with torch.no_grad():
            outputs_bert = self.bert_model(**encoded_input)
            pooler_outputs = outputs_bert.pooler_output  # (B, 768)
        bert_text_features = self.txt(pooler_outputs)  # (B, 768)
        
        # SFIM: Semantic Feature Interaction Module
        image_t = self.imtxt_cross(tgt=res_features, memory=bert_text_features)  # Image queries Text
        text_im = self.imtxt_cross(tgt=bert_text_features, memory=res_features)  # Text queries Image
        
        # RCLM: Representation Contrastive Learning Module
        text_feature2 = self.txt2(torch.cat([text_feature, text_im], dim=-1))
        text_feature2 = text_feature2.unsqueeze(1)
        txt_cat = self.text_self(torch.stack([text_feature, text_im], dim=1))
        txt_output = self.text_cross(tgt=text_feature2, memory=txt_cat)
        txt_output = txt_output.squeeze(1)
        
        image_feature2 = self.vis2(torch.cat([image_feature, image_t], dim=-1))
        image_feature2 = image_feature2.unsqueeze(1)
        image_cat = self.vis_self(torch.stack([image_feature, image_t], dim=1))
        image_output = self.vis_cross(tgt=image_feature2, memory=image_cat)
        image_output = image_output.squeeze(1)
        
        txt_out = self.cross_att(txt_output, image_output, image_output)
        image_out = self.cross_att(image_output, txt_output, txt_output)
        res_bert = 0.6 * image_out + 0.4 * txt_out
        
        # CLIP-View Feature Fusion
        cross_feature_text = self.cross_att(text_feature, image_feature, image_feature)
        cross_feature_image = self.cross_att(image_feature, text_feature, text_feature)
        fuse_feature = 0.7 * cross_feature_text + 0.3 * cross_feature_image
        
        # MuFFM: Multi-level Feature Fusion Module
        att = self.attetion_block(torch.cat([fuse_feature, res_bert], dim=-1))
        fusion_feature = 0.5 * fuse_feature + 0.5 * (att * self.mlp_layer(torch.cat([fuse_feature, res_bert], dim=-1)))
        
        return fusion_feature  # (B, 768)
    
    def forward(
        self,
        inputs,
        batch,
        labels=None,
        qwen_input_ids=None,
        qwen_attention_mask=None,
        qwen_labels=None
    ):

        # Extract fusion features from System 1
        fusion_feature = self._perception_forward(inputs, batch)  # (B, 768)
        
        if self.mode == 'perception':
            # Direct classification
            logits_fuse = self.classifier_fuse(fusion_feature)
            fuse_score = F.softmax(logits_fuse, dim=-1)
            
            outputs = (fuse_score,)
            if labels is not None:
                loss_fuse = self.loss_fct(logits_fuse, labels)
                outputs = (loss_fuse,) + outputs
            return outputs
        
        elif self.mode == 'alignment':
            # Project features to Qwen space
            feature_embeddings = self.projector(fusion_feature)  # (B, 3584)
            
            # Forward through Qwen for LM loss
            if qwen_input_ids is not None and qwen_labels is not None:
                outputs = self.qwen_agent(
                    feature_embeddings=feature_embeddings,
                    input_ids=qwen_input_ids,
                    attention_mask=qwen_attention_mask,
                    labels=qwen_labels
                )
                return (outputs.loss,)
            else:
                # Inference mode for alignment
                text_list = batch[0]
                generated = self.qwen_agent.generate(feature_embeddings, text_list)
                return generated
        
        elif self.mode == 'agent':
            # Full agent mode with LoRA
            feature_embeddings = self.projector(fusion_feature)
            
            if qwen_input_ids is not None and qwen_labels is not None:
                # Training
                outputs = self.qwen_agent(
                    feature_embeddings=feature_embeddings,
                    input_ids=qwen_input_ids,
                    attention_mask=qwen_attention_mask,
                    labels=qwen_labels
                )
                return (outputs.loss,)
            else:
                # Inference
                text_list = batch[0]
                generated_texts = self.qwen_agent.generate(feature_embeddings, text_list)
                
                # Parse predictions
                predictions = []
                explanations = []
                for text in generated_texts:
                    pred, expl = self.qwen_agent.parse_prediction(text)
                    predictions.append(pred)
                    explanations.append(expl)
                
                return predictions, explanations
        
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
    
    def get_fusion_feature(self, inputs, batch) -> Tensor:

        return self._perception_forward(inputs, batch)
    
    def set_mode(self, mode: str):

        assert mode in ['perception', 'alignment', 'agent']
        self.mode = mode
    
    def freeze_perception(self):
        """Freeze all System 1 parameters"""
        for name, param in self.named_parameters():
            if 'projector' not in name and 'qwen_agent' not in name:
                param.requires_grad = False
    
    def freeze_qwen_backbone(self):
        """Freeze Qwen backbone (keep LoRA trainable)"""
        if self.qwen_agent is not None:
            for name, param in self.qwen_agent.named_parameters():
                if 'lora' not in name.lower():
                    param.requires_grad = False
    
    def unfreeze_projector(self):
        """Unfreeze projector parameters"""
        if self.projector is not None:
            for param in self.projector.parameters():
                param.requires_grad = True
    
    def save_checkpoint(self, path: str, save_qwen: bool = False):

        state_dict = {}
        
        # Save perception components
        for name, param in self.named_parameters():
            if 'qwen_agent' not in name:
                state_dict[name] = param.data
        
        torch.save(state_dict, path)
        
        # Save LoRA adapter separately
        if self.qwen_agent is not None and self.mode == 'agent':
            lora_path = path.replace('.pth', '_lora')
            self.qwen_agent.save_lora_adapter(lora_path)
    
    def load_checkpoint(self, path: str, load_lora: bool = False, lora_path: str = None):

        state_dict = torch.load(path, map_location='cpu')
        
        # Load perception components
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        print(f"Loaded checkpoint. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
        
        # Load LoRA if specified
        if load_lora and lora_path and self.qwen_agent is not None:
            self.qwen_agent.load_lora_adapter(lora_path)
