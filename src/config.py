from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class QwenConfig:
    """Qwen-2.5-7B model configuration"""
    model_path: str = ""
    hidden_size: int = 3584  # Qwen-2.5-7B hidden dimension
    max_length: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    

@dataclass
class LoRAConfig:
    """LoRA (Low-Rank Adaptation) configuration"""
    r: int = 16  # LoRA rank
    alpha: int = 32  # LoRA alpha
    dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj"
    ])
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


@dataclass
class ProjectorConfig:
    """Feature Projector configuration"""
    input_dim: int = 768  # Fusion feature dimension
    hidden_dim: int = 2048  # MLP hidden dimension
    output_dim: int = 3584  # Qwen embedding dimension
    dropout: float = 0.1


@dataclass
class Phase1Config:
    """Phase I: Perception Pre-training configuration"""
    # Training settings
    num_epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 5e-4
    clip_learning_rate: float = 1e-6
    weight_decay: float = 0.05
    warmup_proportion: float = 0.2
    
    # Frozen modules
    freeze_bert: bool = True
    freeze_resnet: bool = True
    freeze_qwen: bool = True
    freeze_projector: bool = True
    
    # Output
    output_dir: str = "../output_dir/phase1"
    checkpoint_name: str = "perception_best.pth"


@dataclass
class Phase2Config:
    """Phase II: Projector Alignment configuration"""
    # Training settings
    num_epochs: int = 5
    batch_size: int = 16
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_proportion: float = 0.1
    
    # Frozen modules
    freeze_perception: bool = True  # Freeze entire System 1
    freeze_qwen: bool = True  # Freeze Qwen backbone
    
    # Input checkpoint
    perception_checkpoint: str = "../output_dir/phase1/perception_best.pth"
    
    # Output
    output_dir: str = "../output_dir/phase2"
    checkpoint_name: str = "projector_aligned.pth"


@dataclass
class Phase3Config:
    """Phase III: Agent Instruction Tuning configuration"""
    # Training settings
    num_epochs: int = 3
    batch_size: int = 8
    projector_lr: float = 1e-4
    lora_lr: float = 2e-4
    weight_decay: float = 0.01
    warmup_proportion: float = 0.1
    gradient_accumulation_steps: int = 4
    
    # Frozen modules
    freeze_perception: bool = True  # Recommend freezing System 1
    
    # Input checkpoints
    perception_checkpoint: str = "../output_dir/phase1/perception_best.pth"
    projector_checkpoint: Optional[str] = "../output_dir/phase2/projector_aligned.pth"
    
    # Output
    output_dir: str = "../output_dir/phase3"
    checkpoint_name: str = "agent_tuned.pth"
    lora_adapter_name: str = "lora_adapter"


@dataclass
class AgentConfig:
    """Master configuration combining all sub-configurations"""
    qwen: QwenConfig = field(default_factory=QwenConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    projector: ProjectorConfig = field(default_factory=ProjectorConfig)
    phase1: Phase1Config = field(default_factory=Phase1Config)
    phase2: Phase2Config = field(default_factory=Phase2Config)
    phase3: Phase3Config = field(default_factory=Phase3Config)
    
    # Model mode: 'perception' | 'alignment' | 'agent'
    mode: str = "perception"
    
    # Device settings
    device: str = "cuda"
    mixed_precision: bool = True
    
    # Original RCLMuFN settings (for compatibility)
    text_size: int = 512
    image_size: int = 768
    label_number: int = 2
    layers: int = 3
    simple_linear: bool = False
    dropout_rate: float = 0.1
    max_len: int = 77


# Prompt templates for Qwen reasoning
SARCASM_ANALYSIS_PROMPT = """User: Analyze this multimodal content for sarcasm detection.

Text: {text}
[Visual Feature Embedding]

Please analyze:
1. What is the sentiment/tone of the text?
2. What does the image likely convey?
3. Is there any contradiction between text and image?
4. Is this sarcastic?
"""
