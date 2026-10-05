import torch
import torch.nn as nn


class FeatureProjector(nn.Module):

    def __init__(self, input_dim=768, hidden_dim=2048, output_dim=3584, dropout=0.1):

        super(FeatureProjector, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # MLP projector
        self.projector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier initialization"""
        for module in self.projector:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, fusion_features):

        # Ensure input is 2D
        if fusion_features.dim() == 1:
            fusion_features = fusion_features.unsqueeze(0)
        
        # Project to LLM space
        feature_embeddings = self.projector(fusion_features)
        
        return feature_embeddings
    
    def get_soft_prompt(self, fusion_features, num_tokens=1):

        batch_size = fusion_features.size(0)
        
        # Project features
        embeddings = self.forward(fusion_features)  # (batch_size, output_dim)
        
        # Expand to multiple tokens if needed
        if num_tokens > 1:
            # Use learnable token expansion or simple repeat
            soft_prompts = embeddings.unsqueeze(1).expand(-1, num_tokens, -1)
        else:
            soft_prompts = embeddings.unsqueeze(1)  # (batch_size, 1, output_dim)
        
        return soft_prompts


class MultiTokenProjector(nn.Module):

    
    def __init__(self, input_dim=768, hidden_dim=2048, output_dim=3584, 
                 num_tokens=4, dropout=0.1):

        super(MultiTokenProjector, self).__init__()
        
        self.num_tokens = num_tokens
        self.output_dim = output_dim
        
        # Project to expanded dimension
        expanded_dim = output_dim * num_tokens
        
        self.projector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, expanded_dim)
        )
        
        # Layer norm for each token
        self.layer_norm = nn.LayerNorm(output_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        for module in self.projector:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, fusion_features):

        batch_size = fusion_features.size(0)
        
        # Project to expanded dimension
        projected = self.projector(fusion_features)  # (batch_size, num_tokens * output_dim)
        
        # Reshape to multiple tokens
        soft_prompts = projected.view(batch_size, self.num_tokens, self.output_dim)
        
        # Apply layer norm
        soft_prompts = self.layer_norm(soft_prompts)
        
        return soft_prompts


def build_projector(config):

    if hasattr(config, 'num_tokens') and config.num_tokens > 1:
        return MultiTokenProjector(
            input_dim=config.input_dim,
            hidden_dim=config.hidden_dim,
            output_dim=config.output_dim,
            num_tokens=config.num_tokens,
            dropout=config.dropout
        )
    else:
        return FeatureProjector(
            input_dim=config.input_dim,
            hidden_dim=config.hidden_dim,
            output_dim=config.output_dim,
            dropout=config.dropout
        )
