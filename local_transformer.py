from local_attention import LocalTransformer
import torch
from dataclasses import dataclass
from typing import Optional

@dataclass
class LocalTransformerConfig:
    """Configuration for LocalTransformer model."""
    num_tokens: int = 256
    dim: int = 512
    depth: int = 6
    max_seq_len: int = 8192
    causal: bool = True
    local_attn_window_size: int = 256
    device: str = 'cuda'

def get_local_transformer(config: Optional[LocalTransformerConfig] = None):
    """
    Create and return a LocalTransformer model.
    
    Args:
        config (LocalTransformerConfig): Configuration object for the transformer.
                                       If None, uses default configuration.
    
    Returns:
        LocalTransformer: Configured transformer model
    """
    if config is None:
        config = LocalTransformerConfig()
    
    model = LocalTransformer(
        num_tokens=config.num_tokens,
        dim=config.dim,
        depth=config.depth,
        max_seq_len=config.max_seq_len,
        causal=config.causal,
        local_attn_window_size=config.local_attn_window_size
    )
    
    if config.device and torch.cuda.is_available() and config.device == 'cuda':
        model = model.cuda()
    elif config.device:
        model = model.to(config.device)
    
    return model

# Example usage
if __name__ == "__main__":
    model = get_local_transformer()
    x = torch.randint(0, 256, (1, 8192)).cuda()
    logits = model(x)  # (1, 8192, 256)
    print(logits.shape)