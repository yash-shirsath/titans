import torch as t
from torch.nn import Module, Linear, LayerNorm, Dropout, MultiheadAttention
from typing import Optional
from dataclasses import dataclass
from jaxtyping import Float
from beartype import beartype
from data.shakespeare.dataloader import get_dataloaders


@dataclass(frozen=True)
class MACConfig:
    d_model: int = 512
    seq_len: int = 1024
    num_tokens: int = 256  # Vocabulary size


class MACTransformer(Module):
    def __init__(self, config: MACConfig):
        super().__init__()
        self.d_model = config.d_model
        self.num_tokens = config.num_tokens
        
        # Simple embedding and output layers for basic functionality
        self.token_embedding = t.nn.Embedding(config.num_tokens, config.d_model)
        self.output_projection = Linear(config.d_model, config.num_tokens)

    @beartype  
    def forward(self, x: t.Tensor) -> t.Tensor:
        # Basic implementation: embed tokens and project to logits
        embedded = self.token_embedding(x)  # (batch, seq, d_model)
        logits = self.output_projection(embedded)  # (batch, seq, num_tokens)
        return logits


if __name__ == "__main__":
    config = MACConfig(d_model=512, seq_len=64)
    model = MACTransformer(config)
    model.eval()

    train_loader, _ = get_dataloaders(
        seq_len=config.seq_len, batch_size=4, random_sampling=True
    )

    batch = next(iter(train_loader))
    input_tokens = batch[:, :-1]
    with t.no_grad():
        output = model(input_tokens)
    print(f"Output shape: {output.shape}")
    print("✅ Successfully passed batch through MAC model!")
