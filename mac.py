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


class MACTransformer(Module):
    def __init__(self, config: MACConfig):
        super().__init__()
        self.d_model = config.d_model

    @beartype
    def forward(self, x: Float[t.Tensor, "batch seq"]) -> t.Tensor:
        return x


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
