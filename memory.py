# %% # Implements Neural Memory Block
import torch as t
from torch.nn import Sequential, Module, Linear, SiLU
from beartype import beartype
from jaxtyping import Float, jaxtyped


# %%
class MLP(Module):
    @beartype
    def __init__(self, dim: int, depth: int):
        super().__init__()
        layers = []
        for i in range(depth):
            layers.append(Linear(dim, dim))
            if i < depth - 1:
                layers.append(SiLU())
        self.layers = Sequential(*layers)

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        x: Float[t.Tensor, "batch dim"],  # type: ignore
    ) -> Float[t.Tensor, "batch dim"]:  # type: ignore
        return self.layers(x)


class Memory(Module):
    @beartype
    def __init__(self):
        super().__init__()

    def store(self, x):
        pass

    def retrieve(self, x):
        pass

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        x: Float[t.Tensor, "batch dim"],  # type: ignore
    ) -> Float[t.Tensor, "batch dim"]:  # type: ignore
        return x
