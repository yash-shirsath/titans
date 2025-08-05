# %% # Implements Neural Memory Block
import torch as t
from torch.nn import Sequential, Module, Linear, ReLU
from beartype import beartype
from jaxtyping import Float, jaxtyped


# %%
@beartype
class MLP(Module):
    def __init__(self, dim: int, depth: int):
        super().__init__()
        layers = []
        for i in range(depth):
            layers.append(Linear(dim, dim))
            if i < depth - 1:
                layers.append(ReLU())
        self.layers = Sequential(*layers)

    @jaxtyped
    def forward(self, x: Float[t.Tensor, "batch dim"]) -> Float[t.Tensor, "batch dim"]:
        return self.layers(x)


@beartype
class Memory(Module):
    def __init__(self):
        pass

    def store(self, x):
        pass

    def retrieve(self, x):
        pass

    @jaxtyped
    def forward(self, x: Float[t.Tensor, "batch dim"]) -> Float[t.Tensor, "batch dim"]:
        """
        A placeholder for the memory's forward pass.
        This should likely involve retrieving from memory.
        """
        # For now, we'll just return the input
        return x
