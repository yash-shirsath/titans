# %% # Implements Neural Memory Block
from einops import rearrange
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
        x: Float[t.Tensor, "batch dim"],
    ) -> Float[t.Tensor, "batch dim"]:
        return self.layers(x)


class Memory(Module):
    @beartype
    def __init__(self, dim: int = 512, memory_depth: int = 2):
        super().__init__()
        # Query projection matrix W_Q as mentioned before equation (15)
        self.W_Q = Linear(dim, dim, bias=False)
        # Memory module M - using MLP as specified in the paper
        self.memory_module = MLP(dim, memory_depth)

    def store(self, x):
        pass

    @jaxtyped(typechecker=beartype)
    def retrieve(
        self,
        x: Float[t.Tensor, "batch seq dim"],
    ) -> Float[t.Tensor, "batch seq dim"]:
        B = x.shape[0]
        q = self.W_Q(x)
        ht = self.memory_module(rearrange(q, "b s ... -> (b s) ..."))
        return rearrange(ht, "(b s) ... -> b s ...", b=B)
