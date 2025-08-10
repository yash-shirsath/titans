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
        x: Float[t.Tensor, "batch seq dim"],
    ) -> Float[t.Tensor, "batch seq dim"]:
        # Reshape to (batch * seq, dim) for processing through linear layers
        batch_size, seq_len, dim = x.shape
        x_reshaped = x.view(-1, dim)  # (batch * seq, dim)

        # Process through MLP layers
        output = self.layers(x_reshaped)  # (batch * seq, dim)

        # Reshape back to original shape
        return output.view(batch_size, seq_len, dim)  # (batch, seq, dim)


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
        x: Float[t.Tensor, "batch seq dim"],  # type: ignore
    ) -> Float[t.Tensor, "batch seq dim"]:  # type: ignore
        # Project input to query: q_t = x_t W_Q
        q_t = self.W_Q(x)

        # Retrieve memory via forward pass without weight update: y_t = M*(q_t)
        y_t = self.memory_module(q_t)

        return y_t
