# %% # Implements Neural Memory Block
from re import T
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
    def __init__(self, dim: int = 512, memory_depth: int = 2, lr=1e-3):
        super().__init__()
        self.W_Q = Linear(dim, dim, bias=False)
        self.W_KV = Linear(dim, dim * 2, bias=False)
        self.memory_module = MLP(dim, memory_depth)
        self.lr = lr

    @t.no_grad()
    def _sgd_step(self):
        for p in self.memory_module.parameters():
            if p.grad is not None:
                p -= self.lr * p.grad
                p.grad.zero_()  # type ignore

    @jaxtyped(typechecker=beartype)
    def store(
        self,
        x: Float[t.Tensor, "batch seq dim"],
    ) -> None:
        B, S, D = x.shape
        k, v = self.W_KV(x).chunk(2, dim=-1)

        """
        x_seq: [T, D] tensor (single sequence, no batching for simplicity)

        Implements Eq. (8) with loss from Eq. (12) and k_t, v_t from Eq. (11):
            k_t = x_t W_k
            v_t = x_t W_v
            L_t = || M(k_t) - v_t ||^2
            θ <- θ - η * ∇_θ L_t
        """

        for s in range(S):
            x_t = x[:, s : s + 1]  # [1, D]
            with T.no_grad():
                k_t = self.W_k(x_t)  # Eq. (11)
                v_t = self.W_v(x_t)  # Eq. (11)

            # Forward through memory M(·; θ)
            y_t = self.M(k_t)  # prediction of value

            # Associative-memory loss (Eq. 12)
            loss = F.mse_loss(y_t, v_t, reduction="mean")

            # Backprop into θ only; then naive SGD step (Eq. 8)
            for p in self.M.parameters():
                if p.grad is not None:
                    p.grad.zero_()
            loss.backward()  # ∇_θ L(M_{t-1}; x_t)
            self._sgd_step()  # θ_t = θ_{t-1} - η ∇_θ L

    @jaxtyped(typechecker=beartype)
    def retrieve(
        self,
        x: Float[t.Tensor, "batch seq dim"],
    ) -> Float[t.Tensor, "batch seq dim"]:
        B = x.shape[0]
        q = self.W_Q(x)
        ht = self.memory_module(rearrange(q, "b s ... -> (b s) ..."))
        return rearrange(ht, "(b s) ... -> b s ...", b=B)
