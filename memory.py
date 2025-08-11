# %% # Implements Neural Memory Block
from einops import rearrange
import torch as t
from torch.nn import Sequential, Module, Linear, SiLU
from beartype import beartype
from jaxtyping import Float, jaxtyped
import torch.nn.functional as F
from torch.func import grad, functional_call
from tensordict import TensorDict


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
    def __init__(self, dim: int = 512, memory_depth: int = 2, lr=1e-3, chunk_size=2):
        super().__init__()
        self.W_Q = Linear(dim, dim, bias=False)
        self.W_KV = Linear(dim, dim * 2, bias=False)
        self.memory_mlp = MLP(dim, memory_depth)
        self.lr = lr
        self.chunk_size = chunk_size

    @t.no_grad()
    def _sgd_step(self, params: TensorDict, grads: TensorDict) -> TensorDict:
        # Apply SGD update: params = params - lr * grads
        # This ensures we return a copy
        result_dict = {}
        for key, grad_tensor in grads.items():
            result_dict[key] = params[key] - self.lr * grad_tensor

        return TensorDict(result_dict)

    @jaxtyped(typechecker=beartype)
    def store(
        self,
        x: Float[t.Tensor, "batch seq dim"],
    ) -> None:
        """
        x_seq: [T, D] tensor (single sequence, no batching for simplicity)


        """
        B, S, D = x.shape
        k, v = self.W_KV(x).chunk(2, dim=-1)  # b,s,d

        cur_weights = TensorDict(dict(self.memory_mlp.named_parameters()))

        def memory_loss(
            curr_weights: TensorDict,  # loss w.r.t memory mlp weights
            k_t: Float[t.Tensor, "mb dim"],
            v_t: Float[t.Tensor, "mb dim"],
        ):
            y_t = functional_call(self.memory_mlp, dict(curr_weights), k_t)
            loss = F.mse_loss(y_t, v_t, reduction="mean")
            return loss

        grad_fn = grad(memory_loss)

        # chunk into mb
        k, v = map(
            lambda t: rearrange(t, "b (s c) d -> (b s) c d", c=self.chunk_size), (k, v)
        )

        for mb in range(k.shape[0]):
            k_t, v_t = k[mb], v[mb]
            surprise = grad_fn(cur_weights, k_t, v_t)
            cur_weights = self._sgd_step(cur_weights, surprise)

        self.memory_mlp.load_state_dict(cur_weights)

    @jaxtyped(typechecker=beartype)
    def retrieve(
        self,
        x: Float[t.Tensor, "batch seq dim"],
    ) -> Float[t.Tensor, "batch seq dim"]:
        B = x.shape[0]
        q = self.W_Q(x)
        ht = self.memory_mlp(rearrange(q, "b s ... -> (b s) ..."))
        return rearrange(ht, "(b s) ... -> b s ...", b=B)
