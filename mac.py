from einops import repeat
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module, Linear
from dataclasses import dataclass
from beartype import beartype
from data.shakespeare.dataloader import get_dataloaders
from memory import Memory


@dataclass(frozen=True)
class MACConfig:
    d_model: int = 512
    seq_len: int = 1024
    num_tokens: int = 256  # Vocabulary size
    num_hidden_layers: int = 2
    num_heads: int = 1
    num_longterm_mem_tokens: int = 16


class MacAttention(Module):
    """
    Input:  x  [B, S, D]
    Output: y  [B, S, D]
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        persistent_tokens: int = 16,
        attn_dropout: float = 0.0,
        ff_mult: int = 4,
        use_readback_gate: bool = True,
        memory: Memory | None = None,
    ):
        super().__init__()

        self.dim = dim
        self.n_heads = n_heads
        self.P = persistent_tokens
        self.use_readback_gate = use_readback_gate

        self.W_QKV = nn.Linear(dim, dim * 3)

        # persistent tokens (learnable, shared across batch)
        # shape: [2, P, D], will be repeated along batch at runtime. inlcudes kv
        self.persistent = nn.Parameter(t.randn(2, persistent_tokens, dim))

        self.ln_attn_in = nn.LayerNorm(dim)

        # optional read-back + gate
        if use_readback_gate:
            self.ln_gate = nn.LayerNorm(dim)
            self.gate = nn.Linear(
                dim * 2, dim
            )  # produces "mix" vector; could be sigmoid gate or linear mix
            self.proj_out = nn.Linear(dim, dim)

        # feed-forward (Transformer-style)
        self.ln_mlp = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_mult * dim),
            nn.SiLU(),
            nn.Linear(ff_mult * dim, dim),
        )

        # attach (shared or per-layer) memory module
        assert memory is not None, "MACBlock requires a MemoryModule instance"
        self.memory = memory

    def _build_causal_mask(self, L: int, device) -> t.Tensor:
        """
        Standard causal mask for multihead attention (batch_first=True).
        attn_mask shape: [B*n_heads, L, L] or [L, L]. We'll use [L, L].
        Lower triangle is allowed.
        """
        mask = t.full((L, L), float("-inf"), device=device)
        mask = t.triu(mask, diagonal=1)  # upper triangle is -inf, lower incl diag is 0
        return mask

    def forward(self, x: t.Tensor) -> t.Tensor:
        """
        x: [B, S, D]
        lt_seq: [S,D]
        returns y: [B, S, D]
        """
        B, S, D = x.shape

        x = self.ln_attn_in(x)

        # prepend lt
        ht = self.memory.retrieve(x)  # b s d
        z = t.concat((ht, x), dim=1)  # b 2s d

        q, k, v = self.W_QKV(z).chunk(3, dim=-1)

        # pm only has k v. doesn't query other tokens
        pmk, pmv = repeat(self.persistent, "kv p d -> kv b p d", b=B).chunk(2, dim=0)
        pmk, pmv = (
            pmk.squeeze(0),
            pmv.squeeze(0),
        )  # squeeze the chunk dim. both should be b,p,d

        k = t.cat((pmk, k), dim=-2)
        v = t.cat((pmv, v), dim=-2)

        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        y_cur = attn_out[:, -S:, :]  # remove lt

        self.memory.store(y_cur)

        if self.use_readback_gate:
            m_now = self.memory.retrieve(y_cur)  # b,s,d
            z = t.cat([m_now, self.ln_gate(x)], dim=-1)  # b, s, 2D
            mixed = self.gate(z)  # b,s,d
            x = x + self.proj_out(mixed)

        x = x + self.ff(self.ln_mlp(x))  # [B, S, D]

        return x


class MACTransformer(Module):
    def __init__(self, config: MACConfig):
        super().__init__()
        self.d_model = config.d_model
        self.num_tokens = config.num_tokens
        self.num_hidden_layers = config.num_hidden_layers
        self.max_seq_len = config.seq_len
        self.has_embed_unembed = True  # This model has embedding and unembedding layers

        self.layers = nn.ModuleList(
            [
                MacAttention(
                    dim=config.d_model,
                    n_heads=config.num_heads,
                    memory=Memory(),
                )
                for _ in range(config.num_hidden_layers)
            ]
        )

        self.token_embedding = nn.Embedding(config.num_tokens, config.d_model)
        self.absolute_pos_embedding = nn.Embedding(config.seq_len, config.d_model)
        self.output_norm = nn.RMSNorm(config.d_model)
        self.to_vocab = Linear(config.d_model, config.num_tokens)

        self.num_longterm_mem_tokens = config.num_longterm_mem_tokens
        self.long_term_memory_seq = nn.Parameter(
            t.randn(config.num_longterm_mem_tokens, config.d_model)
        )

    @beartype
    def forward(self, x: t.Tensor):
        B, S = x.shape
        x = self.token_embedding(x)
        position_ids = t.arange(S, device=x.device)
        x = x + self.absolute_pos_embedding(position_ids)

        for layer in self.layers:
            x = layer(x)

        x = self.output_norm(x)
        logits = self.to_vocab(x)
        return logits

    def generate(self, prime, seq_len, temperature=1.0, filter_thres=0.9, **kwargs):
        assert self.has_embed_unembed
        assert temperature >= 0.0

        n = prime.shape[1]
        out = prime

        for _ in range(seq_len):
            logits = self.forward(out[:, -self.max_seq_len :], **kwargs)

            filtered_logits = top_k(logits[:, -1], thres=filter_thres)

            if temperature == 0.0:
                sampled = filtered_logits.argmax(dim=-1, keepdim=True)
            else:
                probs = F.softmax(filtered_logits / temperature, dim=-1)
                sampled = t.multinomial(probs, 1)

            out = t.cat((out, sampled), dim=-1)

        return out[:, n:]


def top_k(logits, thres=0.9):
    k = int((1 - thres) * logits.shape[-1])
    val, ind = t.topk(logits, k)
    probs = t.full_like(logits, float("-inf"))
    probs.scatter_(1, ind, val)
    return probs


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
