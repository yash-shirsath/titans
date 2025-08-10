from einops import repeat
import torch as t
import torch.nn as nn
from torch.nn import Module, Linear, LayerNorm, Dropout, MultiheadAttention
from typing import Optional
from dataclasses import dataclass
from jaxtyping import Float
from beartype import beartype
from data.shakespeare.dataloader import get_dataloaders
from memory import Memory


@dataclass(frozen=True)
class MACConfig:
    d_model: int = 512
    seq_len: int = 1024
    num_tokens: int = 256  # Vocabulary size
    num_hidden_layers: int = 12
    num_heads: int = 1
    num_longterm_mem_tokens: int = 16

class MacAttention(Module):
    """
    Memory-as-a-Context block.
    Input:  x  [B, S, D]
    Output: y  [B, S, D]

    Steps (all inside the block):
      1) q = Wq(LN(x))                                  -> [B, S, D]
      2) m_hist = memory.retrieve(q)                     -> [B, S, D]
      3) seq = concat([persistent, m_hist, x], dim=1)    -> [B, P + S + S, D]
      4) y_all = Attn(seq) (causal on the concat)        -> [B, P + 2S, D]
      5) y_cur = take last S positions                   -> [B, S, D]
      6) memory.update(y_cur)
      7) (optional) read-back + gate; here we do a simple linear gate
      8) MLP + residual (standard transformer epilogue)
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

        # persistent tokens (learnable, shared across batch)
        # shape: [1, P, D], will be repeated along batch at runtime
        self.persistent = nn.Parameter(t.randn(1, persistent_tokens, dim) / dim**0.5)

        # query projection for memory read
        self.ln_q = nn.LayerNorm(dim)
        self.W_q = nn.Linear(dim, dim, bias=True)

        # in-block attention over [persistent || m_hist || x]
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=n_heads, dropout=attn_dropout, batch_first=True
        )
        self.ln_attn_in = nn.LayerNorm(dim)
        self.dropout_attn = nn.Dropout(attn_dropout)

        # optional read-back + gate
        if use_readback_gate:
            self.ln_gate = nn.LayerNorm(dim)
            self.gate = nn.Linear(dim * 2, dim)  # produces "mix" vector; could be sigmoid gate or linear mix
            self.proj_out = nn.Linear(dim, dim)

        # feed-forward (Transformer-style)
        self.ln_mlp = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_mult * dim),
            nn.SiLU(),
            nn.Linear(ff_mult * dim, dim),
        )
        self.dropout_ff = nn.Dropout(attn_dropout)

        # attach (shared or per-layer) memory module
        assert memory is not None, "MACBlock requires a MemoryModule instance"
        self.memory = memory

    def _build_causal_mask(self, B: int, L: int, device) -> t.Tensor:
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
        returns y: [B, S, D]
        """
        B, S, D = x.shape
        assert D == self.dim

        x = self.ln_q(x)

        # 1) queries for memory read
        #    q: [B, S, D]
        q = self.W_q(x)

        # 2) retrieve historical context
        #    m_hist: [B, S, D]
        m_hist = self.memory.retrieve(q)

        # 3) concat persistent + m_hist + x
        #    P_b: [B, P, D]
        P_b = self.persistent.repeat(B, 1, 1)
        #    seq: [B, P + S + S, D]
        seq = t.cat([P_b, m_hist, x], dim=1)

        # 4) attention over the concatenated sequence (causal)
        #    y_all: [B, P + 2S, D]
        L = self.P + 2 * S
        attn_mask = self._build_causal_mask(B=B, L=L, device=x.device)  # [L, L]
        seq_norm = self.ln_attn_in(seq)
        y_all, _ = self.attn(seq_norm, seq_norm, seq_norm, attn_mask=attn_mask)
        y_all = self.dropout_attn(y_all)

        # 5) take last S positions to align with input steps
        #    y_cur: [B, S, D]
        y_cur = y_all[:, -S:, :]

        # residual after attention context injection
        x = x + y_cur

        # 6) update memory with what to keep (post-attn features)
        self.memory.store(y_cur)

        # 7) optional read-back + gate
        if self.use_readback_gate:
            # m_now: [B, S, D]
            m_now = self.memory.retrieve(y_cur)
            z = t.cat([self.ln_gate(x), m_now], dim=-1)   # [B, S, 2D]
            mixed = self.gate(z)                               # [B, S, D]
            x = x + self.proj_out(mixed)                      # residual

        # 8) MLP + residual
        x = x + self.dropout_ff(self.ff(self.ln_mlp(x)))      # [B, S, D]

        return x  # [B, S, D]

class MACTransformer(Module):
    def __init__(self, config: MACConfig):
        super().__init__()
        self.d_model = config.d_model
        self.num_tokens = config.num_tokens
        self.num_hidden_layers = config.num_hidden_layers

        self.layers = nn.ModuleList([MacAttention(
            dim=config.d_model,
            n_heads=config.num_heads,
            memory=Memory(),
        ) for _ in range(config.num_hidden_layers)])


        self.token_embedding = nn.Embedding(config.num_tokens, config.d_model)
        self.absolute_pos_embedding = nn.Embedding(config.seq_len, config.d_model)
        self.output_norm = nn.RMSNorm(config.d_model)
        self.to_vocab = Linear(config.d_model, config.num_tokens)

        self.num_longterm_mem_tokens = config.num_longterm_mem_tokens
        self.long_term_memory_seq = nn.Parameter(t.randn(config.num_longterm_mem_tokens, config.d_model)) 

    @beartype  
    def forward(self, x: t.Tensor) -> t.Tensor:
        B,S = x.shape
        x = self.token_embedding(x)  
        position_ids = t.arange(S, device=x.device)
        x = x + self.absolute_pos_embedding(position_ids)

        x = self.insert_lt_mems(x)

        for layer in self.layers:
            x = layer(x)

        x = self.remove_lt_mems(x)

        x = self.output_norm(x)
        logits = self.to_vocab(x) 
        return logits

    def insert_lt_mems(self, x: t.Tensor) -> t.Tensor:
        B = x.shape[0]
        h_t = repeat(self.long_term_memory_seq, 'S D -> B S D', B = B)
        x = t.cat((h_t, x), dim = -2)
        return x
    
    def view_lt_mems(self, x: t.Tensor) -> t.Tensor:
        return x[:, :self.num_longterm_mem_tokens, :]

    def remove_lt_mems(self, x: t.Tensor) -> t.Tensor:
        return x[:, self.num_longterm_mem_tokens:, :]


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
