# GPT2 language model architecture (Part 1.A), adapted from LAB 04.
# The `dropout` arguments are already threaded through every layer so that
# exercise 2 (add dropout) can be turned on/off just by changing the
# constructor argument, without touching the training code.

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """Masked multi-head self-attention (GPT2 style, single stack of Q/K/V projections per head)."""

    def __init__(self, d_model, n_heads, dropout=0.1):
        """
        Args:
            d_model: embedding size of the model.
            n_heads: number of attention heads (must divide d_model).
            dropout: dropout probability (exercise 1.A.2: apply after the
                attention weights and after the output projection).
        """
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.h_dim = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        self.out_proj = nn.Linear(d_model, d_model)
        # TODO (exercise 1.A.2): self.attn_dropout = nn.Dropout(dropout), applied
        # to `attn` right after the softmax.
        # TODO (exercise 1.A.2): self.proj_dropout = nn.Dropout(dropout), applied
        # to `y` right after self.out_proj.

    def forward(self, x, mask):
        """
        Args:
            x: input embeddings, shape (B, L, d_model).
            mask: causal mask, shape broadcastable to (B, n_heads, L, L).
        Returns:
            Attention output, shape (B, L, d_model).
        """
        B, L, d_model = x.size()

        q = self.w_q(x)  # (B, L, d_model)
        k = self.w_k(x)  # (B, L, d_model)
        v = self.w_v(x)  # (B, L, d_model)

        # reshape to (B, n_heads, L, h_dim) to process each head independently
        q = q.view(B, L, self.n_heads, self.h_dim).transpose(1, 2)
        k = k.view(B, L, self.n_heads, self.h_dim).transpose(1, 2)
        v = v.view(B, L, self.n_heads, self.h_dim).transpose(1, 2)

        # (B, n_heads, L, h_dim) @ (B, n_heads, h_dim, L) -> (B, n_heads, L, L)
        similarity = q @ k.transpose(-2, -1)
        similarity = similarity * (1 / torch.sqrt(torch.tensor(self.h_dim)))

        # future tokens get -inf similarity so softmax zeroes them out
        similarity = similarity.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(similarity, dim=-1)
        # TODO (exercise 1.A.2): attn = self.attn_dropout(attn)

        y = attn @ v  # (B, n_heads, L, h_dim)
        y = y.transpose(1, 2).contiguous().view(B, L, d_model)  # concat heads
        y = self.out_proj(y)
        # TODO (exercise 1.A.2): y = self.proj_dropout(y)

        return y


class FeedForward(nn.Module):
    """Position-wise feed-forward block (GELU activation, GPT2 style)."""

    def __init__(self, d_model, hidden_dim, dropout=0.1):
        """
        Args:
            d_model: embedding size of the model.
            hidden_dim: inner dimension of the feed-forward layer.
            dropout: dropout probability (exercise 1.A.2: apply after the
                last linear layer).
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model),
            # TODO (exercise 1.A.2): nn.Dropout(dropout) as the last layer here.
        )

    def forward(self, x):
        """Applies the feed-forward transformation independently to every token."""
        return self.net(x)


class TransformerBlock(nn.Module):
    """Pre-LayerNorm transformer block: LN -> attention -> residual, LN -> FF -> residual."""

    def __init__(self, d_model, n_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, ff_dim, dropout)

    def forward(self, x, mask):
        """
        Args:
            x: input embeddings, shape (B, L, d_model).
            mask: causal mask passed through to the attention layer.
        Returns:
            Updated embeddings, shape (B, L, d_model).
        """
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.ff(self.ln2(x))
        return x


class GPT2(nn.Module):
    """Decoder-only GPT2 language model, predicting the next token at every position."""

    def __init__(
        self,
        vocab_size,
        pos_emb_size=1024,
        d_model=768,
        n_heads=12,
        num_layers=12,
        ff_dim=3072,
        dropout=0.1,
    ):
        """
        Args:
            vocab_size: size of the tokenizer's vocabulary.
            pos_emb_size: maximum sequence length supported by the positional embeddings.
            d_model: embedding size of the model.
            n_heads: number of attention heads per transformer block.
            num_layers: number of stacked transformer blocks.
            ff_dim: inner dimension of the feed-forward blocks.
            dropout: dropout probability (exercise 1.A.2: apply after summing
                token and positional embeddings).
        """
        super().__init__()
        self.pos_emb_size = pos_emb_size

        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(pos_emb_size, d_model)
        # TODO (exercise 1.A.2): self.embed_dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)
        # TODO (exercise 1.A.3, weight tying): self.lm_head.weight = self.token_embed.weight

        # causal mask: token i can only attend to tokens j <= i
        mask = torch.tril(torch.ones(pos_emb_size, pos_emb_size)).unsqueeze(0).unsqueeze(0)
        # not a learnable parameter, but must move with the model on .to(device)
        self.register_buffer("mask", mask)

    def forward(self, idx):
        """
        Args:
            idx: input token ids, shape (B, L).
        Returns:
            Logits over the vocabulary at every position, shape (B, L, vocab_size).
        """
        B, L = idx.shape
        # positions beyond pos_emb_size were never trained, so longer sequences would be invalid
        assert L <= self.pos_emb_size

        pos = torch.arange(L, device=idx.device)
        x = self.token_embed(idx) + self.pos_embed(pos)
        # TODO (exercise 1.A.2): x = self.embed_dropout(x)

        mask = self.mask[:, :, :L, :L]

        for block in self.blocks:
            x = block(x, mask)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        return logits  # (B, L, vocab_size)
