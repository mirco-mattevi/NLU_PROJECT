# GPT2-based joint intent classification and slot filling model (Part 2.A),
# adapted from LAB 05. Same backbone as LM/part_A's GPT2, but the output
# layer is replaced by a slot tagger (per-token) and an intent classifier
# (pooled from the CLS token appended at the end of every utterance).

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """Masked multi-head self-attention (GPT2 style, single stack of Q/K/V projections per head)."""

    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.h_dim = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, mask):
        """
        Args:
            x: input embeddings, shape (B, L, d_model).
            mask: causal mask, shape broadcastable to (B, n_heads, L, L).
        Returns:
            Attention output, shape (B, L, d_model).
        """
        B, L, d_model = x.size()

        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        q = q.view(B, L, self.n_heads, self.h_dim).transpose(1, 2)
        k = k.view(B, L, self.n_heads, self.h_dim).transpose(1, 2)
        v = v.view(B, L, self.n_heads, self.h_dim).transpose(1, 2)

        similarity = q @ k.transpose(-2, -1)
        similarity = similarity * (1 / torch.sqrt(torch.tensor(self.h_dim)))
        similarity = similarity.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(similarity, dim=-1)

        y = attn @ v
        y = y.transpose(1, 2).contiguous().view(B, L, d_model)
        y = self.out_proj(y)

        return y


class FeedForward(nn.Module):
    """Position-wise feed-forward block (GELU activation, GPT2 style)."""

    def __init__(self, d_model, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model),
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
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.ff(self.ln2(x))
        return x


class GPT2(nn.Module):
    """
    Decoder-only GPT2 backbone with two output heads:
    - slot_out: per-token sequence labeling (slot filling).
    - intent_out: sequence classification from the CLS token appended at the
      end of every utterance (the last valid position of each sequence).
    """

    def __init__(
        self,
        vocab_size,
        slots_size,
        n_intents,
        pos_emb_size=1024,
        d_model=768,
        n_heads=12,
        num_layers=12,
        ff_dim=3072,
        dropout=0.1,
    ):
        """
        Args:
            vocab_size: size of the word vocabulary (see utils.Lang).
            slots_size: number of slot labels (including pad/cls).
            n_intents: number of intent labels.
            pos_emb_size: maximum sequence length supported by the positional embeddings.
            d_model, n_heads, num_layers, ff_dim: transformer backbone hyperparameters.
            dropout: dropout probability (exercise 2.A.2: apply before the
                final output layers, i.e. right before slot_out/intent_out).
        """
        super().__init__()
        self.pos_emb_size = pos_emb_size

        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(pos_emb_size, d_model)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model)
        # TODO (exercise 2.A.2): self.dropout = nn.Dropout(dropout), applied to
        # `x` right after self.ln_f, before computing slot_out/intent_out.

        # slot outputs (sequence labeling): one slot label per input token
        self.slot_out = nn.Linear(d_model, slots_size)
        # intent output (sequence classification): one intent label per utterance
        self.intent_out = nn.Linear(d_model, n_intents)

        # causal mask: (sub)token i can only attend to (sub)tokens j <= i
        mask = torch.tril(torch.ones(pos_emb_size, pos_emb_size)).unsqueeze(0).unsqueeze(0)
        self.register_buffer("mask", mask)

    def forward(self, idx, seq_lens):
        """
        Args:
            idx: input token ids, shape (B, L). The last valid position of
                each sequence holds the CLS token.
            seq_lens: actual (unpadded) length of each sequence in the batch, shape (B,).
        Returns:
            Tuple (slots, intent):
                slots: per-token logits, shape (B, L, slots_size). The
                    prediction at the CLS position is meaningless and must be
                    ignored (its ground truth is the pad label).
                intent: per-utterance logits, shape (B, n_intents).
        """
        B, L = idx.shape
        assert L <= self.pos_emb_size

        pos = torch.arange(L, device=idx.device)
        x = self.token_embed(idx) + self.pos_embed(pos)

        mask = self.mask[:, :, :L, :L]

        for block in self.blocks:
            x = block(x, mask)

        x = self.ln_f(x)
        # TODO (exercise 2.A.2): x = self.dropout(x)

        slots = self.slot_out(x)

        # intent is predicted from each sequence's CLS token (its last valid position)
        cls_tokens = torch.stack([x[i, seq_lens[i] - 1] for i in range(x.shape[0])])
        intent = self.intent_out(cls_tokens)

        return slots, intent  # (B, L, slots_size), (B, n_intents)
