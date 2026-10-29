# GPT2-based joint intent classification and slot filling model.
# Same as part 1A but output layer replaced by a slot tagger (per-token)
# and an intent classifier

import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    """
    Masked multi-head self-attention.
    """

    def __init__(self, d_model, n_heads, dropout=0.1):
        """
        Args:
            d_model: embedding size of the model.
            n_heads: number of attention heads.
            dropout: dropout probability.
        """
        super().__init__()
        assert d_model % n_heads == 0 # dimension must split evenly across heads
        self.n_heads = n_heads
        self.h_dim = d_model // n_heads

        # 4 linear layers: Query, Key, Value, and output projection
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, mask):
        """
        Args:
            x: input embeddings, shape (B, L, d_model).
            mask: causal mask.
        Returns:
            Attention output, shape (B, L, d_model).
        """
        B, L, d_model = x.size()

        q = self.w_q(x) # (B, L, d_model)
        k = self.w_k(x) # (B, L, d_model)
        v = self.w_v(x) # (B, L, d_model)

        # split into heads, reshape to (B, n_heads, L, h_dim)
        q = q.view(B, L, self.n_heads, self.h_dim).transpose(1, 2)
        k = k.view(B, L, self.n_heads, self.h_dim).transpose(1, 2)
        v = v.view(B, L, self.n_heads, self.h_dim).transpose(1, 2)

        # dot-product similarity between each head's Q and K matrices
        # similarity[i][j] is the similarity between the j-th token and the i-th token for each head
        similarity = q @ k.transpose(-2, -1)
        similarity = similarity * (1 / torch.sqrt(torch.tensor(self.h_dim)))

        # future tokens get -inf similarity so softmax zeroes them out (does not look at future tokens)
        similarity = similarity.masked_fill(mask == 0, float('-inf')) # GPT2 mask = lower triangular matrix

        # convert to probability for each token over every head
        attn = F.softmax(similarity, dim=-1)

        # compute the weighted (according to attention prob.) sum of values for each token, for each head
        y = attn @ v
        y = y.transpose(1, 2).contiguous().view(B, L, d_model) # concatenate heads
        y = self.out_proj(y) # mix the heads back together

        return y


class FeedForward(nn.Module):
    """
    Position-wise feed-forward block.
    """
    def __init__(self, d_model, hidden_dim, dropout=0.1):
        """
        Args:
            d_model: embedding size of the model.
            hidden_dim: inner dimension of the feed-forward layer.
            dropout: dropout probability.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim), # higher dimensional projection
            nn.GELU(),
            nn.Linear(hidden_dim, d_model), # back to original dimension
        )

    def forward(self, x):
        """
        Applies the feed-forward transformation independently to every token.
        """
        return self.net(x)


class TransformerBlock(nn.Module):
    """
    Pre-LayerNormalization transformer block: LN -> attention -> out -> LN -> FF -> out.
    """
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
        # apply attention and feed-forward
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
            vocab_size: size of the word vocabulary.
            slots_size: number of slot labels (including pad/cls).
            n_intents: number of intent labels.
            pos_emb_size: maximum sequence length supported by the positional embeddings.
            d_model: embedding size of the model.
            n_heads: number of attention heads per transformer block.
            num_layers: number of stacked transformer blocks.
            ff_dim: inner dimension of the feed-forward blocks.
            dropout: dropout probability.
        """
        super().__init__()
        self.pos_emb_size = pos_emb_size
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(pos_emb_size, d_model)

        # create a stack of transformer blocks
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

        # causal mask: each token cannot see future tokens
        mask = torch.tril(torch.ones(pos_emb_size, pos_emb_size)).unsqueeze(0).unsqueeze(0)
        # not a learnable parameter, but must move with the model on .to(device)
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
        # positions beyond pos_emb_size were never trained, so longer sequences would be invalid
        assert L <= self.pos_emb_size

        # embed the tokens and their positions
        pos = torch.arange(L, device=idx.device)
        x = self.token_embed(idx) + self.pos_embed(pos)

        mask = self.mask[:, :, :L, :L]

        # pass through the transformer blocks
        for block in self.blocks:
            x = block(x, mask)

        x = self.ln_f(x) # normalize
        # TODO (exercise 2.A.2): x = self.dropout(x)

        slots = self.slot_out(x)

        # for sequence i in the batch, get the vector associated
        # with its CLS token (last of the sequence)
        cls_tokens = torch.stack([x[i, seq_lens[i] - 1] for i in range(x.shape[0])])
        intent = self.intent_out(cls_tokens) # logits for the intent classification

        return slots, intent  # (B, L, slots_size), (B, n_intents)
