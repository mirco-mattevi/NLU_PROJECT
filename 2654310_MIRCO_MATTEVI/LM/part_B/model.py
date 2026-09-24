# LoRA-adapted GPT2, starts from the pretrained huggingface GPT2 model

from typing import Optional, Tuple, Union
import torch
from transformers import GPT2LMHeadModel
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention


class CustomGPT2Attention(GPT2Attention):
    """
    GPT2Attention with LoRA adapters on the query/key/value projections.
    """

    def __init__(self, config, rank, alpha):
        """
        Args:
            config: the GPT2 model config.
            rank: LoRA rank.
            alpha: LoRA scaling factor.
        """
        super().__init__(config)
        # TODO (exercise 1.B): add the LoRA low-rank matrices here (e.g. a pair
        # of matrices A, B applied to self.c_attn's query/key/value output,
        # scaled by alpha / rank).

    # TODO (exercise 1.B): edit this method to add the LoRA delta to
    # query/key/value before they are split into heads.
    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:
        if encoder_hidden_states is not None:
            if not hasattr(self, "q_attn"):
                raise ValueError(
                    "If class is used as cross attention, the weights `q_attn` have to be defined. "
                    "Please make sure to instantiate class with `GPT2Attention(..., is_cross_attention=True)`."
                )

            query = self.q_attn(hidden_states)
            key, value = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
            attention_mask = encoder_attention_mask
        else:
            # Q,K,V projections fused into one (faster) and then split into the 3
            query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)

        # split into heads
        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        if use_cache is True:
            present = (key, value)
        else:
            present = None

        if self.reorder_and_upcast_attn:
            attn_output, attn_weights = self._upcast_and_reordered_attn(query, key, value, attention_mask, head_mask)
        else:
            # retrieve the contextualized representation of each token
            attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)

        # concat heads, mix info in the projected layer, add dropout
        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs  # a, present, (attentions)


class GPT2_LoRA(GPT2LMHeadModel):
    """
    Pretrained GPT2LMHeadModel with every attention block's wrapped by LoRA.
    """

    def __init__(self, *model_args, rank, alpha, **model_kwargs):
        """
        Args:
            rank: LoRA rank.
            alpha: LoRA scaling factor.
        """
        super().__init__(*model_args, **model_kwargs)
        for block in self.transformer.h:
            # TODO (exercise 1.B): replace block.attn with
            # CustomGPT2Attention(self.config, rank, alpha), then copy over the
            # pretrained weights with block.attn.load_state_dict(..., strict=False).
            pass

    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)
