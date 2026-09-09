# Joint intent classification / slot filling heads on top of pretrained GPT2
# and BERT (Part 2.B). No notebook skeleton exists for this part: these are
# documented stubs following the architecture described in the assignment and
# in the Joint BERT paper (https://arxiv.org/abs/1902.10909).
#
# Both models are multi-task: one linear layer tags every (sub-word) token
# with a slot label, another classifies the whole utterance's intent. What
# differs between them is how the utterance-level representation for intent
# classification is obtained, since only BERT has a [CLS] token.

import torch.nn as nn
from transformers import BertModel, GPT2Model


class JointBERT(nn.Module):
    """BERT encoder with a slot-tagging head and a [CLS]-pooled intent-classification head."""

    def __init__(self, model_name, n_slots, n_intents, dropout=0.1):
        """
        Args:
            model_name: HuggingFace id of the pretrained BERT checkpoint
                (e.g. "bert-base-uncased").
            n_slots: number of slot labels.
            n_intents: number of intent labels.
            dropout: dropout probability applied before both output layers.
        """
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size

        self.dropout = nn.Dropout(dropout)
        self.slot_out = nn.Linear(hidden_size, n_slots)
        # BERT prepends a [CLS] token to every sequence: its pooled representation
        # (pooler_output) is the standard choice for sequence-level classification.
        self.intent_out = nn.Linear(hidden_size, n_intents)

        # TODO (exercise 2.B): implement forward() using self.bert(input_ids,
        # attention_mask=attention_mask), then:
        #   - slots = self.slot_out(self.dropout(outputs.last_hidden_state))
        #   - intent = self.intent_out(self.dropout(outputs.pooler_output))
        # Sub-word alignment: slot labels must be assigned to each word's first
        # sub-token only (build this at data-loading time in utils.py, not here).


class JointGPT2(nn.Module):
    """GPT2 encoder with a slot-tagging head and a last-token-pooled intent-classification head."""

    def __init__(self, model_name, n_slots, n_intents, dropout=0.1):
        """
        Args:
            model_name: HuggingFace id of the pretrained GPT2 checkpoint
                (e.g. "openai-community/gpt2" or "openai-community/gpt2-medium").
            n_slots: number of slot labels.
            n_intents: number of intent labels.
            dropout: dropout probability applied before both output layers.
        """
        super().__init__()
        self.gpt2 = GPT2Model.from_pretrained(model_name)
        hidden_size = self.gpt2.config.hidden_size

        self.dropout = nn.Dropout(dropout)
        self.slot_out = nn.Linear(hidden_size, n_slots)
        # GPT2 has no [CLS] token, so intent classification must instead pool
        # the hidden state at each sequence's last non-pad position (its
        # right-to-left causal context already saw the whole utterance).
        self.intent_out = nn.Linear(hidden_size, n_intents)

        # TODO (exercise 2.B): implement forward() using self.gpt2(input_ids,
        # attention_mask=attention_mask), then:
        #   - slots = self.slot_out(self.dropout(outputs.last_hidden_state))
        #   - last_idx = attention_mask.sum(dim=1) - 1  # index of the last real token
        #   - pooled = outputs.last_hidden_state[torch.arange(B), last_idx]
        #   - intent = self.intent_out(self.dropout(pooled))
        # Sub-word alignment: same as JointBERT, done in utils.py.
