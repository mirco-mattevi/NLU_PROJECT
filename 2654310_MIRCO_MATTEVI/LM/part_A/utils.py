# Data loading and preprocessing for the PennTreeBank language modeling dataset,
# adapted from LAB 04.

import torch
import torch.utils.data as data


def read_file(path, eos_token="<eos>"):
    """
    Reads a PennTreeBank split, one sentence per line, appending an end-of-sentence token.

    Args:
        path: path to the corpus file.
        eos_token: token appended at the end of every sentence.
    Returns:
        List of sentence strings.
    """
    output = []
    with open(path, "r") as f:
        for line in f.readlines():
            output.append(line.strip() + " " + eos_token)
    return output


class PennTreeBank(data.Dataset):
    """Wraps a list of raw sentences as a PyTorch Dataset."""

    def __init__(self, corpus):
        self.sents = [sent for sent in corpus]

    def __len__(self):
        return len(self.sents)

    def __getitem__(self, idx):
        return self.sents[idx]


def collate_fn(batch, tokenizer, device):
    """
    Tokenizes a batch of sentences and builds next-token-prediction pairs.

    The input is the tokenized sentence without its last token, and the label
    is the same sequence shifted left by one position (predict the next token).

    Args:
        batch: list of raw sentence strings.
        tokenizer: HuggingFace tokenizer (padding enabled, pad_token set).
        device: torch device to move the tensors to.
    Returns:
        Tuple (input_ids, labels, n_tokens), where n_tokens is the number of
        non-pad tokens in the batch (used to average the loss correctly).
    """
    tokenized = tokenizer(batch, padding=True, return_tensors="pt")

    input_ids = tokenized.input_ids[:, :-1].detach().clone().to(device)
    labels = tokenized.input_ids[:, 1:].detach().clone().to(device)

    n_tokens = torch.sum(input_ids != tokenizer.pad_token_id)

    return input_ids, labels, n_tokens
