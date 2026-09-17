# Data loading and preprocessing for the PennTreeBank dataset.

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
    Converts a batch of sentences into tokenized tensors and builds next-token-prediction pairs.

    Args:
        batch: list of raw sentence strings.
        tokenizer: HuggingFace GPT-2 tokenizer.
        device: torch device to move the tensors to.
    Returns:
        Tuple (input_ids, labels, n_tokens), where n_tokens is the number of
        non-pad tokens in the batch (used to average the loss correctly).
    """
    # out = tensor of shape (Batch size, Max length)
    tokenized = tokenizer(batch, padding=True, return_tensors="pt") # padding to have the same length in the batch

    # build next token prediction pairs
    input_ids = tokenized.input_ids[:, :-1].detach().clone().to(device) # every token except the last one
    labels = tokenized.input_ids[:, 1:].detach().clone().to(device) # every token except the first one (left shifted --> predict the next)

    # store the number of non-pad tokens
    n_tokens = torch.sum(input_ids != tokenizer.pad_token_id)

    return input_ids, labels, n_tokens
