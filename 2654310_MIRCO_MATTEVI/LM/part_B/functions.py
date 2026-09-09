# Training, evaluation and setup helpers for LoRA fine-tuning of pretrained
# GPT2 (Part 1.B), adapted from LAB 04.

import copy
import math

import torch
import torch.optim as optim
from tqdm import tqdm
from transformers import AutoTokenizer

from model import GPT2_LoRA


def param_stats(model):
    """Prints the total, trainable and frozen parameter counts of `model`."""
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    print(f"total params: {total:,}")
    print(f"trainable params: {trainable:,}")
    print(f"frozen params: {total - trainable:,}")


def load_tokenizer_and_model(rank, alpha, device):
    """
    Loads the pretrained GPT2 tokenizer and the LoRA-wrapped GPT2 model.

    Args:
        rank: LoRA rank passed to GPT2_LoRA.
        alpha: LoRA scaling factor passed to GPT2_LoRA.
        device: torch device to move the model to.
    Returns:
        Tuple (tokenizer, model).
    """
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    model = GPT2_LoRA.from_pretrained("openai-community/gpt2", rank=rank, alpha=alpha)
    model.to(device)

    return tokenizer, model


def prepare_optimizer(model, lr):
    """
    Freezes every parameter except the LoRA adapters and builds their optimizer.

    Args:
        model: a GPT2_LoRA model.
        lr: learning rate for AdamW.
    Returns:
        AdamW optimizer over the trainable (LoRA) parameters only.
    """
    for param in model.parameters():
        # TODO (exercise 1.B): freeze every parameter by default.
        # param.requires_grad = False
        pass
    for module in model.modules():
        # TODO (exercise 1.B): make only the LoRA adapter parameters trainable,
        # e.g. `if hasattr(module, "lora_A"): ... param.requires_grad = True`.
        pass

    optimizer = optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=lr,
    )
    return optimizer


def train_loop(data, optimizer, model):
    """
    Runs one training epoch over `data`, using the model's own causal-LM loss.

    Args:
        data: DataLoader yielding (input_ids, labels, n_tokens) batches.
        optimizer: optimizer updating the trainable (LoRA) parameters.
        model: the GPT2_LoRA model being trained.
    Returns:
        Token-weighted average training loss over the whole epoch.
    """
    model.train()
    loss_array = []
    number_of_tokens = []

    pbar = tqdm(data, desc="Training:", unit="batch", total=len(data))
    for i, (input_ids, labels, n_tokens) in enumerate(pbar):
        optimizer.zero_grad()
        output = model(input_ids, labels=input_ids)
        loss_array.append(output.loss.item() * n_tokens)
        number_of_tokens.append(n_tokens)
        output.loss.backward()
        optimizer.step()

        if i % 100 == 0:
            pbar.set_postfix(loss=(sum(loss_array) / sum(number_of_tokens)).item())

    return sum(loss_array) / sum(number_of_tokens)


def eval_loop(data, model):
    """
    Evaluates the model on `data` without updating its weights.

    Args:
        data: DataLoader yielding (input_ids, labels, n_tokens) batches.
        model: the GPT2_LoRA model being evaluated.
    Returns:
        Tuple (perplexity, token-weighted average loss).
    """
    model.eval()
    loss_array = []
    number_of_tokens = []

    with torch.no_grad():
        for input_ids, labels, n_tokens in tqdm(data, desc="Evaluating: ", unit="batch", total=len(data)):
            output = model(input_ids, labels=input_ids)
            loss_array.append(output.loss.item() * n_tokens)
            number_of_tokens.append(n_tokens)

    loss_to_return = sum(loss_array) / sum(number_of_tokens)
    ppl = math.exp(loss_to_return)
    return ppl, loss_to_return


def train_model(model, optimizer, train_loader, dev_loader, test_loader, n_epochs=100, patience=3):
    """
    Trains `model` with early stopping on dev-set perplexity, then reports test PPL.

    Args:
        model: the GPT2_LoRA model to train.
        optimizer: optimizer updating the trainable (LoRA) parameters.
        train_loader, dev_loader, test_loader: DataLoaders for each split.
        n_epochs: maximum number of training epochs.
        patience: number of consecutive non-improving evals before stopping early.
    Returns:
        Tuple (best_model, best_dev_ppl, test_ppl).
    """
    best_ppl = math.inf
    best_model = None
    current_patience = patience

    pbar = tqdm(range(n_epochs))
    for epoch in pbar:
        train_loop(train_loader, optimizer, model)
        ppl_dev, _ = eval_loop(dev_loader, model)
        pbar.set_description("PPL: %f" % ppl_dev)

        if ppl_dev < best_ppl:
            best_ppl = ppl_dev
            best_model = copy.deepcopy(model).to('cpu')
            current_patience = patience
        else:
            current_patience -= 1

        if current_patience <= 0:
            break

    device = next(model.parameters()).device
    best_model.to(device)
    test_ppl, _ = eval_loop(test_loader, best_model)

    return best_model, best_ppl, test_ppl
