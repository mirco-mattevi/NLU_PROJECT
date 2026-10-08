# Training, evaluation and setup helpers 
# for LoRA fine-tuning of pretrained GPT2

import copy
import math
import os

import torch
import torch.optim as optim
from tqdm import tqdm
from transformers import AutoTokenizer

from model import GPT2_LoRA


def param_stats(model):
    """
        Prints the total, trainable and frozen parameter counts of `model`.
    """
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    print(f"total params: {total:,}")
    print(f"trainable params: {trainable:,}")
    print(f"frozen params: {total - trainable:,}")


def load_tokenizer():
    """
    Loads the pretrained GPT2 tokenizer.

    Returns:
        the GPT2 tokenizer, with the eos token used as pad token.
    """
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    tokenizer.pad_token = tokenizer.eos_token # GPT2 does not have a pad token by default
    return tokenizer


def load_model(rank, alpha, device):
    """
    Loads the pretrained GPT2 wrapped with fresh LoRA adapters.

    Args:
        rank: LoRA rank passed to GPT2_LoRA.
        alpha: LoRA scaling factor passed to GPT2_LoRA.
        device: torch device to move the model to.
    Returns:
        the GPT2_LoRA model.
    """
    model = GPT2_LoRA.from_pretrained("openai-community/gpt2", rank=rank, alpha=alpha)
    model.to(device)
    return model


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
        # freeze every parameter by default
        param.requires_grad = False

    for module in model.modules():
        # make only the LoRA parameters trainable
        if hasattr(module, "lora_A"): # it has also LoRA_B
            for param in module.lora_A.parameters():
                param.requires_grad = True
            for param in module.lora_B.parameters():
                param.requires_grad = True

    optimizer = optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=lr,
    )
    return optimizer


def train_loop(data, optimizer, model):
    """
    Runs one training epoch over `data`.

    Args:
        data: DataLoader yielding (input_ids, labels, n_tokens) batches.
        optimizer: optimizer updating the trainable (LoRA) parameters.
        model: the GPT2_LoRA model being trained.
    Returns:
        average training loss over the whole epoch.
    """
    model.train()
    loss_array = []
    number_of_tokens = []

    # progress bar
    pbar = tqdm(data, desc="Training:", unit="batch", total=len(data))
    for i, (input_ids, labels, n_tokens) in enumerate(pbar):
        optimizer.zero_grad() # zero the gradient so that it doesn't accumulate across batches
        output = model(input_ids, labels=labels) # forward pass (loss is computed by the model, pad ignored)
        # from avg loss to total loss (batches may have different number of tokens, ex. for padding)
        # we want to compute the avg loss of the epoch correctly
        loss_array.append(output.loss.item() * n_tokens)
        number_of_tokens.append(n_tokens) # keep track of the number of tokens in this batch for weighted averaging
        output.loss.backward() # backpropagation
        optimizer.step() # update the model's parameter

        if i % 100 == 0:
            pbar.set_postfix(loss=(sum(loss_array) / sum(number_of_tokens)).item())

    return sum(loss_array) / sum(number_of_tokens) # epoch avg loss


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
    loss_to_return = []
    loss_array = []
    number_of_tokens = []

    with torch.no_grad():
        for input_ids, labels, n_tokens in tqdm(data, desc="Evaluating: ", unit="batch", total=len(data)):
            output = model(input_ids, labels=labels) # forward pass (loss is computed by the model, pad ignored)
            # from avg loss to total loss (batches may have different number of tokens, ex. for padding)
            # we want to compute the avg loss of the epoch correctly
            loss_array.append(output.loss.item() * n_tokens)
            number_of_tokens.append(n_tokens) # keep track of the number of tokens in this batch for weighted averaging

    loss_to_return = sum(loss_array) / sum(number_of_tokens)
    ppl = math.exp(loss_to_return) # perplexity
    return ppl, loss_to_return


def train_model(
        model,
        optimizer,
        train_loader,
        dev_loader,
        test_loader,
        n_epochs=20,
        patience=3
):
    """
    Trains `model` with early stopping on dev-set perplexity, then reports test PPL.

    Args:
        model: the GPT2_LoRA model to train.
        optimizer: optimizer updating the trainable (LoRA) parameters.
        train_loader, dev_loader, test_loader: DataLoaders for each split.
        n_epochs: maximum number of training epochs.
        patience: number of consecutive non-improving evals before stopping early.
    Returns:
        Tuple (best_model, best_dev_ppl, test_ppl, history), where history is a
        dict with "sampled_epochs", "losses_train", "losses_dev".
    """
    losses_train = []
    losses_dev = []
    sampled_epochs = []
    best_ppl = math.inf
    best_model = None
    current_patience = patience

    pbar = tqdm(range(n_epochs))
    for epoch in pbar:
        loss = train_loop(train_loader, optimizer, model)
        sampled_epochs.append(epoch)
        losses_train.append(loss.item())

        ppl_dev, loss_dev = eval_loop(dev_loader, model)
        losses_dev.append(loss_dev.item())
        pbar.set_description("PPL: %f" % ppl_dev)

        if ppl_dev < best_ppl:
            best_ppl = ppl_dev
            best_model = copy.deepcopy(model).to('cpu')
            current_patience = patience
        else:
            current_patience -= 1

        if current_patience <= 0: # early stopping
            break

    device = next(model.parameters()).device
    best_model.to(device)
    test_ppl, _ = eval_loop(test_loader, best_model)

    history = {
        "sampled_epochs": sampled_epochs,
        "losses_train": losses_train,
        "losses_dev": losses_dev,
    }
    return best_model, best_ppl, test_ppl, history


def save_adapters(model, path):
    """
    Saves only the trainable (LoRA) parameters of `model`: the frozen GPT2 weights
    can always be reloaded from Huggingface, so there is no need to store them.

    Args:
        model: a GPT2_LoRA model.
        path: destination file.
    """
    adapters = {name: param.detach().cpu() for name, param in model.named_parameters() if param.requires_grad}
    torch.save(adapters, path)


def run_lora(rank, alpha, lr, device, train_loader, dev_loader, test_loader):
    """
    Trains one LoRA configuration starting from the pretrained GPT2.

    Args:
        rank: LoRA rank.
        alpha: LoRA scaling factor.
        lr: learning rate for the AdamW optimizer.
        device: torch device to train on.
        train_loader, dev_loader, test_loader: DataLoaders for each split.
    Returns:
        Tuple (best_model, best_ppl, test_ppl, history).
    """
    model = load_model(rank, alpha, device) # fresh pretrained weights + new adapters for every run
    optimizer = prepare_optimizer(model, lr)
    param_stats(model)

    return train_model(model, optimizer, train_loader, dev_loader, test_loader)


def run_lr_tuning(lrs, rank, alpha, device, train_loader, dev_loader, test_loader):
    """
    0: learning rate tuning.

    Args:
        lrs: learning rates to try.
        rank, alpha: fixed LoRA hyperparameters.
        device: torch device to train on.
        train_loader, dev_loader, test_loader: DataLoaders for each split.
    Returns:
        Tuple (best_model, best_ppl, test_ppl, best_lr).
    """
    best_model, best_ppl, test_ppl, best_lr = None, math.inf, math.inf, None

    with open("README.md", "a") as f:
        f.write(f"\n\n[0 - learning rate tuning] rank={rank} alpha={alpha}\n")

    for lr in lrs:
        cand_model, cand_ppl, cand_test_ppl, _ = run_lora(
            rank, alpha, lr, device, train_loader, dev_loader, test_loader
        )
        with open("README.md", "a") as f:
            f.write(f"lr={lr} | dev PPL: {cand_ppl:.2f} | test PPL: {cand_test_ppl:.2f}\n")
        save_adapters(cand_model, os.path.join("bin", f"0_lr{lr}.pt"))

        if cand_ppl < best_ppl:
            best_model, best_ppl, test_ppl, best_lr = cand_model, cand_ppl, cand_test_ppl, lr

    with open("README.md", "a") as f:
        f.write(f"--> best lr={best_lr} | dev PPL: {best_ppl:.2f} | test PPL: {test_ppl:.2f}\n")
    return best_model, best_ppl, test_ppl, best_lr


def run_rank_tuning(ranks, lr, device, train_loader, dev_loader, test_loader):
    """
    1: LoRA rank tuning, with alpha = 2 * rank so that the LoRA
    scaling (alpha / rank) stays constant and only the adapter capacity changes.

    Args:
        ranks: LoRA ranks to try.
        lr: learning rate found in experiment 0.
        device: torch device to train on.
        train_loader, dev_loader, test_loader: DataLoaders for each split.
    Returns:
        Tuple (best_model, best_ppl, test_ppl, best_rank).
    """
    best_model, best_ppl, test_ppl, best_rank = None, math.inf, math.inf, None

    with open("README.md", "a") as f:
        f.write(f"\n\n[1 - rank tuning] lr={lr} alpha=2*rank\n")

    for rank in ranks:
        alpha = 2 * rank
        cand_model, cand_ppl, cand_test_ppl, _ = run_lora(
            rank, alpha, lr, device, train_loader, dev_loader, test_loader
        )
        with open("README.md", "a") as f:
            f.write(f"rank={rank} alpha={alpha} | dev PPL: {cand_ppl:.2f} | test PPL: {cand_test_ppl:.2f}\n")
        save_adapters(cand_model, os.path.join("bin", f"1_rank{rank}.pt"))

        if cand_ppl < best_ppl:
            best_model, best_ppl, test_ppl, best_rank = cand_model, cand_ppl, cand_test_ppl, rank

    with open("README.md", "a") as f:
        f.write(f"--> best rank={best_rank} | dev PPL: {best_ppl:.2f} | test PPL: {test_ppl:.2f}\n")
    return best_model, best_ppl, test_ppl, best_rank


def run_alpha_tuning(alphas, rank, lr, device, train_loader, dev_loader, test_loader):
    """
    2: LoRA alpha tuning.

    Args:
        alphas: LoRA scaling factors to try.
        rank: LoRA rank found in experiment 1.
        lr: learning rate found in experiment 0.
        device: torch device to train on.
        train_loader, dev_loader, test_loader: DataLoaders for each split.
    Returns:
        Tuple (best_model, best_ppl, test_ppl, best_alpha).
    """
    best_model, best_ppl, test_ppl, best_alpha = None, math.inf, math.inf, None

    with open("README.md", "a") as f:
        f.write(f"\n\n[2 - alpha tuning] lr={lr} rank={rank}\n")

    for alpha in alphas:
        cand_model, cand_ppl, cand_test_ppl, _ = run_lora(
            rank, alpha, lr, device, train_loader, dev_loader, test_loader
        )
        with open("README.md", "a") as f:
            f.write(f"alpha={alpha} | dev PPL: {cand_ppl:.2f} | test PPL: {cand_test_ppl:.2f}\n")
        save_adapters(cand_model, os.path.join("bin", f"2_alpha{alpha}.pt"))

        if cand_ppl < best_ppl:
            best_model, best_ppl, test_ppl, best_alpha = cand_model, cand_ppl, cand_test_ppl, alpha

    with open("README.md", "a") as f:
        f.write(f"--> best alpha={best_alpha} | dev PPL: {best_ppl:.2f} | test PPL: {test_ppl:.2f}\n")
    return best_model, best_ppl, test_ppl, best_alpha
