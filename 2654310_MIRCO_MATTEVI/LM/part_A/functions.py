# Training, evaluation and weight-init helpers for the GPT2 model

import copy
import math
import torch
import torch.nn as nn
from tqdm import tqdm


def init_weights(mat):
    """
    Uniformly initializes every nn.Linear layer's weights and sets its bias to a small constant.
    
    Args:
        mat: the module to initialize.
    """
    for m in mat.modules():
        if type(m) in [nn.Linear]:
            torch.nn.init.uniform_(m.weight, -0.01, 0.01) # set weights: uniform distribution in [-0.01, 0.01]
            if m.bias is not None: 
                m.bias.data.fill_(0.01) # set bias as a small constant


def train_loop(data, optimizer, criterion, model):
    """
    Runs one training epoch over `data`.

    Args:
        data: DataLoader yielding (input_ids, labels, n_tokens) batches.
        optimizer: optimizer updating the model's parameters.
        criterion: loss function (CrossEntropyLoss, ignoring the pad token).
        model: the GPT2 model being trained.
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
        output = model(input_ids) # forward pass
        loss = criterion(output.permute(0, 2, 1), labels) # cross-entropy
        # from avg loss to total loss (batches may have different number of tokens, ex. for padding)
        # we want to compute the avg loss of the epoch correctly
        loss_array.append(loss.item() * n_tokens) 
        number_of_tokens.append(n_tokens) # keep track of the number of tokens in this batch for weighted averaging
        loss.backward() # backpropagation
        optimizer.step() # update the model's parameters

        if i % 100 == 0:
            pbar.set_postfix(loss=(sum(loss_array) / sum(number_of_tokens)).item())

    return sum(loss_array) / sum(number_of_tokens) # epoch avg loss


def eval_loop(data, eval_criterion, model):
    """
    Evaluates the model on `data` without updating its weights.

    Args:
        data: DataLoader yielding (input_ids, labels, n_tokens) batches.
        eval_criterion: loss function (CrossEntropyLoss, ignoring the pad token).
        model: the GPT2 model being evaluated.
    Returns:
        Tuple (perplexity, token-weighted average loss).
    """
    model.eval()
    loss_array = []
    number_of_tokens = []

    with torch.no_grad():
        for input_ids, labels, n_tokens in tqdm(data, desc="Evaluating: ", unit="batch", total=len(data)):
            output = model(input_ids)
            loss = eval_criterion(output.permute(0, 2, 1), labels)
            # from avg loss to total loss (batches may have different number of tokens, ex. for padding)
            # we want to compute the avg loss of the epoch correctly
            loss_array.append(loss.item() * n_tokens)
            number_of_tokens.append(n_tokens) # keep track of the number of tokens in this batch for weighted averaging

    loss_to_return = sum(loss_array) / sum(number_of_tokens)
    ppl = math.exp(loss_to_return) # perplexity
    return ppl, loss_to_return


def train_and_evaluate_model(
    model,
    optimizer,
    criterion_train,
    criterion_eval,
    train_loader,
    dev_loader,
    test_loader,
    n_epochs=20,
    patience=3,
):
    """
    Trains `model` with early stopping on dev-set perplexity, then reports test PPL.

    Args:
        model: the GPT2 model to train.
        optimizer: optimizer updating the model's parameters.
        criterion_train: training loss function.
        criterion_eval: evaluation loss function.
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
        loss = train_loop(train_loader, optimizer, criterion_train, model)
        sampled_epochs.append(epoch)
        losses_train.append(loss.item())

        ppl_dev, loss_dev = eval_loop(dev_loader, criterion_eval, model)
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
    test_ppl, _ = eval_loop(test_loader, criterion_eval, best_model)

    history = {
        "sampled_epochs": sampled_epochs,
        "losses_train": losses_train,
        "losses_dev": losses_dev,
    }
    return best_model, best_ppl, test_ppl, history
