# Training, evaluation and weight-init helpers for the GPT2 joint
# intent classification / slot filling model

import copy
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report
from tqdm import tqdm

from conll import evaluate
from model import GPT2


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


def train_loop(data, optimizer, criterion_slots, criterion_intents, model):
    """
    Runs one training epoch over `data`, summing the slot and intent losses.

    Args:
        data: DataLoader yielding batches from utils.collate_fn.
        optimizer: optimizer updating the model's parameters.
        criterion_slots: slot-filling loss function (CrossEntropyLoss, ignores pad/cls labels).
        criterion_intents: intent classification loss function.
        model: the GPT2 model being trained.
    Returns:
        List of per-batch joint losses.
    """
    model.train()
    loss_array = []

    for batch in data:
        optimizer.zero_grad() # zero the gradient so that it doesn't accumulate across batches
        slots, intent = model(batch['utterances'], batch['slots_len']) # forward pass
        slots = slots.permute(0, 2, 1) # (B, slots_size, L), required by CrossEntropyLoss

        loss_intent = criterion_intents(intent, batch['intents'])
        loss_slot = criterion_slots(slots, batch['y_slots'])
        loss = loss_intent + loss_slot # simplest joint-training strategy: unweighted sum
        loss_array.append(loss.item())
        loss.backward() # backpropagation
        optimizer.step() # update the model's parameters

    return loss_array


def eval_loop(data, criterion_slots, criterion_intents, model, lang):
    """
    Evaluates the model on `data` without updating its weights:
    slot F1 (conll) and intent classification report.

    Args:
        data: DataLoader yielding batches from utils.collate_fn.
        criterion_slots: slot-filling loss function (CrossEntropyLoss, ignores pad/cls labels).
        criterion_intents: intent classification loss function.
        model: the GPT2 model being evaluated.
        lang: utils.Lang instance, used to decode ids back to words/labels.
    Returns:
        Tuple (slot_results, intent_report, loss_array):
            slot_results: conll.evaluate() output (dict with per-class + "total" P/R/F1).
            intent_report: sklearn classification_report() output_dict.
            loss_array: list of per-batch joint losses.
    """
    model.eval()
    loss_array = []
    ref_intents, hyp_intents = [], []
    ref_slots, hyp_slots = [], []

    with torch.no_grad():
        for batch in data:
            slots, intents = model(batch['utterances'], batch['slots_len']) # forward pass
            slots = slots.permute(0, 2, 1) # (B, slots_size, L), required by CrossEntropyLoss
            loss_intent = criterion_intents(intents, batch['intents'])
            loss_slot = criterion_slots(slots, batch['y_slots'])
            loss_array.append((loss_intent + loss_slot).item())

            # intent inference: id of the most likely intent -> label
            out_intents = [lang.id2intent[x] for x in torch.argmax(intents, dim=1).tolist()] # dim = 1 -> for each row (sentence)
            gt_intents = [lang.id2intent[x] for x in batch['intents'].tolist()]
            ref_intents.extend(gt_intents)
            hyp_intents.extend(out_intents)

            # slot inference: decode each sequence back to (word, label) pairs for conll
            output_slots = torch.argmax(slots, dim=1) # Best slot label for each token
            for id_seq, seq in enumerate(output_slots):
                # exclude the CLS token from each sentence
                length = batch['slots_len'].tolist()[id_seq] - 1
                utt_ids = batch['utterances'][id_seq][:length].tolist()
                gt_ids = batch['y_slots'][id_seq][:length].tolist()
                # retrieve the slot labels and words from the ids
                gt_slots = [lang.id2slot[elem] for elem in gt_ids]
                utterance = [lang.id2word[elem] for elem in utt_ids]

                # exclude the CLS token from each predicted sentence ids
                to_decode = seq[:length].tolist()
                # build (word, slot_label) for predicted and gt as expected by conll
                ref_slots.append([(utterance[id_el], elem) for id_el, elem in enumerate(gt_slots)])
                hyp_slots.append([(utterance[id_el], lang.id2slot[elem]) for id_el, elem in enumerate(to_decode)])

    try:
        results = evaluate(ref_slots, hyp_slots)
    except Exception as ex:
        # the model may predict a slot class absent in REF, which conll.evaluate rejects
        print("Warning:", ex)
        ref_s = set([x[1] for x in ref_slots])
        hyp_s = set([x[1] for x in hyp_slots])
        print(hyp_s.difference(ref_s))
        results = {"total": {"f": 0}}

    report_intent = classification_report(ref_intents, hyp_intents, zero_division=False, output_dict=True)
    return results, report_intent, loss_array


def train_and_evaluate_model(
    model,
    optimizer,
    criterion_slots,
    criterion_intents,
    train_loader,
    dev_loader,
    test_loader,
    lang,
    n_epochs=200,
    patience=3,
):
    """
    Trains `model` with early stopping on dev-set slot F1, then reports test performance.

    Args:
        model: the GPT2 model to train.
        optimizer: optimizer updating the model's parameters.
        criterion_slots, criterion_intents: loss functions.
        train_loader, dev_loader, test_loader: DataLoaders for each split.
        lang: utils.Lang instance, needed by eval_loop.
        n_epochs: maximum number of training epochs.
        patience: number of consecutive non-improving evals before stopping early.
    Returns:
        Tuple (best_model, best_f1, results_test, intent_test, history), where
        history is a dict with "sampled_epochs", "losses_train", "losses_dev".
    """
    losses_train = []
    losses_dev = []
    sampled_epochs = []
    best_f1 = -1 # so that a model is always kept, even if dev F1 stays 0
    best_loss_dev = float('inf')
    best_model = None
    current_patience = patience

    pbar = tqdm(range(n_epochs))
    for epoch in pbar:
        loss = train_loop(train_loader, optimizer, criterion_slots, criterion_intents, model)
        if epoch % 1 == 0: # sample every epoch
            sampled_epochs.append(epoch)
            losses_train.append(np.asarray(loss).mean())

            results_dev, intent_res, loss_dev = eval_loop(
                dev_loader, criterion_slots, criterion_intents, model, lang
            )
            losses_dev.append(np.asarray(loss_dev).mean())
            pbar.set_description(
                f"Slot F1: {results_dev['total']['f']:.2f}; Intent Acc: {intent_res['accuracy']:.2f}"
            )

            # the model is selected on dev slot F1, but patience is also reset when the dev loss
            # improves: in the first epochs slot F1 stays ~0 (only "O" predicted) while the
            # loss keeps decreasing, and stopping on F1 alone would end training there
            improved = False
            f1 = results_dev['total']['f']
            if f1 > best_f1:
                best_f1 = f1
                best_model = copy.deepcopy(model).to('cpu')
                improved = True
            if losses_dev[-1] < best_loss_dev:
                best_loss_dev = losses_dev[-1]
                improved = True

            if improved:
                current_patience = patience
            else:
                current_patience -= 1

            if current_patience <= 0: # early stopping
                break

    device = next(model.parameters()).device
    best_model.to(device)
    results_test, intent_test, _ = eval_loop(test_loader, criterion_slots, criterion_intents, best_model, lang)

    history = {
        "sampled_epochs": sampled_epochs,
        "losses_train": losses_train,
        "losses_dev": losses_dev,
    }
    return best_model, best_f1, results_test, intent_test, history


def save_model(path, model, lang):
    """
    Saves the model weights together with the vocabularies needed to reload it.

    Args:
        path: destination file.
        model: the GPT2 model.
        lang: utils.Lang instance whose word/slot/intent vocabularies are saved.
    """
    saving_object = {
        "model": model.state_dict(),
        "w2id": lang.word2id,
        "slot2id": lang.slot2id,
        "intent2id": lang.intent2id,
    }
    torch.save(saving_object, path)


def plot_losses(history, path, title='Train and Dev Losses'):
    """
    Saves the training and dev loss curves collected by train_and_evaluate_model to `path`.

    Args:
        history: the `history` dict returned by train_and_evaluate_model.
        path: destination image file.
        title: title of the plot.
    """
    import matplotlib
    matplotlib.use("Agg") # Render plots directly to image files
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(8, 5))
    fig.patch.set_facecolor('white')
    plt.title(title)
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.plot(history["sampled_epochs"], history["losses_train"], label='Train loss')
    plt.plot(history["sampled_epochs"], history["losses_dev"], label='Dev loss')
    plt.legend()
    fig.savefig(path)
    plt.close(fig) # otherwise every new plot is drawn on top of the previous ones


def multiple_runs(train_loader, dev_loader, test_loader, lang, model_kwargs, lr,
                  device, n_epochs=200, runs=5, patience=3):
    """
    Trains the model from scratch `runs` times and aggregates dev and test performance.

    Used to avoid the noise of a single training run on small datasets (like ATIS).
    Candidates must be compared on the dev scores, the test scores are only reported.

    Args:
        train_loader, dev_loader, test_loader: DataLoaders for each split.
        lang: utils.Lang instance.
        model_kwargs: kwargs forwarded to model.GPT2 (besides the vocab/label sizes).
        lr: learning rate for the AdamW optimizer.
        device: torch device to build each model on.
        n_epochs: maximum number of training epochs per run.
        runs: number of independent training runs (seeded 0 .. runs-1).
        patience: early-stopping patience per run.
    Returns:
        dict with:
            "dev_f1s", "test_f1s", "test_accs": numpy arrays, one entry per run.
            "best_model", "history": model and loss history of the run with the best dev slot F1.
    """
    vocab_len = len(lang.word2id)
    slots_len = len(lang.id2slot)
    n_intents = len(lang.intent2id)

    dev_f1s, test_f1s, test_accs = [], [], []
    best_run_f1, best_run_model, best_run_history = -1, None, None
    for seed in tqdm(range(runs)):
        # run i always uses seed i: every candidate sees the same initializations and batch
        # orders, so differences between candidates are not due to the random seed
        torch.manual_seed(seed)
        model = GPT2(vocab_len, slots_len, n_intents, **model_kwargs).to(device)
        model.apply(init_weights)

        optimizer = optim.AdamW(model.parameters(), lr=lr)
        criterion_slots = nn.CrossEntropyLoss(ignore_index=lang.pad_token)
        criterion_intents = nn.CrossEntropyLoss()

        run_model, run_f1, results_test, intent_test, history = train_and_evaluate_model(
            model, optimizer, criterion_slots, criterion_intents,
            train_loader, dev_loader, test_loader, lang, n_epochs=n_epochs, patience=patience,
        )
        dev_f1s.append(run_f1)
        test_f1s.append(results_test['total']['f'])
        test_accs.append(intent_test['accuracy'])

        if run_f1 > best_run_f1:
            best_run_f1, best_run_model, best_run_history = run_f1, run_model.to('cpu'), history

    res = {
        "dev_f1s": np.asarray(dev_f1s),
        "test_f1s": np.asarray(test_f1s),
        "test_accs": np.asarray(test_accs),
        "best_model": best_run_model,
        "history": best_run_history,
    }
    print('Dev Slot F1', round(res["dev_f1s"].mean(), 3), '+-', round(res["dev_f1s"].std(), 3))
    print('Test Slot F1', round(res["test_f1s"].mean(), 3), '+-', round(res["test_f1s"].std(), 3))
    print('Test Intent Acc', round(res["test_accs"].mean(), 3), '+-', round(res["test_accs"].std(), 3))
    return res


def format_results(res):
    """
    Formats the output of multiple_runs as a single README line (mean +- std over runs).

    Args:
        res: dict returned by multiple_runs.
    Returns:
        the formatted string.
    """
    return (
        f"dev slot F1: {res['dev_f1s'].mean():.4f} +- {res['dev_f1s'].std():.4f} | "
        f"test slot F1: {res['test_f1s'].mean():.4f} +- {res['test_f1s'].std():.4f} | "
        f"test intent acc: {res['test_accs'].mean():.4f} +- {res['test_accs'].std():.4f}"
    )


def run_tuning(name, values, config, best_f1, lang, device, train_loader, dev_loader, test_loader, runs=5):
    """
    Tunes a single hyperparameter: trains one candidate per value of `name`, keeping all the
    other hyperparameters fixed to `config`. A value is accepted only if it improves the mean
    dev slot F1, so calling this once per hyperparameter gives the incremental search
    (0: lr, 1: d_model -> n_heads -> num_layers -> ff_dim -> lr again for the larger model).

    Args:
        name: hyperparameter to tune, "lr" or one of the model.GPT2 kwargs.
        values: candidate values for `name`.
        config: current best configuration, dict with "lr" and the model.GPT2 kwargs.
        best_f1: mean dev slot F1 to beat.
        lang: utils.Lang instance.
        device: torch device to train on.
        train_loader, dev_loader, test_loader: DataLoaders for each split.
        runs: number of independent training runs per candidate.
    Returns:
        Tuple (best_config, best_f1, best_res), where best_res is the multiple_runs dict of
        the accepted candidate (None if no value beats `best_f1`).
    """
    best_config = config
    best_res = None

    with open("README.md", "a") as f:
        f.write(f"\n[{name} tuning] starting from {config} (dev slot F1 to beat: {best_f1:.4f})\n")

    for value in values:
        candidate = {**config, name: value}
        lr = candidate["lr"]
        model_kwargs = {k: v for k, v in candidate.items() if k != "lr"}
        start = time.time()
        res = multiple_runs(
            train_loader, dev_loader, test_loader, lang,
            model_kwargs, lr, device, runs=runs,
        )
        minutes = (time.time() - start) / 60

        with open("README.md", "a") as f:
            f.write(f"{name}={value} | {format_results(res)} | {minutes:.1f} min\n")
        plot_losses(
            res["history"], os.path.join("plots", f"{name}{value}.png"),
            title=f"{name}={value} (best of {runs} runs)",
        )

        if res["dev_f1s"].mean() > best_f1:
            best_f1, best_res, best_config = res["dev_f1s"].mean(), res, candidate

    with open("README.md", "a") as f:
        if best_res is None:
            f.write(f"--> no {name} improves the dev slot F1, kept {name}={config[name]}\n")
        else:
            f.write(f"--> best {name}={best_config[name]} | {format_results(best_res)}\n")
    if best_res is not None:
        save_model(os.path.join("bin", f"{name}{best_config[name]}.pt"), best_res["best_model"], lang)

    return best_config, best_f1, best_res

