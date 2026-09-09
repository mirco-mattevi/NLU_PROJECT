# Training, evaluation and reporting helpers for the from-scratch GPT2 joint
# intent classification / slot filling model (Part 2.A), adapted from LAB 05.

import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report
from tqdm import tqdm

# conll.py is provided in the LAB materials rather than duplicated here.
_LAB_LABS_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..", "LAB", "labs")
)
if _LAB_LABS_DIR not in sys.path:
    sys.path.insert(0, _LAB_LABS_DIR)

from conll import evaluate
from model import GPT2


def init_weights(mat):
    """Uniformly initializes every nn.Linear layer's weights and sets its bias to a small constant."""
    for m in mat.modules():
        if type(m) in [nn.Linear]:
            torch.nn.init.uniform_(m.weight, -0.01, 0.01)
            if m.bias is not None:
                m.bias.data.fill_(0.01)


def train_loop(data, optimizer, criterion_slots, criterion_intents, model):
    """
    Runs one training epoch over `data`, summing the slot and intent losses.

    Args:
        data: DataLoader yielding batches from utils.collate_fn.
        optimizer: optimizer updating the model's parameters.
        criterion_slots: slot-filling loss function (ignores the pad/cls label).
        criterion_intents: intent classification loss function.
        model: the GPT2 model being trained.
    Returns:
        List of per-batch joint losses.
    """
    model.train()
    loss_array = []

    for batch in data:
        optimizer.zero_grad()

        slots, intent = model(batch['utterances'], batch['slots_len'])
        slots = slots.permute(0, 2, 1)  # (B, slots_size, L), required by CrossEntropyLoss

        loss_intent = criterion_intents(intent, batch['intents'])
        loss_slot = criterion_slots(slots, batch['y_slots'])
        loss = loss_intent + loss_slot  # simplest joint-training strategy: unweighted sum
        loss_array.append(loss.item())
        loss.backward()
        optimizer.step()

    return loss_array


def eval_loop(data, criterion_slots, criterion_intents, model, lang):
    """
    Evaluates the model on `data`: slot F1 (conll) and intent classification report.

    Args:
        data: DataLoader yielding batches from utils.collate_fn.
        criterion_slots: slot-filling loss function.
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
            slots, intents = model(batch['utterances'], batch['slots_len'])
            slots = slots.permute(0, 2, 1)
            loss_intent = criterion_intents(intents, batch['intents'])
            loss_slot = criterion_slots(slots, batch['y_slots'])
            loss_array.append((loss_intent + loss_slot).item())

            out_intents = [lang.id2intent[x] for x in torch.argmax(intents, dim=1).tolist()]
            gt_intents = [lang.id2intent[x] for x in batch['intents'].tolist()]
            ref_intents.extend(gt_intents)
            hyp_intents.extend(out_intents)

            output_slots = torch.argmax(slots, dim=1)
            for id_seq, seq in enumerate(output_slots):
                length = batch['slots_len'].tolist()[id_seq] - 1  # exclude the CLS token

                utt_ids = batch['utterances'][id_seq][:length].tolist()
                gt_ids = batch['y_slots'][id_seq][:length].tolist()
                gt_slots = [lang.id2slot[elem] for elem in gt_ids]
                utterance = [lang.id2word[elem] for elem in utt_ids]

                to_decode = seq[:length].tolist()
                ref_slots.append([(utterance[id_el], elem) for id_el, elem in enumerate(gt_slots)])
                hyp_slots.append([(utterance[id_el], lang.id2slot[elem]) for id_el, elem in enumerate(to_decode)])

    try:
        results = evaluate(ref_slots, hyp_slots)
    except Exception as ex:
        # the model may predict a slot class absent from the references, which conll.evaluate rejects
        print("Warning:", ex)
        results = {"total": {"f": 0}}

    report_intent = classification_report(ref_intents, hyp_intents, zero_division=False, output_dict=True)
    return results, report_intent, loss_array


def train_NN(train_loader, dev_loader, test_loader, model, optimizer, criterion_slots,
             criterion_intents, lang, n_epochs=200, patience=3):
    """
    Trains `model` with early stopping on dev-set slot F1, then reports test performance.

    Reusable across every experiment of Part 2.A (baseline, hyperparameter
    search, dropout) so main.py doesn't repeat the training loop.

    Args:
        train_loader, dev_loader, test_loader: DataLoaders for each split.
        model: the GPT2 model to train.
        optimizer: optimizer updating the model's parameters.
        criterion_slots, criterion_intents: loss functions.
        lang: utils.Lang instance, needed by eval_loop.
        n_epochs: maximum number of training epochs.
        patience: number of consecutive non-improving evals before stopping early.
    Returns:
        Tuple (results_test, intent_test, history), where history is a dict
        with "sampled_epochs", "losses_train", "losses_dev".
    """
    losses_train, losses_dev, sampled_epochs = [], [], []
    best_f1 = 0
    current_patience = patience

    pbar = tqdm(range(n_epochs))
    for i in pbar:
        loss = train_loop(train_loader, optimizer, criterion_slots, criterion_intents, model)
        sampled_epochs.append(i)
        losses_train.append(np.asarray(loss).mean())

        results_dev, intent_res, loss_dev = eval_loop(
            dev_loader, criterion_slots, criterion_intents, model, lang
        )
        losses_dev.append(np.asarray(loss_dev).mean())
        pbar.set_description(
            f"Slot F1: {results_dev['total']['f']:.2f}; Intent Acc: {intent_res['accuracy']:.2f}"
        )

        f1 = results_dev['total']['f']
        if f1 > best_f1:
            best_f1 = f1
            current_patience = patience
        else:
            current_patience -= 1
        if current_patience <= 0:
            break

    results_test, intent_test, _ = eval_loop(test_loader, criterion_slots, criterion_intents, model, lang)
    history = {"sampled_epochs": sampled_epochs, "losses_train": losses_train, "losses_dev": losses_dev}
    return results_test, intent_test, history


def save_model(path, epoch, model, optimizer, w2id, slot2id, intent2id):
    """Saves the model/optimizer state together with the vocabularies needed to reload it."""
    saving_object = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "w2id": w2id,
        "slot2id": slot2id,
        "intent2id": intent2id,
    }
    torch.save(saving_object, path)


def plot_train_and_valid_losses(sampled_epochs, losses_train, losses_dev):
    """Plots the training and dev loss curves collected by train_NN."""
    import matplotlib.pyplot as plt

    plt.figure(num=3, figsize=(8, 5)).patch.set_facecolor('white')
    plt.title('Train and Dev Losses')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.plot(sampled_epochs, losses_train, label='Train loss')
    plt.plot(sampled_epochs, losses_dev, label='Dev loss')
    plt.legend()
    plt.show()


def multiple_runs(train_loader, dev_loader, test_loader, lang, model_kwargs, lr,
                   device, n_epochs=200, runs=5, patience=3):
    """
    Trains the model from scratch `runs` times and reports mean +/- std test performance.

    Small corpora like ATIS give noisy single-run results, so the notebook
    recommends averaging over several random initializations.

    Args:
        train_loader, dev_loader, test_loader: DataLoaders for each split.
        lang: utils.Lang instance.
        model_kwargs: kwargs forwarded to model.GPT2 (besides the vocab/label sizes).
        lr: learning rate for AdamW.
        device: torch device to build each model on.
        n_epochs: maximum number of training epochs per run.
        runs: number of independent training runs.
        patience: early-stopping patience per run.
    Returns:
        Tuple (slot_f1s, intent_accs) as numpy arrays, one entry per run.
    """
    vocab_len = len(lang.word2id)
    slots_len = len(lang.id2slot)
    n_intents = len(lang.intent2id)

    slot_f1s, intent_acc = [], []
    for _ in tqdm(range(runs)):
        model = GPT2(vocab_len, slots_len, n_intents, **model_kwargs).to(device)
        model.apply(init_weights)

        optimizer = optim.AdamW(model.parameters(), lr=lr)
        criterion_slots = nn.CrossEntropyLoss(ignore_index=lang.pad_token)
        criterion_intents = nn.CrossEntropyLoss()

        results_test, intent_test, _ = train_NN(
            train_loader, dev_loader, test_loader, model, optimizer,
            criterion_slots, criterion_intents, lang, n_epochs=n_epochs, patience=patience,
        )
        intent_acc.append(intent_test['accuracy'])
        slot_f1s.append(results_test['total']['f'])

    slot_f1s = np.asarray(slot_f1s)
    intent_acc = np.asarray(intent_acc)
    print('Slot F1', round(slot_f1s.mean(), 3), '+-', round(slot_f1s.std(), 3))
    print('Intent Acc', round(intent_acc.mean(), 3), '+-', round(intent_acc.std(), 3))
    return slot_f1s, intent_acc
