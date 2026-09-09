# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from functions import init_weights, train_NN
from model import GPT2
from utils import IntentsAndSlots, Lang, collate_fn, create_dev_split, load_data

if __name__ == "__main__":
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    PAD_TOKEN = 0
    DATASET_DIR = os.path.join("dataset", "ATIS")

    tmp_train_raw = load_data(os.path.join(DATASET_DIR, "train.json"))
    test_raw = load_data(os.path.join(DATASET_DIR, "test.json"))
    train_raw, dev_raw = create_dev_split(tmp_train_raw, portion=0.10)

    words = sum([x['utterance'].split() for x in train_raw], [])
    corpus = train_raw + dev_raw + test_raw
    slots = set(sum([line['slots'].split() for line in corpus], []))
    intents = set([line['intent'] for line in corpus])
    lang = Lang(words, intents, slots, pad_token=PAD_TOKEN, cutoff=0)

    train_dataset = IntentsAndSlots(train_raw, lang)
    dev_dataset = IntentsAndSlots(dev_raw, lang)
    test_dataset = IntentsAndSlots(test_raw, lang)

    def collate(batch):
        return collate_fn(batch, device=DEVICE, pad_token=PAD_TOKEN)

    train_loader = DataLoader(train_dataset, batch_size=128, collate_fn=collate, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=64, collate_fn=collate)
    test_loader = DataLoader(test_dataset, batch_size=64, collate_fn=collate)

    vocab_len = len(lang.word2id)
    slots_len = len(lang.id2slot)
    n_intents = len(lang.intent2id)

    # ---- Experiment 0: baseline, find a suitable learning rate ----
    lr = 0.0005
    model = GPT2(
        vocab_len, slots_len, n_intents,
        pos_emb_size=1024, d_model=20, n_heads=1, num_layers=1, ff_dim=20,
    ).to(DEVICE)
    model.apply(init_weights)

    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion_slots = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
    criterion_intents = nn.CrossEntropyLoss()

    results_test, intent_test, _ = train_NN(
        train_loader, dev_loader, test_loader, model, optimizer,
        criterion_slots, criterion_intents, lang,
    )
    print(f"[Experiment 0 - baseline] lr={lr}")
    print(f"Slot F1: {results_test['total']['f']:.3f}")
    print(f"Intent Accuracy: {intent_test['accuracy']:.3f}")

    # ---- Experiment 1: hyperparameter search (d_model, n_heads, num_layers, ff_dim) ----
    # TODO: change one hyperparameter at a time, re-tuning lr for larger models.

    # ---- Experiment 2: dropout before the final output layers ----
    # TODO: requires the dropout layer marked as TODO in model.py to be implemented first.
