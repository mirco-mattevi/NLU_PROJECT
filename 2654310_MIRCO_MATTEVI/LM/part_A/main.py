# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

import os
from functools import partial

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from functions import init_weights, train_model
from model import GPT2
from utils import PennTreeBank, collate_fn, read_file

if __name__ == "__main__":
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    DATASET_DIR = os.path.join("dataset", "PennTreeBank")

    train_raw = read_file(os.path.join(DATASET_DIR, "ptb.train.txt"))
    dev_raw = read_file(os.path.join(DATASET_DIR, "ptb.valid.txt"))
    test_raw = read_file(os.path.join(DATASET_DIR, "ptb.test.txt"))

    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    vocab_len = len(tokenizer)

    train_dataset = PennTreeBank(train_raw)
    dev_dataset = PennTreeBank(dev_raw)
    test_dataset = PennTreeBank(test_raw)

    collate = partial(collate_fn, tokenizer=tokenizer, device=DEVICE)
    train_loader = DataLoader(train_dataset, batch_size=8, collate_fn=collate, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=16, collate_fn=collate)
    test_loader = DataLoader(test_dataset, batch_size=16, collate_fn=collate)

    criterion_train = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    criterion_eval = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    # ---- Experiment 0: baseline, find a suitable learning rate ----
    lr = 0.001
    model = GPT2(vocab_len, pos_emb_size=1024, d_model=20, n_heads=1, num_layers=1, ff_dim=20).to(DEVICE)
    model.apply(init_weights)
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    best_model, best_ppl, test_ppl, _ = train_model(
        model, optimizer, criterion_train, criterion_eval, train_loader, dev_loader, test_loader
    )
    print(f"[Experiment 0 - baseline] lr={lr} | dev PPL: {best_ppl:.2f} | test PPL: {test_ppl:.2f}")

    # ---- Experiment 1: hyperparameter search (d_model, n_heads, num_layers, ff_dim) ----
    # TODO: change one hyperparameter at a time, re-tuning lr for larger models, and
    # print the PPL for each configuration, keeping only the ones that improve it.

    # ---- Experiment 2: dropout ----
    # TODO: requires the dropout layers marked as TODO in model.py to be implemented first.

    # ---- Experiment 3: weight tying ----
    # TODO: requires lm_head.weight = token_embed.weight to be set in model.py first.
