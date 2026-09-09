# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

import os
from functools import partial

import torch
from torch.utils.data import DataLoader

from functions import load_tokenizer_and_model, param_stats, prepare_optimizer, train_model
from utils import PennTreeBank, collate_fn, read_file

if __name__ == "__main__":
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    DATASET_DIR = os.path.join("dataset", "PennTreeBank")

    train_raw = read_file(os.path.join(DATASET_DIR, "ptb.train.txt"))
    dev_raw = read_file(os.path.join(DATASET_DIR, "ptb.valid.txt"))
    test_raw = read_file(os.path.join(DATASET_DIR, "ptb.test.txt"))

    rank, alpha, lr = 8, 16, 1e-4
    tokenizer, model = load_tokenizer_and_model(rank, alpha, DEVICE)

    train_dataset = PennTreeBank(train_raw)
    dev_dataset = PennTreeBank(dev_raw)
    test_dataset = PennTreeBank(test_raw)

    collate = partial(collate_fn, tokenizer=tokenizer, device=DEVICE)
    train_loader = DataLoader(train_dataset, batch_size=8, collate_fn=collate, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=16, collate_fn=collate)
    test_loader = DataLoader(test_dataset, batch_size=16, collate_fn=collate)

    optimizer = prepare_optimizer(model, lr)
    param_stats(model)

    # TODO: once the LoRA TODOs in model.py and functions.py are implemented,
    # run the training and print the resulting PPL:
    # best_model, best_ppl, test_ppl = train_model(
    #     model, optimizer, train_loader, dev_loader, test_loader
    # )
    # print(f"[LoRA] rank={rank} alpha={alpha} lr={lr} | dev PPL: {best_ppl:.2f} | test PPL: {test_ppl:.2f}")
    # Requirement: test PPL must be < 250 and lower than Part 1.A's best.
