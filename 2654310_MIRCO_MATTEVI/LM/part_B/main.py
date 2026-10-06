import os
from functools import partial

import torch
from torch.utils.data import DataLoader

from functions import load_tokenizer, run_lr_tuning, run_rank_tuning, run_alpha_tuning
from utils import PennTreeBank, collate_fn, read_file

if __name__ == "__main__":
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    DATASET_DIR = os.path.join("dataset", "PennTreeBank")

    train_raw = read_file(os.path.join(DATASET_DIR, "ptb.train.txt"))
    dev_raw = read_file(os.path.join(DATASET_DIR, "ptb.valid.txt"))
    test_raw = read_file(os.path.join(DATASET_DIR, "ptb.test.txt"))

    tokenizer = load_tokenizer()

    train_dataset = PennTreeBank(train_raw)
    dev_dataset = PennTreeBank(dev_raw)
    test_dataset = PennTreeBank(test_raw)

    collate = partial(collate_fn, tokenizer=tokenizer, device=DEVICE)
    train_loader = DataLoader(train_dataset, batch_size=8, collate_fn=collate, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=16, collate_fn=collate)
    test_loader = DataLoader(test_dataset, batch_size=16, collate_fn=collate)

    # Requirement: test PPL must be < 250 and lower than Part 1.A's best.

    # 0: learning rate tuning (rank=8, alpha=16 fixed)
    best_model, best_ppl, test_ppl, lr = run_lr_tuning(
        [1e-4, 3e-4, 1e-3], 8, 16, DEVICE, train_loader, dev_loader, test_loader
    )

    # 1: rank tuning (alpha = 2 * rank)
    best_model, best_ppl, test_ppl, rank = run_rank_tuning(
        [4, 8, 16, 32], lr, DEVICE, train_loader, dev_loader, test_loader
    )

    # 2: alpha tuning
    best_model, best_ppl, test_ppl, alpha = run_alpha_tuning(
        [rank, 2 * rank, 4 * rank], rank, lr, DEVICE, train_loader, dev_loader, test_loader
    )

    print(f"[LoRA] best: lr={lr} rank={rank} alpha={alpha} | dev PPL: {best_ppl:.2f} | test PPL: {test_ppl:.2f}")
