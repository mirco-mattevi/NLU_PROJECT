import os
from functools import partial
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from functions import run_lr_tuning, run_hyperparameter_tuning, run_lr_sensitivity_check
from utils import PennTreeBank, collate_fn, read_file

if __name__ == "__main__":
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    DATASET_DIR = os.path.join("dataset", "PennTreeBank")

    train_raw = read_file(os.path.join(DATASET_DIR, "ptb.train.txt"))
    dev_raw = read_file(os.path.join(DATASET_DIR, "ptb.valid.txt"))
    test_raw = read_file(os.path.join(DATASET_DIR, "ptb.test.txt"))

    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    vocab_len = len(tokenizer) # number of ids in the tokenizer's vocabulary

    train_dataset = PennTreeBank(train_raw)
    dev_dataset = PennTreeBank(dev_raw)
    test_dataset = PennTreeBank(test_raw)

    collate = partial(collate_fn, tokenizer=tokenizer, device=DEVICE)
    train_loader = DataLoader(train_dataset, batch_size=8, collate_fn=collate, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=16, collate_fn=collate)
    test_loader = DataLoader(test_dataset, batch_size=16, collate_fn=collate)

    criterion_train = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    criterion_eval = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    # 0: baseline, learning rate tuning (already run once, lr=0.001 is the winner - not rerun)
    lr = 0.001
    # best_model, best_ppl, test_ppl = run_lr_tuning(
    #     vocab_len, lr, DEVICE, train_loader, dev_loader, test_loader, criterion_train, criterion_eval
    # )

    # 1: hyperparameter tuning (d_model, n_heads, num_layers, ff_dim) (already run once - not rerun)
    # best_model, best_ppl, test_ppl, best_config = run_hyperparameter_tuning(
    #     vocab_len, lr, DEVICE, train_loader, dev_loader, test_loader, criterion_train, criterion_eval,
    # )

    # 1b: lr sensitivity spot-check - does the fixed lr=0.001 unfairly penalize the widest/
    # deepest candidates (d_model=256, num_layers=6) of experiment 1?
    run_lr_sensitivity_check(
        vocab_len, DEVICE, train_loader, dev_loader, test_loader, criterion_train, criterion_eval
    )

    # ---- Experiment 2: dropout ----
    # TODO: requires the dropout layers marked as TODO in model.py to be implemented first.

    # ---- Experiment 3: weight tying ----
    # TODO: requires lm_head.weight = token_embed.weight to be set in model.py first.
