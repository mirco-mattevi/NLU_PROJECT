
import os
from functools import partial

import torch
from torch.utils.data import DataLoader


from functions import run_tuning
from utils import IntentsAndSlots, Lang, collate_fn, create_dev_split, load_data

if __name__ == "__main__":
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    PAD_TOKEN = 0
    DATASET_DIR = os.path.join("dataset", "ATIS")

    tmp_train_raw = load_data(os.path.join(DATASET_DIR, "train.json"))
    test_raw = load_data(os.path.join(DATASET_DIR, "test.json"))
    train_raw, dev_raw = create_dev_split(tmp_train_raw, portion=0.10)

    words = sum([x['utterance'].split() for x in train_raw], []) # get all words in the training set utterances
    corpus = train_raw + dev_raw + test_raw 
    # sorted: the order of a set of strings changes at every Python start, and so would the label ids
    slots = sorted(set(sum([line['slots'].split() for line in corpus], []))) # get all slots in the corpus
    intents = sorted(set([line['intent'] for line in corpus])) # get all intents in the corpus
    lang = Lang(words, intents, slots, pad_token=PAD_TOKEN, cutoff=0)

    train_dataset = IntentsAndSlots(train_raw, lang)
    dev_dataset = IntentsAndSlots(dev_raw, lang)
    test_dataset = IntentsAndSlots(test_raw, lang)

    def collate(batch):
        return collate_fn(batch, device=DEVICE, pad_token=PAD_TOKEN)

    train_loader = DataLoader(train_dataset, batch_size=128, collate_fn=collate, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=64, collate_fn=collate)
    test_loader = DataLoader(test_dataset, batch_size=64, collate_fn=collate)

    os.makedirs("bin", exist_ok=True)
    os.makedirs("plots", exist_ok=True)

    tune = partial(
        run_tuning, lang=lang, device=DEVICE,
        train_loader=train_loader, dev_loader=dev_loader, test_loader=test_loader,
    )

    LRS = [0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]

    # baseline configuration
    config = {"lr": 0.001, "d_model": 20, "n_heads": 1, "num_layers": 1, "ff_dim": 20, "dropout": 0.0}
    best_f1 = -1

    # 0: baseline, learning rate tuning
    config, best_f1, _ = tune("lr", LRS, config, best_f1)

    # 1: hyperparameter tuning, one at a time (d_model, n_heads, num_layers, ff_dim)
    # after d_model and num_layers, the larger models are also trained with the 3 lr smaller than the current
    config, best_f1, _ = tune("d_model", [32, 64, 128, 256, 512], config, best_f1)
    i = LRS.index(config["lr"])
    lower_lrs = LRS[i + 1:i + 4]
    for d_model in [128, 256, 512]:
        candidate, f1, res = tune("lr", lower_lrs, {**config, "d_model": d_model}, best_f1)
        if res is not None: # f1 has improved
            config, best_f1 = candidate, f1

    config, best_f1, _ = tune("n_heads", [h for h in [2, 4, 8, 16] if config["d_model"] % h == 0], config, best_f1)

    config, best_f1, _ = tune("num_layers", [2, 3, 4, 6, 8], config, best_f1)
    i = LRS.index(config["lr"])
    lower_lrs = LRS[i + 1:i + 4]
    for num_layers in [4, 6, 8]:
        candidate, f1, res = tune("lr", lower_lrs, {**config, "num_layers": num_layers}, best_f1)
        if res is not None:
            config, best_f1 = candidate, f1

    config, best_f1, _ = tune("ff_dim", [m * config["d_model"] for m in [1, 2, 4, 8]], config, best_f1)

    # readjust the learning rate for the final model: one value bigger and one smaller the current one
    i = LRS.index(config["lr"])
    config, best_f1, _ = tune("lr", LRS[max(i - 1, 0):i] + LRS[i + 1:i + 2], config, best_f1)

    # 2: dropout before the final output layers
    # TODO: requires the dropout layer marked as TODO in model.py to be implemented first.
