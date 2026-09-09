# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

import os

from utils import build_label_vocab, create_dev_split, load_data

if __name__ == "__main__":
    DATASET_DIR = os.path.join("dataset", "ATIS")

    tmp_train_raw = load_data(os.path.join(DATASET_DIR, "train.json"))
    test_raw = load_data(os.path.join(DATASET_DIR, "test.json"))
    train_raw, dev_raw = create_dev_split(tmp_train_raw, portion=0.10)

    corpus = train_raw + dev_raw + test_raw
    slots = [line['slots'] for line in corpus]
    intents = [line['intent'] for line in corpus]
    flat_slots = [s for line in slots for s in line.split()]
    slot2id, intent2id = build_label_vocab(flat_slots, intents)

    # TODO: for each backbone ("openai-community/gpt2" -> JointGPT2,
    # "bert-base-uncased" -> JointBERT):
    #   1. tokenize train/dev/test with the model's own AutoTokenizer
    #      (is_split_into_words=True), then align slot labels to sub-tokens
    #      with utils.align_labels_with_tokens.
    #   2. build DataLoaders over the tokenized batches.
    #   3. fine-tune with functions.train_loop / functions.eval_loop.
    #   4. print intent accuracy and slot F1 for each model.
