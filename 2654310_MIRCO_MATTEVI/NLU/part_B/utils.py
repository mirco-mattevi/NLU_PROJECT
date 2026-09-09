# Data loading and preprocessing for fine-tuning pretrained GPT2/BERT on ATIS
# (Part 2.B). Loading is shared with Part 2.A; what's new here is aligning
# word-level slot labels to the sub-word tokens produced by each model's own
# tokenizer, following https://arxiv.org/abs/1902.10909 (Joint BERT).

import json
from collections import Counter

from sklearn.model_selection import train_test_split


def load_data(path):
    """
    Loads an ATIS split from a JSON file.

    Args:
        path: path to the dataset file.
    Returns:
        List of dicts, each with "utterance", "slots" and "intent" keys.
    """
    with open(path) as f:
        dataset = json.loads(f.read())
    return dataset


def create_dev_split(train_raw, portion=0.10):
    """
    Carves a stratified dev split out of the training set (ATIS has no dev set).

    Intents occurring only once are kept in the training set, since they can't
    be split across train/dev while preserving stratification.

    Args:
        train_raw: list of ATIS training examples.
        portion: fraction of the (splittable) training data to use as dev set.
    Returns:
        Tuple (train_raw, dev_raw).
    """
    intents = [x['intent'] for x in train_raw]
    count_y = Counter(intents)

    labels, inputs, mini_train = [], [], []
    for id_y, y in enumerate(intents):
        if count_y[y] > 1:
            inputs.append(train_raw[id_y])
            labels.append(y)
        else:
            mini_train.append(train_raw[id_y])

    x_train, x_dev, _, _ = train_test_split(
        inputs, labels, test_size=portion, random_state=42, shuffle=True, stratify=labels
    )
    x_train.extend(mini_train)
    return x_train, x_dev


def build_label_vocab(slots, intents):
    """
    Builds the slot/intent label -> id vocabularies (unlike Part 2.A, no word
    vocabulary is needed here: tokenization is delegated to the pretrained
    model's own tokenizer).

    Args:
        slots: iterable of all slot labels appearing in the corpus.
        intents: iterable of all intent labels appearing in the corpus.
    Returns:
        Tuple (slot2id, intent2id).
    """
    slot2id = {slot: i for i, slot in enumerate(sorted(set(slots)))}
    intent2id = {intent: i for i, intent in enumerate(sorted(set(intents)))}
    return slot2id, intent2id


def align_labels_with_tokens(word_ids, word_labels, pad_label=-100):
    """
    Projects word-level slot labels onto a tokenizer's sub-word tokens.

    Following the Joint BERT paper, only the first sub-token of each word
    keeps its slot label; every other sub-token (and special tokens such as
    [CLS]/[SEP]/pad, which have `word_ids` entry None) gets `pad_label`, so
    the loss (with ignore_index=pad_label) never scores them.

    Args:
        word_ids: output of a fast tokenizer's `BatchEncoding.word_ids(i)` for
            one example: one entry per sub-word token, None for special tokens.
        word_labels: slot label ids, one per whitespace-separated word.
        pad_label: label id to ignore in the loss (CrossEntropyLoss ignore_index).
    Returns:
        List of label ids, one per sub-word token (same length as word_ids).
    """
    # TODO (exercise 2.B): for each id in word_ids, emit word_labels[id] the
    # first time that word index is seen, and pad_label for every subsequent
    # sub-token of the same word (and for None entries).
    raise NotImplementedError
