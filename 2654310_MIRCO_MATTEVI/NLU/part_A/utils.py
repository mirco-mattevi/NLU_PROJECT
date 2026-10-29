# Data loading and preprocessing for the ATIS intent classification and slot filling dataset

import json
from collections import Counter

import torch
import torch.utils.data as data
from sklearn.model_selection import train_test_split


def load_data(path):
    """
    Loads an ATIS split from a JSON file.

    Args:
        path: path to the dataset.
    Returns:
        List of dicts, each with "utterance", "slots" and "intent" keys.
    """
    with open(path) as f:
        dataset = json.loads(f.read())
    return dataset


def create_dev_split(train_raw, portion=0.10):
    """
    Creates a stratified dev split out of the training set (ATIS has no dev set).

    Intents occurring only once are kept in the training set, since they can't
    be split across train/dev while preserving stratification (= proportions).

    Args:
        train_raw: list of ATIS training examples.
        portion: fraction of the (splittable) training data to use as dev set.
    Returns:
        Tuple (train_raw, dev_raw).
    """
    intents = [x['intent'] for x in train_raw] # stratify on intents
    count_y = Counter(intents)

    labels = []
    inputs = []
    mini_train = [] # intents to keep in the training set

    for id_y, y in enumerate(intents):
        if count_y[y] > 1: # only splittable intents (more than one occurrence)
            inputs.append(train_raw[id_y])
            labels.append(y)
        else:
            mini_train.append(train_raw[id_y]) # keep in the training set

    # Randomly split
    x_train, x_dev, _, _ = train_test_split(
        inputs, labels, test_size=portion, random_state=42, shuffle=True, stratify=labels
    )
    x_train.extend(mini_train)
    return x_train, x_dev


class Lang:
    """Builds and stores the word/slot/intent <-> id vocabularies, plus their inverse mappings."""

    def __init__(self, words, intents, slots, pad_token=0, cutoff=0, cls=True):
        """
        Args:
            words: list of training-set words (used to build word2id).
            intents: iterable of all intent labels (train+dev+test).
            slots: iterable of all slot labels (train+dev+test).
            pad_token: id reserved for padding.
            cutoff: minimum word frequency to be included in the vocabulary.
            cls: whether to reserve a CLS token/label (appended at the end of every utterance).
        """
        # init word/label -> id mappings
        self.pad_token = pad_token
        self.word2id = self.w2id(words, cutoff=cutoff, unk=True, cls=cls)
        self.slot2id = self.lab2id(slots, cls=cls)
        self.intent2id = self.lab2id(intents, pad=False, cls=False)

        # init id -> word/label mappings
        self.id2word = {v: k for k, v in self.word2id.items()}
        # cls will have the same id as the pad token
        self.id2slot = {v: k for k, v in self.slot2id.items() if not cls or k != 'cls'}
        self.id2intent = {v: k for k, v in self.intent2id.items()}

    def w2id(self, elements, cutoff=None, unk=True, cls=True):
        """Builds a word -> id mapping"""
        vocab = {'pad': self.pad_token}
        # len(vocab) gives unique and contiguous ids
        if unk:
            vocab['unk'] = len(vocab)
        if cls:
            vocab['cls'] = len(vocab)
        count = Counter(elements)
        for k, v in count.items():
            if v > cutoff: # only include words above the cutoff frequency
                vocab[k] = len(vocab)
        return vocab

    def lab2id(self, elements, pad=True, cls=True):
        """Builds a label -> id mapping. 'cls' shares the pad id (ignored as a slot)."""
        vocab = {}
        # len(vocab) gives unique and contiguous ids
        if pad:
            vocab['pad'] = self.pad_token
        for elem in elements:
            vocab[elem] = len(vocab)
        if cls:
            # when predicting the slots, we want to ignore the CLS
            # CLS will only be used for intent classification
            vocab['cls'] = self.pad_token
        return vocab


class IntentsAndSlots(data.Dataset):
    """Maps raw ATIS examples to id sequences, appending a CLS token to every utterance/slot sequence."""
    def __init__(self, dataset, lang, unk='unk', cls='cls', add_cls=True):
        self.utterances = [x['utterance'] for x in dataset]
        self.slots = [x['slots'] for x in dataset]
        self.intents = [x['intent'] for x in dataset]
        self.unk = unk
        self.cls = cls
        self.add_cls = add_cls

        self.utt_ids = self.mapping_seq(self.utterances, lang.word2id)
        self.slot_ids = self.mapping_seq(self.slots, lang.slot2id)
        self.intent_ids = self.mapping_lab(self.intents, lang.intent2id)

    def __len__(self):
        return len(self.utterances)

    def __getitem__(self, idx):
        utt = torch.Tensor(self.utt_ids[idx])
        slots = torch.Tensor(self.slot_ids[idx])
        intent = self.intent_ids[idx]
        return {'utterance': utt, 'slots': slots, 'intent': intent}

    def mapping_lab(self, data, mapper):
        """Maps labels to ids, falling back to 'unk' for unseen ones."""
        return [mapper[x] if x in mapper else mapper[self.unk] for x in data]

    def mapping_seq(self, data, mapper):
        """Maps whitespace-tokenized sequences to id sequences, appending CLS if requested."""
        res = []
        for seq in data:
            tmp_seq = [mapper[x] if x in mapper else mapper[self.unk] for x in seq.split()]
            if self.add_cls:
                tmp_seq.append(mapper[self.cls])
            res.append(tmp_seq)
        return res


def collate_fn(data, device, pad_token=0):
    """
    Pads a batch of variable-length utterances/slots and stacks the intents.

    Args:
        data: list of samples.
        device: torch device to move the batch tensors to.
        pad_token: pad id
    Returns:
        Dict with "utterances", "intents", "y_slots" and "slots_len" tensors.
    """
    def merge(sequences):
        '''
        merge from batch * sent_len to batch * max_len (= pad utterances/slots)
        Args:
            sequences: list of variable-length sequences
        '''
        lengths = [len(seq) for seq in sequences]
        max_len = 1 if max(lengths) == 0 else max(lengths)
        # matrix batch_size * maximum length of 0s (pad id)
        padded_seqs = torch.LongTensor(len(sequences), max_len).fill_(pad_token)
        # pad the sequences
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq
        return padded_seqs, lengths

    data_by_key = {}
    for key in data[0].keys(): # 'utterance', 'slots', 'intent'
        data_by_key[key] = [d[key] for d in data] # collect each key from every sample

    # We just need one length for packed pad seq, since len(utt) == len(slots)
    src_utt, _ = merge(data_by_key['utterance'])
    y_slots, y_lengths = merge(data_by_key['slots'])
    intent = torch.LongTensor(data_by_key['intent'])

    src_utt = src_utt.to(device)
    y_slots = y_slots.to(device)
    intent = intent.to(device)
    y_lengths = torch.LongTensor(y_lengths).to(device)

    return {
        "utterances": src_utt,
        "intents": intent,
        "y_slots": y_slots,
        "slots_len": y_lengths,
    }
