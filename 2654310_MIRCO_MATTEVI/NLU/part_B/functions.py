# Fine-tuning helpers for the pretrained GPT2/BERT joint models (Part 2.B).
# No notebook skeleton exists for this part: train_loop/eval_loop are
# documented stubs, while test_huggingface/test_huggingface_bert (moved here
# from LAB 05) are a working smoke test showing how to load each backbone and
# inspect its tokenizer/output shapes before writing the real training code.

from pprint import pprint

from transformers import AutoModel, AutoTokenizer


def test_huggingface():
    """Smoke test: loads pretrained GPT2 and inspects its tokenizer/hidden-state shapes."""
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModel.from_pretrained("openai-community/gpt2")

    inputs = tokenizer(
        ["I saw a man with a telescope", "StarLord was here", "I didn't"],
        return_tensors="pt", padding=True,
    )
    pprint(inputs)

    outputs = model(**inputs)
    print(outputs.last_hidden_state.shape)

    print(inputs["input_ids"][0])
    print(tokenizer.convert_ids_to_tokens(inputs["input_ids"][1]))


def test_huggingface_bert():
    """Smoke test: loads pretrained BERT and inspects its tokenizer/hidden-state shapes."""
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased")

    inputs = tokenizer(
        ["I saw a man with a telescope", "StarLord was here", "I didn't"],
        return_tensors="pt", padding=True,
    )
    pprint(inputs)

    outputs = model(**inputs)
    print(outputs.last_hidden_state.shape)

    print(inputs["input_ids"][0])
    print(tokenizer.convert_ids_to_tokens(inputs["input_ids"][1]))


def train_loop(data, optimizer, model):
    """
    Runs one training epoch of joint intent/slot fine-tuning.

    Args:
        data: DataLoader yielding tokenized batches (input_ids, attention_mask,
            slot labels aligned to sub-tokens via utils.align_labels_with_tokens
            with ignore_index=-100, and intent labels).
        optimizer: optimizer updating the model's parameters.
        model: a JointBERT or JointGPT2 instance.
    Returns:
        List of per-batch joint losses.
    """
    # TODO (exercise 2.B): forward the batch through `model`, compute
    # CrossEntropyLoss(ignore_index=-100) on the slot logits and a plain
    # CrossEntropyLoss on the intent logits, sum them, and run the usual
    # zero_grad/backward/step steps (see NLU/part_A's train_loop).
    raise NotImplementedError


def eval_loop(data, model, lang):
    """
    Evaluates the model on `data`: slot F1 (conll) and intent classification report.

    Sub-token predictions must be collapsed back to one label per word before
    scoring (keep only each word's first sub-token, matching how the labels
    were aligned in utils.align_labels_with_tokens).

    Args:
        data: DataLoader yielding tokenized batches.
        model: a JointBERT or JointGPT2 instance.
        lang: object exposing id2slot/id2intent (see NLU/part_A's Lang).
    Returns:
        Tuple (slot_results, intent_report, loss_array), same shape as
        NLU/part_A's eval_loop.
    """
    # TODO (exercise 2.B): mirror NLU/part_A's eval_loop, but decode only the
    # first sub-token of every word (skip positions whose label is -100)
    # before calling conll.evaluate.
    raise NotImplementedError
