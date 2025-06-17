from torch.nn import CrossEntropyLoss
from transformers import (AutoModelForTokenClassification,
                          BertTokenizerFast)


def get_loss_fn(loss_fn=CrossEntropyLoss()):
    """
    Get the loss function for token classification.
    """
    def _loss_fn(predictions, labels):
        """
        Compute the loss for token classification.
        """
        ignore = labels != -100

        # Discard the predictions and labels for the ignored tokens.
        # Consider only the ground truth labels to discover what are
        # the tokens to be ignored.
        active_labels = labels[ignore]
        active_predictions = predictions[ignore]

        # Compute the loss
        return loss_fn(active_predictions, active_labels)

    return _loss_fn


def get_bert_tokenizer(model_name="bert-base-cased"):
    """
    Load a BERT tokenizer.
    """
    return BertTokenizerFast.from_pretrained(model_name)


def get_bert_model(model_name, num_labels):
    """
    Load a BERT model for token classification.
    """
    if model_name == 'bert_tiny':
        return get_tiny_bert_model_for_classification(num_labels)
    elif model_name == 'bert_mini':
        return get_mini_bert_model_for_classification(num_labels)
    elif model_name == 'bert_small':
        return get_small_bert_model_for_classification(num_labels)
    elif model_name == 'blue_bert':
        return get_blue_bert_model_for_classification(num_labels)


def get_tiny_bert_model_for_classification(num_labels):
    return AutoModelForTokenClassification.from_pretrained(
        "prajjwal1/bert-tiny", num_labels=num_labels
    )


def get_mini_bert_model_for_classification(num_labels):
    return AutoModelForTokenClassification.from_pretrained(
        "prajjwal1/bert-mini", num_labels=num_labels
    )


def get_small_bert_model_for_classification(num_labels):
    return AutoModelForTokenClassification.from_pretrained(
        "prajjwal1/bert-small", num_labels=num_labels
    )


def get_blue_bert_model_for_classification(num_labels):
    return AutoModelForTokenClassification.from_pretrained(
        "bionlp/bluebert_pubmed_uncased_L-12_H-768_A-12", num_labels=num_labels
    )


def tokenize_and_align_labels(tokenizer, examples, label_all_tokens=True):
    """
    Tokenize all words in the examples and align the labels with the tokenized inputs.

    Args:
        tokenizer: The tokenizer to use.
        examples: The examples to tokenize. It's a dictionary with two keys: "tokens" and "ner_tags".
        The tokens are the word list and the ner_tags are the corresponding NER tags.
        label_all_tokens: Whether or not to set the labels of all tokens.
    """
    tokenized_inputs = tokenizer(
        examples["tokens"],
        is_split_into_words=True,
        truncation=True,
        padding=True,
    )
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        # word_ids() => Return a list mapping the tokens
        # to their actual word in the initial sentence.
        # It Returns a list indicating the word corresponding to each token.
        previous_word_idx = None
        label_ids = []
        # Special tokens like `` and `<\s>` are originally mapped to None
        # We need to set the label to -100 so they are automatically ignored in the loss function.
        for word_idx in word_ids:
            if word_idx is None:
                # set –100 as the label for these special tokens
                label_ids.append(-100)
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            elif word_idx != previous_word_idx:
                # if current word_idx is != prev then its the most regular case
                # and add the corresponding token
                label_ids.append(label[word_idx])
            else:
                # to take care of sub-words which have the same word_idx
                # set -100 as well for them, but only if label_all_tokens == False
                label_ids.append(label[word_idx] if label_all_tokens else -100)
                # mask the subword representations after the first subword

            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs


if __name__ == "__main__":
    from data import load_data
    # Example usage
    tokenizer = get_bert_tokenizer("bert-base-cased")
    print(f"Tokenizer: {tokenizer}")

    # Example input
    dataset = load_data('BC2GM')
    tokenized_data = dataset["train"].map(
        lambda batch: tokenize_and_align_labels(tokenizer, batch), batched=True
    )

    # Example output
    example = tokenized_data[1]
    print(f"Tokenized example: {example}")

    # Generate words from input IDs
    input_ids = example["input_ids"]
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    print(f"Tokens: {tokens}")
    print(f"Length of tokens: {len(tokens)}")
