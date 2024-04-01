import json
from types import SimpleNamespace


def load_vocab(vocab_path: str) -> dict:
    """
    Loads and returns the vocabulary from a specified JSON configuration file.

    The function expects the JSON file to contain a dictionary with a key 'model',
    which itself contains a key 'vocab' that maps tokens to their respective IDs.

    Parameters:
    - vocab_path (str): The path to the JSON file containing the vocabulary.

    Returns:
    - dict: A dictionary representing the vocabulary, where keys are tokens and values are their corresponding IDs.
    """
    with open(vocab_path, 'r', encoding='utf-8') as file:
        config = json.load(file)
    vocab = config['model']['vocab']
    return vocab


def tokenize_custom(tokenizer, tokenizer_config_path: str, input_text: str, sub_words_fraction: int = 2):
    """
    Performs tokenization of the input text using both default and sub-word tokenization methods.

    Default tokenization is performed directly using the provided tokenizer.
    Sub-word tokenization attempts to further divide tokens into smaller sub-tokens, based on the
    sub_words_fraction parameter. This aims to create shorter tokens than the default tokenization,
    using the same vocabulary as the tokenizer.

    Parameters:
    - tokenizer: An instance of a tokenizer that supports the tokenize method for default tokenization.
    - tokenizer_config_path (str): The path to the JSON configuration file containing the tokenizer vocabulary.
    - input_text (str): The text to be tokenized.
    - sub_words_fraction (int, optional): Determines the granularity of sub-word tokenization. A higher value
      results in shorter sub-tokens. Default is 2.

    Returns:
    - SimpleNamespace: An object with two attributes, 'default_tokens' and 'sub_word_tokens', containing the results
      of default and sub-word tokenization, respectively.
    """
    # Load tokenizer vocabulary
    vocab = load_vocab(tokenizer_config_path)

    # Default tokenization
    default_tokens = tokenizer.tokenize(input_text)

    # Function to find sub-tokens in vocab
    def find_sub_tokens(token, fraction):
        sub_tokens = []
        sub_token_length = max(1, len(token) // fraction)

        for i in range(0, len(token), sub_token_length):
            sub_token = token[i:i + sub_token_length]
            # If sub-token is not directly in vocab, we need a more complex strategy
            # For simplicity, we are just checking direct matches here
            if sub_token in vocab:
                sub_tokens.append(sub_token)
            else:
                # If sub-token not found, use the original token (not ideal, needs better handling)
                sub_tokens.append(token)
                break
        return sub_tokens

    # Sub-word tokenization
    sub_word_tokens = []
    for token in default_tokens:
        # Extend sub_word_tokens with the sub-tokens found
        sub_word_tokens.extend(find_sub_tokens(token, sub_words_fraction))

    return SimpleNamespace(default_tokens=default_tokens, sub_word_tokens=sub_word_tokens)
