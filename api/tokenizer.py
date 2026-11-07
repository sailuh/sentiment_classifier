import re
import warnings

# Dictionary to count token replacements
counters = {}


def replace_token(regex, token_name, text):
    """
    Replace matched patterns in the text with the specified token.

    This function uses regular expressions to find occurrences of the pattern
    and replaces them with a token name. The number of replacements made is counted.

    Args:
        regex (str): The regular expression pattern to match.
        token_name (str): The replacement token name.
        text (str): The input text.

    Returns:
        tuple: A tuple containing:
            - str: The text with the tokens replacing the matches.
            - int: The number of replacements made.
    """
    replaced_text, replacements = re.subn(regex, f" {token_name} ", text, flags=re.MULTILINE)
    counters[token_name] = counters.get(token_name, 0) + replacements
    return replaced_text, replacements

def tokenize_text(text):
    """
    Tokenizes a given text by replacing specific elements such as emails, mentions, URLs, etc.

    This function processes the input text and replaces various elements, such as:
    - Email addresses (replaced with 'MEMAIL').
    - GitHub mentions (replaced with 'MMENTION').
    - Code blocks (replaced with 'MICODE').
    - Version numbers (replaced with 'MVERSIONNUMBER').
    - Issue mentions (replaced with 'MISSUEMENTION').
    - URLs (replaced with 'MURL').

    Args:
        text (str): The input text.

    Returns:
        tuple: A tuple containing:
            - str: The tokenized text.
            - int: The total number of replacements made.
    """
    total_replacements = 0

    text, replacements = replace_token(r"\S+@\S*\s?", "MEMAIL", text)
    total_replacements += replacements

    text, replacements = replace_token(USERNAME_REGEX, "MMENTION", text)
    total_replacements += replacements

    text, replacements = replace_token(r"`([^`]*)`", "MICODE", text)
    total_replacements += replacements

    text, replacements = replace_token(r"\b\d+\.\d+(\.\d+)*\b", "MVERSIONNUMBER", text)
    total_replacements += replacements

    text, replacements = replace_token(r"(\s|^)#\d+", "MISSUEMENTION", text)
    total_replacements += replacements

    text, replacements = replace_token(
        r"([a-zA-Z0-9]+):\/\/([\w_-]+(?:\.[\w_-]+)*)[\w.,@?^=%&:\/~+#-]*[\w@?^=%&\/~+#-]",
        "MURL",
        text,
    )
    total_replacements += replacements

    return text, total_replacements

def transform_text(row):
    """
    Transforms a row by cleaning and tokenizing its text content.

    This function extracts the "Text" key from the input dictionary and processes
    it using the `tokenize_text` function. The text is also cleaned by removing
    newline characters.

    Args:
        row (dict): A dictionary containing a 'Text' key.

    Returns:
        tuple: A tuple containing:
            - str: The processed text after cleaning and tokenization.
            - int: The number of replacements made.
    """
    text = row.get("Text", "")

    if not isinstance(text, str):
        warnings.warn(f"Converting non-string type to string: {type(text)}")
        text = str(text)

    text, replaced_count = tokenize_text(text)
    text = text.replace("\n", "")
    return text, replaced_count