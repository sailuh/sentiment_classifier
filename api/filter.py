import sys
import re
from unicodedata import category
from bs4 import BeautifulSoup
from markdown import markdown


USERNAME_REGEX = r"(\s|^)@(\S*\s?)"

# Generate Unicode punctuation set
punctuation = {chr(i) for i in range(sys.maxunicode + 1) if category(chr(i)).startswith(("P", "S"))}

# Dictionary to count token replacements
counters = {}

def remove_punctuation(text):
    """
    Remove all punctuation characters from the given text.

    Args:
        text (str): The input text.

    Returns:
        str: The text without any punctuation.
    """
    return "".join(char for char in text if char not in punctuation)

def clean_text(text):
    """
    Remove quoted text and large code blocks from GitHub issues or comments.

    This function performs the following clean-up:
    - Removes quoted email/notification text from GitHub.
    - Removes code blocks enclosed in triple backticks.

    Args:
        text (str): The input text (typically from a GitHub issue or comment).

    Returns:
        str: The cleaned text without quoted text or code blocks.
    """
    # Remove quoted text from emails/notifications
    text = re.sub(r"^(On[\s\S]*?notifications@github\.com\s*?wrote:\s*?)?(^(\>).*\s)*", '', text, flags=re.MULTILINE)

    # Remove code blocks enclosed in triple backticks
    text = re.sub(r"```[a-z]*\n[\s\S]*?\n```", "", text)

    return text

def remove_markdown_content(text):
    """
    Converts Markdown content to plain text by removing all Markdown formatting.

    This function processes the input Markdown text and converts it to plain text
    by removing all Markdown syntax.

    Args:
        text (str): The input Markdown text.

    Returns:
        str: Cleaned text without Markdown formatting.
    """
    html = markdown(text)
    return "".join(BeautifulSoup(html, "lxml").findAll(text=True))