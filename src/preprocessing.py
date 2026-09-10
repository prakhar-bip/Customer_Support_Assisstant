"""
Preprocessing and Text Normalization Module for Customer Support Messages.

Provides clean_customer_message() to sanitize incoming customer queries
prior to intent classification, semantic retrieval, and response generation.
"""

import re
import unicodedata
from typing import Optional

def clean_customer_message(text: Optional[str]) -> str:
    """
    Sanitize and normalize an incoming customer message.

    Operations performed:
      1. Handles null, empty, or non-string inputs safely.
      2. Unicode normalization (NFKC) to resolve special/accented characters.
      3. Strips non-printable and invisible control characters.
      4. Collapses repetitive whitespace, tabs, and newlines into single spaces.
      5. Preserves essential punctuation, emojis, and case nuance (important for sentiment).

    Args:
        text (str, optional): Raw incoming customer query.

    Returns:
        str: Cleaned, normalized message string.
    """
    if text is None:
        return ""

    if not isinstance(text, str):
        text = str(text)

    # 1. Unicode normalization
    text = unicodedata.normalize("NFKC", text)

    # 2. Remove invisible control characters (except standard whitespace like space, tab, newline)
    text = "".join(ch for ch in text if ch.isspace() or not unicodedata.category(ch).startswith("C"))

    # 3. Normalize multiple spaces / tabs / newlines to single space
    text = re.sub(r'\s+', ' ', text)

    return text.strip()

if __name__ == "__main__":
    test_cases = [
        "   Hello   @AmazonHelp!   Where is my package???   ",
        "Account hacked!!\n\nSomeone drained my funds \x00\x08",
        None,
        "Ordered on 22nd Sept... still not delivered 📦😡"
    ]
    for tc in test_cases:
        print(f"Original: {repr(tc)}")
        print(f"Cleaned:  {repr(clean_customer_message(tc))}\n")
