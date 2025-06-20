# src/utils/mfa_text_normalizer.py
import re

def normalize_text_for_mfa(text: str) -> str:
    """
    Normalizes a text string to be compatible with the Montreal Forced Aligner.

    This involves:
    1. Removing parenthetical content (as a safeguard).
    2. Converting the text to uppercase.
    3. Removing all punctuation except apostrophes.
    4. Collapsing multiple whitespace characters into a single space.

    Args:
        text: The input string from the Scribe transcript.

    Returns:
        A cleaned, MFA-compatible version of the text.
    """
    # --- MODIFICATION: Remove any parenthetical content ---
    text = re.sub(r'\([^)]*\)', '', text)

    # Convert to uppercase
    text = text.upper()
    
    # Remove punctuation except for apostrophes
    text = re.sub(r"[^A-Z'\s]", '', text)
    
    # Replace multiple whitespace characters with a single space
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text
