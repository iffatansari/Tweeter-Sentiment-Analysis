"""Twitter-specific preprocessing utilities. Kept intentionally conservative."""


def preprocess_tweet(text: str) -> str:
    """Return a lightly normalized tweet without destroying sentiment cues."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    return " ".join(text.strip().split())
