from __future__ import annotations

from typing import Iterable


EMOJI_RANGES = (
    (0x1F300, 0x1F5FF),
    (0x1F600, 0x1F64F),
    (0x1F680, 0x1F6FF),
    (0x1F700, 0x1F77F),
    (0x1F780, 0x1F7FF),
    (0x1F800, 0x1F8FF),
    (0x1F900, 0x1F9FF),
    (0x1FA00, 0x1FAFF),
    (0x2600, 0x26FF),
    (0x2700, 0x27BF),
)

VARIATION_SELECTORS = {0xFE0E, 0xFE0F}
SKIN_TONES = range(0x1F3FB, 0x1F400)
ZWJ = "\u200d"
KEYCAP = "\u20E3"


POSITIVE_EMOJIS = {
    "😀", "😃", "😄", "😁", "😆", "😅",
    "😂", "🤣", "😊", "😇", "🙂", "🙃",
    "😉", "😌", "😍", "🥰", "😘", "😎",
    "🤩", "🥳", "👍", "👏", "🙌", "💯",
    "🔥", "🎉", "✨", "💖", "💗", "💓",
    "💕", "💞", "💝", "❤️", "❤",
}

NEGATIVE_EMOJIS = {
    "😞", "😔", "😟", "😕", "🙁", "☹️",
    "😣", "😖", "😫", "😩", "🥺", "😢",
    "😭", "😡", "😠", "🤬", "😒", "🙄",
    "😤", "😑", "😐", "🤦", "💔", "👎",
    "🤢", "🤮", "😱", "😨", "😰",
}

LAUGHTER_EMOJIS = {
    "😂", "🤣", "😆", "😅",
}

HEART_EMOJIS = {
    "❤️", "❤", "💖", "💗", "💓",
    "💕", "💞", "💝",
}

SAD_EMOJIS = {
    "😞", "😔", "😟", "😕", "🙁",
    "☹️", "😣", "😖", "😫", "😩",
    "🥺", "😢", "😭", "💔",
}

ANGRY_EMOJIS = {
    "😡", "😠", "🤬", "😤",
}

SURPRISE_EMOJIS = {
    "😱", "😨", "😰", "😲",
    "😮", "😯", "😳",
}


def _is_emoji_base(char: str) -> bool:
    codepoint = ord(char)

    return any(
        start <= codepoint <= end
        for start, end in EMOJI_RANGES
    )


def _is_skin_tone(char: str) -> bool:
    return ord(char) in SKIN_TONES


def extract_emoji_tokens(text: str) -> list[str]:
    tokens: list[str] = []
    i = 0

    while i < len(text):
        char = text[i]

        if not _is_emoji_base(char):
            i += 1
            continue

        token = char
        i += 1

        while i < len(text):
            codepoint = ord(text[i])

            if (
                codepoint in VARIATION_SELECTORS
                or _is_skin_tone(text[i])
            ):
                token += text[i]
                i += 1
            else:
                break

        while (
            i < len(text)
            and text[i] == ZWJ
            and i + 1 < len(text)
            and _is_emoji_base(text[i + 1])
        ):
            token += ZWJ + text[i + 1]
            i += 2

            while i < len(text):
                codepoint = ord(text[i])

                if (
                    codepoint in VARIATION_SELECTORS
                    or _is_skin_tone(text[i])
                ):
                    token += text[i]
                    i += 1
                else:
                    break

        tokens.append(token)

    return tokens


EMOJI_FEATURE_NAMES = [
    "emoji_count",
    "emoji_unique_count",
    "positive_emoji_count",
    "negative_emoji_count",
    "neutral_emoji_count",
    "laughter_emoji_count",
    "heart_emoji_count",
    "sad_emoji_count",
    "angry_emoji_count",
    "surprise_emoji_count",
    "emoji_polarity_score",
    "emoji_density",
]


def _count_matches(
    tokens: Iterable[str],
    vocabulary: set[str],
) -> int:
    count = 0

    for token in tokens:
        if token in vocabulary:
            count += 1
            continue

        base = next(
            (
                char
                for char in token
                if _is_emoji_base(char)
            ),
            token,
        )

        if base in vocabulary:
            count += 1

    return count


def extract_emoji_features(
    text: str,
) -> dict[str, float]:
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    tokens = extract_emoji_tokens(text)

    emoji_count = len(tokens)
    unique_count = len(set(tokens))

    positive_count = _count_matches(
        tokens,
        POSITIVE_EMOJIS,
    )

    negative_count = _count_matches(
        tokens,
        NEGATIVE_EMOJIS,
    )

    laughter_count = _count_matches(
        tokens,
        LAUGHTER_EMOJIS,
    )

    heart_count = _count_matches(
        tokens,
        HEART_EMOJIS,
    )

    sad_count = _count_matches(
        tokens,
        SAD_EMOJIS,
    )

    angry_count = _count_matches(
        tokens,
        ANGRY_EMOJIS,
    )

    surprise_count = _count_matches(
        tokens,
        SURPRISE_EMOJIS,
    )

    neutral_count = max(
        emoji_count
        - positive_count
        - negative_count,
        0,
    )

    polarity = (
        (
            positive_count
            - negative_count
        )
        / emoji_count
        if emoji_count
        else 0.0
    )

    non_space_chars = sum(
        not char.isspace()
        for char in text
    )

    density = (
        emoji_count / non_space_chars
        if non_space_chars
        else 0.0
    )

    return {
        "emoji_count": float(emoji_count),
        "emoji_unique_count": float(unique_count),
        "positive_emoji_count": float(
            positive_count
        ),
        "negative_emoji_count": float(
            negative_count
        ),
        "neutral_emoji_count": float(
            neutral_count
        ),
        "laughter_emoji_count": float(
            laughter_count
        ),
        "heart_emoji_count": float(
            heart_count
        ),
        "sad_emoji_count": float(
            sad_count
        ),
        "angry_emoji_count": float(
            angry_count
        ),
        "surprise_emoji_count": float(
            surprise_count
        ),
        "emoji_polarity_score": float(
            polarity
        ),
        "emoji_density": float(density),
    }


def emoji_feature_vector(
    text: str,
) -> list[float]:
    features = extract_emoji_features(text)

    return [
        features[name]
        for name in EMOJI_FEATURE_NAMES
    ]
