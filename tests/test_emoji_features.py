import unittest

from src.features.emoji_features import (
    extract_emoji_features,
    extract_emoji_tokens,
)


class TestEmojiFeatures(unittest.TestCase):

    def test_no_emoji(self):
        result = extract_emoji_features(
            "plain text"
        )

        self.assertEqual(
            result["emoji_count"],
            0.0,
        )

    def test_positive_emoji(self):
        result = extract_emoji_features(
            "I love this 😍🔥"
        )

        self.assertGreater(
            result["emoji_count"],
            0,
        )

        self.assertGreater(
            result["positive_emoji_count"],
            0,
        )

    def test_negative_emoji(self):
        result = extract_emoji_features(
            "Terrible 😡💔"
        )

        self.assertGreater(
            result["negative_emoji_count"],
            0,
        )

    def test_tokens(self):
        tokens = extract_emoji_tokens(
            "Hello 😂❤️👍"
        )

        self.assertGreaterEqual(
            len(tokens),
            3,
        )


if __name__ == "__main__":
    unittest.main()
