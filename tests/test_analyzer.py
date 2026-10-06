import unittest
from unittest.mock import patch

from src.inference.analyzer import analyze_text


class TestAnalyzer(unittest.TestCase):

    @patch(
        "src.inference.analyzer.predict_sarcasm"
    )
    @patch(
        "src.inference.analyzer.predict_text"
    )
    def test_final_fusion(
        self,
        mock_sentiment,
        mock_sarcasm,
    ):
        mock_sentiment.return_value = {
            "label": "positive",
            "confidence": 0.9,
            "probabilities": {
                "negative": 0.02,
                "neutral": 0.08,
                "positive": 0.90,
            },
            "model": "test",
            "model_status": "test",
        }

        mock_sarcasm.return_value = {
            "label": "not_sarcastic",
            "is_sarcastic": False,
            "sarcasm_probability": 0.02,
            "threshold": 0.36,
            "temperature": 1.1819,
        }

        result = analyze_text("Great 😍")

        self.assertEqual(
            result["final"]["label"],
            "positive",
        )

        self.assertIn("sarcasm", result)
        self.assertIn("emoji", result)

        self.assertEqual(
            result["pipeline_status"],
            "final_fusion_active",
        )

        self.assertEqual(
            result["final"]["model_status"],
            "fusion_v2_verified",
        )


if __name__ == "__main__":
    unittest.main()
