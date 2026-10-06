import unittest

from api.main import (
    health,
    root,
)


class TestAPI(unittest.TestCase):

    def test_root(self):
        result = root()

        self.assertEqual(
            result["status"],
            "ok",
        )

    def test_health(self):
        result = health()

        self.assertEqual(
            result["status"],
            "ok",
        )


if __name__ == "__main__":
    unittest.main()
