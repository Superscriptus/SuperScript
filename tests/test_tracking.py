import unittest

from superscript_model.tracking import safe_mean


class TestTracking(unittest.TestCase):

    def test_safe_mean(self):
        self.assertEqual(
            safe_mean([1, 2, 3]),
            2
        )
        self.assertEqual(
            safe_mean([]),
            0
        )
