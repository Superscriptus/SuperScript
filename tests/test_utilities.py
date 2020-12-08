import unittest

from superscript_model.utilities import Random


class TestRandom(unittest.TestCase):

    def test_choice(self):

        r = Random()
        self.assertTrue(r.choice([1,2,3]) in [1,2,3])

    def test_choices(self):

        r = Random()
        values = r.choices([1,2,3,4,5], k=3)
        for v in values:
            self.assertTrue(v in [1,2,3,4,5])

    def test_randint(self):

        r = Random()
        for i in range(10):
            value = r.randint(1,5)
            self.assertIsInstance(value, int)
            self.assertTrue((value >= 1)
                            & (value <= 5))

    def test_uniform(self):

        r = Random()
        for i in range(100):
            value = r.uniform()
            self.assertIsInstance(value, float)
            self.assertTrue((value >= 0.0)
                            & (value <= 1.0))
