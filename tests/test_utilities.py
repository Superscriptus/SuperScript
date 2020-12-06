import unittest

from superscript_model.utilities import Random


class TestSuperRandom(unittest.TestCase):

    def test_choice(self):

        r = Random()
        self.assertTrue(r.choice([1,2,3]) in [1,2,3])