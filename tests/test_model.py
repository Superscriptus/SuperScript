import unittest
from unittest.mock import patch

from superscript_model.model import SuperScriptModel
from mesa import Model
from mesa.time import BaseScheduler


class TestSuperScriptModel(unittest.TestCase):

    def test_init(self):

        model = SuperScriptModel(worker_count=10)
        self.assertEqual(model.worker_count, 10)
        self.assertIsInstance(model, Model)
        self.assertIsInstance(model.schedule, BaseScheduler)

    @patch('superscript_model.model.RandomActivation.step')
    def test_step(self, mock_schedule):

        model = SuperScriptModel(worker_count=10)
        model.step()
        mock_schedule.assert_called_once()


if __name__ == '__main__':
    unittest.main()
