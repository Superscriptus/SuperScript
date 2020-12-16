import unittest
from unittest.mock import patch

from superscript_model.model import SuperScriptModel
from mesa import Model
from mesa.time import BaseScheduler


class TestSuperScriptModel(unittest.TestCase):

    def test_init(self):

        model = SuperScriptModel(worker_count=100)
        self.assertEqual(model.worker_count, 100)
        self.assertIsInstance(model, Model)
        self.assertIsInstance(model.schedule, BaseScheduler)
        self.assertEqual(model.schedule.get_agent_count(), 100)

    @patch('superscript_model.model.RandomActivation.step')
    def test_step(self, mock_schedule):

        model = SuperScriptModel(worker_count=100)
        model.step()
        mock_schedule.assert_called_once()

    @patch('superscript_model.model.RandomActivation.step')
    def test_run_model(self, mock_step):
        model = SuperScriptModel(worker_count=1000)
        model.run_model(2)
        self.assertEqual(mock_step.call_count, 2)
        self.assertEqual(model.schedule.get_agent_count(), 1000)


if __name__ == '__main__':
    unittest.main()
