import unittest
from unittest.mock import patch

from superscript_model.model import SuperScriptModel
from mesa import Model
from mesa.time import BaseScheduler


class TestSuperScriptModel(unittest.TestCase):

    @patch('superscript_model.model.SuperScriptModel.step')
    def test_init(self, mock_step):

        model = SuperScriptModel(worker_count=10)
        self.assertEqual(model.worker_count, 10)
        self.assertIsInstance(model, Model)
        self.assertIsInstance(model.schedule, BaseScheduler)
        self.assertTrue(mock_step.called)

    @patch('superscript_model.model.SuperScriptModel')
    def test_step(self, mock_model):

        mock_model.step()
        mock_model.schedule.step.assert_called_once()




if __name__ == '__main__':
    unittest.main()
