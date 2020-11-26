import unittest
from unittest.mock import patch

from superscript_model.model import SuperScriptModel


class TestSuperScriptModel(unittest.TestCase):

    @patch('superscript_model.model.SuperScriptModel.step')
    def test_init(self, mock_step):

        model = SuperScriptModel()
        self.assertTrue(mock_step.called)



if __name__ == '__main__':
    unittest.main()
