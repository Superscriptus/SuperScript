import unittest
from unittest.mock import patch
import numpy as np

from .test_worker import implements_interface
from superscript_model.function import (FunctionInterface,
                                        TimelineFlexibility)


class TestFunctionInterface(unittest.TestCase):

    def test_interface(self):
        self.assertRaises(TypeError, lambda: FunctionInterface())


class TestTimelineFlexibility(unittest.TestCase):

    def test_init(self):

        func = TimelineFlexibility(parameters=(50, -0.8))
        self.assertEqual(func.a, 50)
        self.assertEqual(func.b, -0.8)
        self.assertTrue(implements_interface(func, FunctionInterface))

    def test_get_value(self):

        func = TimelineFlexibility(parameters=(50, -0.8))
        self.assertEqual(np.round(func.get_value(1.0), 2), 22.47)
        self.assertTrue((np.round(func.get_value([2.1, -0.5]), 2)
                         == np.array([9.32, 74.59])).all())

    @patch('matplotlib.pyplot.show')
    def test_plot_function(self, mock_show):
        func = TimelineFlexibility(parameters=(50, -0.8))
        func.plot_function([1,2,3])
        self.assertEqual(mock_show.call_count, 1)

    def test_print_function(self):
        func = TimelineFlexibility(parameters=(50, -0.8))
        self.assertEqual(func.print_function(),
                         "TimelineFlexibility = 50.00 * (e^(-0.80 * X))")





