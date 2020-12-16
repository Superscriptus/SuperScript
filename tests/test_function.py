import unittest
from unittest.mock import patch
import numpy as np

from .test_worker import implements_interface
from superscript_model.function import (FunctionInterface,
                                        TimelineFlexibility,
                                        NoFlexibility,
                                        LinearFunction,
                                        SaturatingFunction)
from superscript_model.config import (SUCCESS_PROBABILITY_OVR_GRADIENT,
                                      SUCCESS_PROBABILITY_SKILL_BALANCE_GRADIENT,
                                      SUCCESS_PROBABILITY_CREATIVITY_MATCH_RATE,
                                      SUCCESS_PROBABILITY_CREATIVITY_MATCH_INTERCEPT)


class TestFunctionInterface(unittest.TestCase):

    def test_interface(self):
        self.assertRaises(TypeError, lambda: FunctionInterface())


class TestTimelineFlexibility(unittest.TestCase):

    def test_init(self):

        func = TimelineFlexibility(parameters=(50, -0.8))
        self.assertEqual(func.a, 50)
        self.assertEqual(func.b, -0.8)
        self.assertTrue(implements_interface(func, FunctionInterface))

    def test_normalise(self):
        func = TimelineFlexibility(parameters=(50, -0.8))
        self.assertTrue((func.normalise(np.array([1,2,3]))
                         == np.array([1,2,3])/6).all())

    def test_get_value(self):

        func = TimelineFlexibility(parameters=(50, -0.8))
        self.assertEqual(np.round(func.get_values([1.0]), 2), np.array(1.0))
        self.assertTrue((np.round(func.get_values([2.1, -0.5]), 2)
                         == np.array([0.11, 0.89])).all())

    @patch('matplotlib.pyplot.show')
    def test_plot_function(self, mock_show):
        func = TimelineFlexibility(parameters=(50, -0.8))
        func.plot_function([1,2,3])
        self.assertEqual(mock_show.call_count, 1)

    def test_print_function(self):
        func = TimelineFlexibility(parameters=(50, -0.8))
        self.assertEqual(func.print_function(),
                         "TimelineFlexibility = 50.00 * (e^(-0.80 * X))")


class TestNoFlexibility(unittest.TestCase):

    def test_init(self):

        func = NoFlexibility(parameters=(0,))
        self.assertEqual(func.a, 0)
        self.assertTrue(implements_interface(func, FunctionInterface))

    def test_get_value(self):

        func = NoFlexibility(parameters=(0,))
        self.assertEqual(np.round(func.get_values([1.0]), 2), np.array(1.0))
        self.assertTrue((np.round(func.get_values([2.1, -0.5]), 2)
                         == np.array([1.0, 0.0])).all())

    @patch('matplotlib.pyplot.show')
    def test_plot_function(self, mock_show):
        func = NoFlexibility()
        func.plot_function([1,2,3])
        self.assertEqual(mock_show.call_count, 1)

    def test_print_function(self):
        func = NoFlexibility(parameters=(0,))
        self.assertEqual(func.print_function(),
                         "NoFlexibility : all probability assigned to 0 element")


class TestLinearFunction(unittest.TestCase):

    def test_init(self):

        func = LinearFunction()
        self.assertEqual(func.gradient, SUCCESS_PROBABILITY_OVR_GRADIENT)

    def test_get_values(self):

        func = LinearFunction()
        x = np.asarray([0.0, 50.0, 100.0])
        self.assertEqual(list(func.get_values(x)),
                         [0.0, 37.5, 75.0])

    @patch('matplotlib.pyplot.show')
    def test_plot_function(self, mock_show):
        func = LinearFunction()
        func.plot_function([1, 2, 3])
        self.assertEqual(mock_show.call_count, 1)

    def test_print_function(self):
        func = LinearFunction()
        self.assertEqual(func.print_function(),
                         "SuccessProbabilityOVR = 0.00 + 0.75 * X")

    @patch('matplotlib.pyplot.show')
    def test_success_probability_skill_balance(self, mock_show):

        func = LinearFunction(name="SuccessProbabilitySkillBalance",
                              gradient=SUCCESS_PROBABILITY_SKILL_BALANCE_GRADIENT,
                              intercept=0)

        x = np.asarray([0.0, 8.0, 16.0])
        self.assertEqual(list(func.get_values(x)),
                     [0.0, -20.0, -40.0])

        func.plot_function([1, 2, 3])
        self.assertEqual(mock_show.call_count, 1)

        self.assertEqual(func.print_function(),
                         "SuccessProbabilitySkillBalance = 0.00 + -2.50 * X")


class TestSaturatingFunction(unittest.TestCase):

    @patch('matplotlib.pyplot.show')
    def test_success_probability_creativity_match(self, mock_show):
        func = SaturatingFunction(
            name="SuccessProbabilityCreativityMatch",
            rate=SUCCESS_PROBABILITY_CREATIVITY_MATCH_RATE,
            intercept=SUCCESS_PROBABILITY_CREATIVITY_MATCH_INTERCEPT
        )
        x = [0.0, 1.0, 16.0]
        y = [10.0, -15.28, -30.00]
        for xi, yi in zip(x, y):
            self.assertEqual(round(func.get_values(xi), 2), yi)

        func.plot_function([1, 2, 3])
        self.assertEqual(mock_show.call_count, 1)

        self.assertEqual(func.print_function(),
                         "SuccessProbabilityCreativityMatch = 10.00 - 40.00 * (1 - exp(X))")
