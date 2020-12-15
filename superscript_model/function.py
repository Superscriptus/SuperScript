import numpy as np
import matplotlib.pyplot as plt

from interface import Interface, implements
from .config import (SUCCESS_PROBABILITY_OVR_GRADIENT,
                     SUCCESS_PROBABILITY_SKILL_BALANCE_GRADIENT,
                     SUCCESS_PROBABILITY_SKILL_BALANCE_RATE,
                     SUCCESS_PROBABILITY_SKILL_BALANCE_INTERCEPT)


class FunctionFactory:

    @staticmethod
    def get(function_name):
        if function_name == 'TimelineFlexibility':
            return TimelineFlexibility()
        elif function_name == 'NoFlexibility':
            return NoFlexibility()
        elif function_name == 'SuccessProbabilityOVR':
            return LinearFunction()
        elif function_name == 'SuccessProbabilitySkillBalance':
            return LinearFunction(
                name='SuccessProbabilityOVR',
                gradient=SUCCESS_PROBABILITY_SKILL_BALANCE_GRADIENT
            )


class FunctionInterface(Interface):

    def get_values(self, x: np.ndarray) -> np.ndarray:
        pass

    def plot_function(self, xrange, title=None):
        pass

    def print_function(self) -> str:
        pass

# Get parameters from config...
class TimelineFlexibility(implements(FunctionInterface)):

    def __init__(self, parameters=(50, -0.8)):
        self.a = parameters[0]
        self.b = parameters[1]

    @staticmethod
    def normalise(y):
        denominator = sum(y)
        return y / denominator

    def get_values(self, x: np.ndarray) -> np.ndarray:
        return self.normalise(
            self.a * (np.exp(self.b * np.asarray(x)))
        )

    def plot_function(self, xrange, title=None):

        plt.plot(xrange, self.get_values(xrange))
        plt.title(title)
        plt.show()

    def print_function(self):
        return ("TimelineFlexibility = %.2f * (e^(%.2f * X))"
                % (self.a, self.b))


class NoFlexibility(implements(FunctionInterface)):

    def __init__(self, parameters=(0,)):
        self.a = int(parameters[0])

    def get_values(self, x: np.ndarray) -> np.ndarray:
        values = np.zeros(len(x))
        values[self.a] = 1.0
        return values

    def plot_function(self, xrange, title=None):

        plt.plot(xrange, self.get_values(xrange))
        plt.title(title)
        plt.show()

    def print_function(self):
        return ("NoFlexibility : all probability assigned to %d element"
                % (self.a,))


class LinearFunction(implements(FunctionInterface)):

    def __init__(self, name="SuccessProbabilityOVR",
                 gradient=SUCCESS_PROBABILITY_OVR_GRADIENT,
                 intercept=0):

        self.name = name
        self.gradient = gradient
        self.intercept = intercept

    def get_values(self, x: np.ndarray) -> np.ndarray:
        return self.intercept + self.gradient * np.asarray(x)

    def plot_function(self, xrange, title=None):
        plt.plot(xrange, self.get_values(xrange))
        plt.title(title)
        plt.show()

    def print_function(self):
        return "%s = %.2f + %.2f * X" % (self.name,
                                         self.intercept,
                                         self.gradient)


class SaturatingFunction(implements(FunctionInterface)):

    def __init__(self, name="SuccessProbabilityCreativityMatch",
                 rate=SUCCESS_PROBABILITY_SKILL_BALANCE_RATE,
                 intercept=SUCCESS_PROBABILITY_SKILL_BALANCE_INTERCEPT):

        self.name = name
        self.rate = rate
        self.intercept = intercept

    def get_values(self, x: np.ndarray) -> np.ndarray:
        return self.intercept - self.rate * (1 - np.exp(-np.asarray(x)))

    def plot_function(self, xrange, title=None):
        plt.plot(xrange, self.get_values(xrange))
        plt.title(title)
        plt.show()

    def print_function(self):
        return "%s = %.2f - %.2f * (1 - exp(X))" % (self.name,
                                                   self.intercept,
                                                   self.rate)
