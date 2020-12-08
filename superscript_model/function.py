import numpy as np
import matplotlib.pyplot as plt

from interface import Interface, implements


class FunctionFactory:

    @staticmethod
    def get(function_name):
        if function_name == 'TimelineFlexibility':
            return TimelineFlexibility()
        elif function_name == 'NoFlexibility':
            return NoFlexibility()


class FunctionInterface(Interface):

    def get_values(self, x: np.ndarray) -> float:
        pass

    def plot_function(self, xrange, title=None):
        pass

    def print_function(self) -> str:
        pass


class TimelineFlexibility(implements(FunctionInterface)):

    def __init__(self, parameters=(50, -0.8)):
        self.a = parameters[0]
        self.b = parameters[1]

    @staticmethod
    def normalise(y):
        denominator = sum(y)
        return y / denominator

    def get_values(self, x: np.ndarray) -> float:
        return self.normalise(
            self.a * (np.exp(self.b * np.array(x)))
        )

    def plot_function(self, xrange, title=None):

        plt.plot(xrange, self.get_values(xrange))
        plt.title(title)
        plt.show()

    def print_function(self):
        return "TimelineFlexibility = %.2f * (e^(%.2f * X))" % (self.a, self.b)


class NoFlexibility(implements(FunctionInterface)):

    def __init__(self, parameters=(0,)):
        self.a = int(parameters[0])

    def get_values(self, x: np.ndarray) -> float:
        values = np.zeros(len(x))
        values[self.a] = 1.0
        return values

    def plot_function(self, xrange, title=None):

        plt.plot(xrange, self.get_values(xrange))
        plt.title(title)
        plt.show()

    def print_function(self):
        return "NoFlexibility : all probability assigned to %d element" % (self.a,)
