import numpy as np
import matplotlib.pyplot as plt

from interface import Interface, implements


class FunctionFactory:

    @staticmethod
    def get(function_name):
        if function_name == 'TimelineFlexibility':
            return TimelineFlexibility()


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
