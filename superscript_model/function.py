import numpy as np
import matplotlib.pyplot as plt

from interface import Interface, implements


class FunctionInterface(Interface):

    def get_value(self, x: float) -> float:
        pass

    def plot_function(self, xrange, title=None):
        pass

    def print_function(self) -> str:
        pass


class TimelineFlexibility(implements(FunctionInterface)):

    def __init__(self, parameters):
        self.a = parameters[0]
        self.b = parameters[1]

    def get_value(self, x: float) -> float:
        return self.a * (np.exp(self.b * np.array(x)))

    def plot_function(self, xrange, title=None):

        plt.plot(xrange, self.get_value(xrange))
        plt.title(title)
        plt.show()

    def print_function(self):
        return "TimelineFlexibility = %.2f * (e^(%.2f * X))" % (self.a, self.b)
