"""
SuperScript function module
===========

Supplies real-valued functions for use in different parts of the
SuperScript code via the factory method.

Each function is a class the implements a standard interface, and
is returned via the FunctionFactory.get() method.

Note:
    More detail on the function choice and parameterisation can be
    found in model_development/function_definitions.ipynb

Classes:
    FunctionFactory
        Supplies functions via get() method.
    FunctionInterface
        Standard interface that all function class must implement.
    TimelineFlexibility

    NoFlexibility

    LinearFunction

    SaturatingFunction
"""
import numpy as np
import matplotlib.pyplot as plt

from interface import Interface, implements
from .config import (SUCCESS_PROBABILITY_OVR_GRADIENT,
                     SUCCESS_PROBABILITY_SKILL_BALANCE_GRADIENT,
                     SUCCESS_PROBABILITY_CREATIVITY_MATCH_RATE,
                     SUCCESS_PROBABILITY_CREATIVITY_MATCH_INTERCEPT,
                     SUCCESS_PROBABILITY_RISK_GRADIENT,
                     SUCCESS_PROBABILITY_RISK_INTERCEPT,
                     SUCCESS_PROBABILITY_CHEMISTRY_GRADIENT,
                     SKILL_UPDATE_BY_RISK_GRADIENT,
                     SKILL_UPDATE_BY_RISK_INTERCEPT)


class FunctionFactory:
    """FunctionFactory class.
    """

    @staticmethod
    def get(function_name):
        """Returns a function instance.

        Args:
            function_name: str
                Indicates which function to get.
        """
        if function_name == 'TimelineFlexibility':
            return TimelineFlexibility()
        elif function_name == 'NoFlexibility':
            return NoFlexibility()
        elif function_name == 'SuccessProbabilityOVR':
            return LinearFunction()
        elif function_name == 'SuccessProbabilitySkillBalance':
            return LinearFunction(
                name='SuccessProbabilitySkillBalance',
                gradient=SUCCESS_PROBABILITY_SKILL_BALANCE_GRADIENT
            )
        elif function_name == 'SuccessProbabilityCreativityMatch':
            return SaturatingFunction(
                name='SuccessProbabilityCreativityMatch',
                intercept=SUCCESS_PROBABILITY_CREATIVITY_MATCH_INTERCEPT,
                rate=SUCCESS_PROBABILITY_CREATIVITY_MATCH_RATE
            )
        elif function_name == 'SuccessProbabilityRisk':
            return LinearFunction(
                name='SuccessProbabilityRisk',
                gradient=SUCCESS_PROBABILITY_RISK_GRADIENT,
                intercept=SUCCESS_PROBABILITY_RISK_INTERCEPT
            )
        elif function_name == 'SuccessProbabilityChemistry':
            return LinearFunction(
                name='SuccessProbabilityChemistry',
                gradient=SUCCESS_PROBABILITY_CHEMISTRY_GRADIENT
            )
        elif function_name == 'SkillUpdateByRisk':
            return LinearFunction(
                name='SkillUpdateByRisk',
                gradient=SKILL_UPDATE_BY_RISK_GRADIENT,
                intercept=SKILL_UPDATE_BY_RISK_INTERCEPT
            )
        elif function_name == 'IdentityFunction':
            return LinearFunction(
                name='IdentityFunction',
                gradient=0.0,
                intercept=1.0
            )


class FunctionInterface(Interface):
    """Function Interface.

    All function classes must implement this interface.

    TODO:
        Refactor to use Pythonic ABC pattern, OR inject plotter and
        printer objects.
    """
    def get_values(self, x: np.ndarray) -> np.ndarray:
        pass

    def plot_function(self, xrange, title=None):
        pass

    def print_function(self) -> str:
        pass


class TimelineFlexibility(implements(FunctionInterface)):
    """Function used to determine what level of timeline flexibility
     is permissible for a given project.

    The get_values() method returns an array/list of probabilities y,
    where each yi=f(xi) corresponds to the probability of that start
    time offest xi being used.

    This is a decaying exponential function, whereby ~50% of
    projects start 'now' (i.e. offset=0), ~25% have the flexibility
    to start one timestep ahead (i.e. offset=1) etc..

    Note:
        If the standard offset values of x=[0,1,2,3,4], then this
        function returns (using default a,b values):
        [0.5609451  0.25204888 0.11325286 0.05088779 0.02286536]

    Args:
        a: float
            y-intercept
        b: float
            decay rate
     """

    def __init__(self, parameters=(50, -0.8)):
        self.a = parameters[0]
        self.b = parameters[1]

    @staticmethod
    def normalise(y):
        """Normalises the y values so that they sum to one.

        Args:
            y: numpy.ndarray or list
                Values to normalise

        Returns:
            numpy.ndarray/list: normalised y values
        """
        denominator = sum(y)
        return y / denominator

    def get_values(self, x: np.ndarray) -> np.ndarray:
        """Returns an array of y values for this x.

        Each value yi=f(xi) corresponds to the probability of the
        project starting at offset=xi

        Args:
            x: numpy.ndarray
                Array of possible start time offsets.
                Normally [0,1,2,3,4]

        Returns:
            np.ndarray: array of probabilities
        """

        return self.normalise(
            self.a * (np.exp(self.b * np.asarray(x)))
        )

    def plot_function(self, xrange, title=None):
        """Plot the function.

        Args:
            xrange: numpy.ndarray
                x range to plot over
            title: str (optional)
                plot title
        """

        plt.plot(xrange, self.get_values(xrange))
        plt.title(title)
        plt.show()

    def print_function(self):
        """Returns string of function definition."""
        return ("TimelineFlexibility = %.2f * (e^(%.2f * X))"
                % (self.a, self.b))


class NoFlexibility(implements(FunctionInterface)):
    """Function, similar to TimelineFlexibility, but used to specify
    that the project must start at a specific offset (by default 0).

    Args:
        a: int
            Indicates offset to start at.
    """

    def __init__(self, parameters=(0,)):
        self.a = int(parameters[0])

    def get_values(self, x: np.ndarray) -> np.ndarray:
        """Returns array of y values corresponding to probabilities
        for each start time offset.

        Args:
            x: numpy.ndarray
                Array of possible start time offsets.
                Normally [0,1,2,3,4]

        Returns:
            np.ndarray: array of probabilities
            All entries zero except for one.
        """

        values = np.zeros(len(x))
        values[self.a] = 1.0
        return values

    def plot_function(self, xrange, title=None):
        """Plot the function.

        Args:
            xrange: numpy.ndarray
                x range to plot over
            title: str (optional)
                plot title
        """

        plt.plot(xrange, self.get_values(xrange))
        plt.title(title)
        plt.show()

    def print_function(self):
        """Returns string of function definition."""
        return ("NoFlexibility : all probability assigned to %d element"
                % (self.a,))


class LinearFunction(implements(FunctionInterface)):
    """Generic linear function.

    Used in various places where a parametersied linear function is
    required.

    Args:
        name: str
            Name of this function
        gradient: float
            Line gradient
        intercept: float
            y intercept
    """

    def __init__(self, name="SuccessProbabilityOVR",
                 gradient=SUCCESS_PROBABILITY_OVR_GRADIENT,
                 intercept=0):

        self.name = name
        self.gradient = gradient
        self.intercept = intercept

    def get_values(self, x: np.ndarray) -> np.ndarray:
        """Returns function values.

        Args:
            x: numpy.ndarray
                x values
        Returns:
            numpy.ndarray: y values
        """

        return self.intercept + self.gradient * np.asarray(x)

    def plot_function(self, xrange, title=None):
        """Plot the function.

        Args:
            xrange: numpy.ndarray
                x range to plot over
            title: str (optional)
                plot title
        """

        plt.plot(xrange, self.get_values(xrange))
        plt.title(title)
        plt.show()

    def print_function(self):
        """Returns string of function definition."""
        return "%s = %.2f + %.2f * X" % (self.name,
                                         self.intercept,
                                         self.gradient)


class SaturatingFunction(implements(FunctionInterface)):
    """Generic saturating function.

    Can control intercept and saturation value.

    Note:
        function asymptotically saturates to y = rate + intercept

    Args:
        name: str
            Name of this function
        rate: float
            Rate of saturation
        intercept: float
            y intercept
    """
    def __init__(self, name="SuccessProbabilityCreativityMatch",
                 rate=SUCCESS_PROBABILITY_CREATIVITY_MATCH_RATE,
                 intercept=SUCCESS_PROBABILITY_CREATIVITY_MATCH_INTERCEPT):

        self.name = name
        self.rate = rate
        self.intercept = intercept

    def get_values(self, x: np.ndarray) -> np.ndarray:
        """Returns function values.

        Args:
            x: numpy.ndarray
                x values
        Returns:
            numpy.ndarray: y values
        """
        return self.intercept - self.rate * (1 - np.exp(-np.asarray(x)))

    def plot_function(self, xrange, title=None):
        """Plot the function.

        Args:
            xrange: numpy.ndarray
                x range to plot over
            title: str (optional)
                plot title
        """

        plt.plot(xrange, self.get_values(xrange))
        plt.title(title)
        plt.show()

    def print_function(self):
        """Returns string of function definition."""
        return "%s = %.2f - %.2f * (1 - exp(X))" % (
            self.name,
            self.intercept,
            self.rate
        )
