"""
SuperScript utilities module.
===========

Helper method for use throughout the code.

Currently just a Random number generation class.

"""

import random
import numpy as np


def normalise_success(data, _min=-1.075, _max=0.95):
    
    mini = _min if _min is not None else min(data)
    maxi = _max if _max is not None else max(data)
    
    return (data - mini) / (maxi - mini)


class Random:

    @staticmethod
    def choice(iterable):
        return random.choice(iterable)

    @staticmethod
    def choices(iterable, k):
        return random.choices(iterable, k=k)

    @staticmethod
    def weighted_choice(iterable, k, replace=False, p=None):

        if k > len(iterable) or len(iterable) == 0:
            return iterable

        if p is not None:
            p = np.array(p)
            p = p/sum(p)
            p[-1] = 1 - sum(p[:-1])

        return np.random.choice(
            iterable, size=k,
            replace=replace, p=p
        )

    @staticmethod
    def randint(a, b):
        return random.randint(a, b)

    @staticmethod
    def uniform(a=0.0, b=1.0):
        return random.uniform(a, b)

    @staticmethod
    def shuffle(x):
        random.shuffle(x)

    @staticmethod
    def normal(mean, std):
        return random.normalvariate(mean, std)
