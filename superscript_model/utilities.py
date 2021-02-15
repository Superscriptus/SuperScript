import random
import numpy as np

class Random:

    @staticmethod
    def choice(iterable):
        return random.choice(iterable)

    @staticmethod
    def choices(iterable, k):
        return random.choices(iterable, k=k)

    @staticmethod
    def weighted_choice(iterable, k, replace=False, p=None):
        return random.np.choice(iterable, size=k,
                                replace=replace, p=p)

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