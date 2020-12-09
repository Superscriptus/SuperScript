import random


class Random:

    @staticmethod
    def choice(iterable):
        return random.choice(iterable)

    @staticmethod
    def choices(iterable, k):
        return random.choices(iterable, k=k)

    @staticmethod
    def randint(a, b):
        return random.randint(a, b)

    @staticmethod
    def uniform(a=0.0, b=1.0):
        return random.uniform(a, b)