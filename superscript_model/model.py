from mesa import Model
from mesa.time import RandomActivation


class SuperScriptModel(Model):

    def __init__(self, worker_count):

        self.worker_count = worker_count
        self.schedule = RandomActivation(self)

    def step(self):
        self.schedule.step()
