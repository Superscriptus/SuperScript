from mesa import Model
from mesa.time import RandomActivation

from .worker import Worker
from .project import Project


# TODO:
# - move parameters to config.py
# - write project_creator class
# - rename private data members _XX

class SuperScriptModel(Model):

    def __init__(self, worker_count):

        self.worker_count = worker_count
        self.schedule = RandomActivation(self)

        for i in range(self.worker_count):
            w = Worker(i, self)
            self.schedule.add(w)

    def step(self):
        new_project = Project(0, 5)
        self.schedule.step()
        new_project.advance()

    def run_model(self, step_count: int):
        for i in range(step_count):
            self.step()
