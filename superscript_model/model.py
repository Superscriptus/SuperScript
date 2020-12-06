from mesa import Model
from mesa.time import RandomActivation

from .worker import Worker
from .project import Project, ProjectInventory


# TODO:
# - move parameters to config.py
# - write project_creator class
# - rename private data members _XX
# - does project inventory need to be an order list?
# - refactor so that project are advance by project lead
# - refactor so that projects use inventory.add() and inventory.delete()
# - change remote name
# - add requirements.txt and try installing on different system

class SuperScriptModel(Model):

    def __init__(self, worker_count):

        self.worker_count = worker_count
        self.schedule = RandomActivation(self)
        self.inventory = ProjectInventory()

        for i in range(self.worker_count):
            w = Worker(i, self)
            self.schedule.add(w)

    def step(self):
        self.inventory.create_projects(5)
        self.schedule.step()
        self.inventory.advance_projects()

    def run_model(self, step_count: int):
        for i in range(step_count):
            self.step()


if __name__ == '__main__':

    model = SuperScriptModel(10)
    model.run_model(5)