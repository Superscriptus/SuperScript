from mesa import Model
from mesa.time import RandomActivation

from .worker import Worker
from .project import ProjectInventory


# TODO:
# - add project late start functionality.
# - move parameters to config.py
# - write project_creator class
# - rename private data members _XX
# - does project inventory need to be an ordered dict?
# - refactor so that project are advance by project lead
# - change remote name
# - add requirements.txt and try installing on different system
# - remove heavy dependencies for deployment (e.g. ipykernel + jupyterlab)
# - remove accept method from worker strategy?
# - refactor so that strategies are injected to clients (not constructed internally)
# - are we safe passing around references to project and worker (eg. for project_lead?)
# #    i.e. is memory correctly cleaned up on project/worker deletion?
# - refactor to pass project length in to create_projects (from config)

class SuperScriptModel(Model):

    def __init__(self, worker_count):

        self.worker_count = worker_count
        self.schedule = RandomActivation(self)
        self.inventory = ProjectInventory()

        for i in range(self.worker_count):
            w = Worker(i, self)
            self.schedule.add(w)

    def step(self):
        self.inventory.create_projects(20)
        self.schedule.step()

    def run_model(self, step_count: int):
        for i in range(step_count):
            self.step()
