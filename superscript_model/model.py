from mesa import Model
from mesa.time import RandomActivation

from .worker import Worker
from .project import ProjectInventory
from .organisation import TeamAllocator


# TODO:
# - coverage run -m unittest discover && coverage report
# - move parameters to config.py
# - refactor to pass project length in to create_projects (from config)
# - rename private data members _XX
# - change FunctionInterface to abstract base class (plot and print never change)

# - set worker contribution to project from within team/project on project start..(when progress reaches 0)
# - change active_project_equilibrium.py to compute active projects at same point in timestep (currently mid-timestep)
# - does project inventory need to be an ordered dict?
# - change remote name
# - add requirements.txt and try installing on different system
# - remove heavy dependencies for deployment (e.g. ipykernel + jupyterlab)
# - remove accept method from worker strategy?
# - refactor so that strategies are injected to clients (not constructed internally)
# - are we safe passing around references to project and worker (eg. for project_lead?)
# #    i.e. is memory correctly cleaned up on project/worker deletion?
# - patch TeamAllocator throughout whole class/file in test_project?
# - refactor so that plot_function takes a function plotter object

class SuperScriptModel(Model):

    def __init__(self, worker_count):

        self.worker_count = worker_count
        self.schedule = RandomActivation(self)
        self.inventory = ProjectInventory(
            TeamAllocator(self),
            timeline_flexibility='TimelineFlexibility'
        )

        for i in range(self.worker_count):
            w = Worker(i, self)
            self.schedule.add(w)

    def step(self):
        self.inventory.create_projects(20)
        self.schedule.step()

    def run_model(self, step_count: int):
        for i in range(step_count):
            self.step()
