from mesa import Model
from mesa.time import RandomActivation

from .worker import Worker
from .project import ProjectInventory
from .organisation import TeamAllocator, Department
from .config import (PROJECT_LENGTH,
                     NEW_PROJECTS_PER_TIMESTEP,
                     WORKER_COUNT,
                     DEPARTMENT_COUNT)

# TODO:
# 30 minutes left over
# - implement checking if dept workload is met (for bid, and when assinging worker contributions
# - check that worker not contributing too many units..
# handle negative probability?

# - refactor so that Team creation does not automatically assign worker contributions -
#       need to be able to create hypothetical teams to compare success prob
#       solution: only call assign_contributions_to_members once team is finalised

# - update units tests (and test creativity_match in jupyter)
# - rename skill balance - degree of mismatch..
# - refactor to use Contributions class to log worker and department contributions?
# - add other components to success calculator
# - inject SuccessCalculator (not create)
# - refactor ovr and skill_balance tests to use the smae objects?
# - time the model running in jupyter..

# - delete old code from inventory.get_starttime_offset once confirmed new version works
# - coverage run -m unittest discover && coverage report
# - add training and skill decay functionality
# - add department functionality
# - add budget constraint functionality
# - add function parameters to config
# - change FunctionInterface to abstract base class (plot and print never change)
# - rename private data members _XX

# - ensure worker is deleted from dept when worker 'dies'
# - inject strategy into TeamAllocator
# - manually calculate active_projects (80 versus 85)
# - remove historical work contributions from worker.contributes? (to free up memory)
# - reorder and annotate config.py (and refactor tests to use config variables?)
# - change remote name
# - add requirements.txt and try installing on different system
# - remove heavy dependencies for deployment (e.g. ipykernel + jupyterlab)
# - remove accept method from worker strategy?
# - refactor so that strategies are injected to clients (not constructed internally)
# - are we safe passing around references to project and worker (eg. for project_lead?)
# #    i.e. is memory correctly cleaned up on project/worker deletion?
# - patch TeamAllocator throughout whole class/file in test_project?
# - refactor so that plot_function takes a function plotter object
# - assert that time=start_time when project progress=0


class SuperScriptModel(Model):

    def __init__(self, worker_count=WORKER_COUNT,
                 department_count=DEPARTMENT_COUNT):

        self.worker_count = worker_count
        self.departments = dict()

        self.schedule = RandomActivation(self)
        self.inventory = ProjectInventory(
            TeamAllocator(self),
            timeline_flexibility='TimelineFlexibility'
        )

        for di in range(department_count):
            self.departments[di] = Department(di)

        workers_per_department = department_count / worker_count
        assert workers_per_department * worker_count == department_count

        di = 0
        assigned_to_di = 0
        for i in range(self.worker_count):
            w = Worker(i, self, self.departments[di])
            self.schedule.add(w)

            assigned_to_di +=1
            if assigned_to_di == workers_per_department:
                di += 1
                assigned_to_di = 0

        self.time = 0

    def step(self):
        self.inventory.create_projects(NEW_PROJECTS_PER_TIMESTEP,
                                       self.time, PROJECT_LENGTH)
        self.schedule.step()
        self.time += 1

    def run_model(self, step_count: int):
        for i in range(step_count):
            self.step()
