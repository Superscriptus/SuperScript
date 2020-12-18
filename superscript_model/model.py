from mesa import Model
from mesa.time import RandomActivation

from .worker import Worker
from .project import ProjectInventory
from .organisation import (TeamAllocator,
                           Department,
                           Trainer)
from .config import (PROJECT_LENGTH,
                     NEW_PROJECTS_PER_TIMESTEP,
                     WORKER_COUNT,
                     DEPARTMENT_COUNT)

# TODO:
# 10 minutes left over

# write test_determine_success, test_replace_worker
# what to do if cannot assign team to project e.g. Cannot select 4 workers from bid_pool of size 0...??
# Add Social network
# Implement go_settle
# (- * add contribution class for Dept.)
# - **add budget constraint functionality
# - add chemistry booster
# - implement retire/replace worker
# - ensure worker is deleted from dept when worker 'dies'

# unit test for probability success - test not over 1

# - refactor so that Team creation does not automatically assign worker contributions -
#       need to be able to create hypothetical teams to compare success prob
#       solution: only call assign_contributions_to_members once team is finalised

# - rename skill balance - degree of mismatch..
# - inject SuccessCalculator (not create)
# - delete old code from inventory.get_starttime_offset once confirmed new version works
# - coverage run -m unittest discover && coverage report
# - add function parameters to config
# - change FunctionInterface to abstract base class (plot and print never change)
# - rename private data members _XX

# - confirm that skill balance calculations are correct when worker is unable to supply skill due to dept constraint
# - inject strategy into TeamAllocator
# - manually calculate active_projects (80 versus 85)
# - remove historical work contributions from worker.contributes? (to free up memory) + remove department history?
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
# - number of workers needs to be divisible by number of departments
# to run single test: python -m unittest tests.test_organisation.TestRandomStrategy.test_invite_bids


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
        self.trainer = Trainer(self)

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
        self.trainer.update_skill_quartiles()
        self.inventory.create_projects(NEW_PROJECTS_PER_TIMESTEP,
                                       self.time, PROJECT_LENGTH)
        self.schedule.step()
        self.time += 1

    def run_model(self, step_count: int):
        for i in range(step_count):
            self.step()
