from mesa import Model
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
import networkx as nx
import numpy as np

from .worker import Worker
from .project import ProjectInventory
from .network import SocialNetwork
from .organisation import (TeamAllocator,
                           Department,
                           Trainer)
from .config import (PROJECT_LENGTH,
                     NEW_PROJECTS_PER_TIMESTEP,
                     WORKER_COUNT,
                     DEPARTMENT_COUNT,
                     TRAINING_ON,
                     BUDGET_FUNCTIONALITY_FLAG)


# TODO:
# check that budget interpretation is correct
# ! message Michael about the Null teams issue
# **what to do if cannot assign team to project e.g. Cannot select 4 workers from bid_pool of size 0...??
#       -> notify Michael about this (and that actual average is 0.22)

# add requested tracking functions...

# Implement go_settle - peer assessment

# - optimise AllInStrategy.bid - takes ~60% of run time...
# remove pyplot?!
# update all units tests


# For visualisation:
# - allow turn network on/off (display reduced network(only recent edges)? specify fixed node positions?)
# - add controls for main parameters (
# - allow training to be switched on/off
# - allow turn budget constraint on/off
# - add graph displays for main tracking variables
# - edit layout of plots?
# - show only recent edges (but track all?)
# - display nodes in circle with node size and colour determined by...activity, success?
# ( - add description to model for "About")

# - model will only work with a constant number of agents because of Grid (network) constraints.

# - refactor to use .get() for safe dictionary access
# - refactor so that Team creation does not automatically assign worker contributions -
#       need to be able to create hypothetical teams to compare success prob
#       solution: only call assign_contributions_to_members once team is finalised

# - write better unit test for budget (and check usage in TeamAllocator - should be elsewhere?)
#           check why budget is None (catch needed currently in TeamAllocator)

# change use of time below to steps()
# - calculate theoretical maximum/minimum prob for each component with current functions
# - rename skill balance - degree of mismatch..
# - inject SuccessCalculator (not create)
# - delete old code from inventory.get_starttime_offset once confirmed new version works
# - add requirements.txt and try installing on different system
# - add function parameters to config
# - consider removing budget_functionality_off/flag (or improve switching)
# - remove historical work contributions from worker.contributes? (to free up memory) + remove department history?
# - reorder and annotate config.py (and refactor tests to use config variables?)
# - improve chemistry booster unit test.
# - improve success calculator unit test
# - coverage run -m unittest discover && coverage report

# - check why timesteps are 2 behind in data collector compared to success histroy?
# - refactor: create success history class.
# - write unit test to confirm that Training is fully blocking.
# - wrtie test that idle + active + training == 1000
# - refactor to use Counter or Defaultdict for counting...
# - change FunctionInterface to abstract base class (plot and print never change)
# - rename private data members _XX
# - confirm that skill balance calculations are correct when worker is unable to supply skill due to dept constraint
# - inject strategy into TeamAllocator
# - manually calculate active_projects (80 versus 85)

# - change remote name

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
# (- * add contribution class for Dept.)

def active_project_count(model):
    return model.inventory.active_count


def recent_success_rate(model):
    return np.mean([
        worker.history.get_success_rate()
        for worker in model.schedule.agents
    ])


def number_successful_projects(model):
    # print(model.inventory.success_history)
    return model.inventory.success_history.get(
        model.schedule.steps - 1, 0.0
    )


def number_failed_projects(model):
    return model.inventory.fail_history.get(
        model.schedule.steps - 1, 0.0
    )


def training_workers(model):
    return sum(
        [1 for worker in model.schedule.agents
         if worker.training_remaining > 0]
    )


def idle_workers(model):
    return sum(
        [1 for worker in model.schedule.agents
         if worker.is_free(worker.now, 1)]
    )


def active_workers(model):
    return sum(
        [1 for worker in model.schedule.agents
         if ((not worker.is_free(worker.now, 1))
             and worker.training_remaining == 0)]
    )


class SuperScriptModel(Model):

    def __init__(self, worker_count=WORKER_COUNT,
                 department_count=DEPARTMENT_COUNT,
                 new_projects_per_timestep=NEW_PROJECTS_PER_TIMESTEP,
                 project_length=PROJECT_LENGTH,
                 training_on=TRAINING_ON,
                 budget_functionality_flag=BUDGET_FUNCTIONALITY_FLAG):

        self.worker_count = worker_count
        self.new_projects_per_timestep = new_projects_per_timestep
        self.project_length = project_length
        self.budget_functionality_flag = budget_functionality_flag
        self.new_workers = 0
        self.departments = dict()

        self.G = nx.Graph()
        self.grid = SocialNetwork(self, self.G)
        self.schedule = RandomActivation(self)
        self.inventory = ProjectInventory(
            TeamAllocator(self),
            timeline_flexibility='TimelineFlexibility',
            social_network=self.grid,
            model=self,
            budget_functionality_flag=self.budget_functionality_flag
        )
        self.training_on = training_on
        self.trainer = Trainer(self)

        for di in range(department_count):
            self.departments[di] = Department(di)

        workers_per_department = worker_count / department_count
        assert workers_per_department * department_count == worker_count

        di = 0
        assigned_to_di = 0
        for i in range(self.worker_count):
            w = Worker(i, self, self.departments[di])
            self.schedule.add(w)

            assigned_to_di += 1
            if assigned_to_di == workers_per_department:
                di += 1
                assigned_to_di = 0

        self.grid.initialise()
        self.time = 0  # replace with schedule.steps
        self.running = True

        self.datacollector = DataCollector(
            model_reporters={
                "ActiveProjects": active_project_count,
                "RecentSuccessRate": recent_success_rate,
                "SuccessfulProjects": number_successful_projects,
                "FailedProjects": number_failed_projects,
                "ActiveWorkers": active_workers,
                "IdleWorkers": idle_workers,
                "TrainingWorkers": training_workers},
            # agent_reporters={"RecentSuccessRate": recent_success_rate}
        )

    def step(self):

        #print(active_workers(self), idle_workers(self), training_workers(self))
        #assert (active_workers(self)
        #        + idle_workers(self)
        #        + training_workers(self)
        #        == self.worker_count)
        self.datacollector.collect(self)

        self.trainer.update_skill_quartiles()
        self.inventory.create_projects(self.new_projects_per_timestep,
                                       self.time, self.project_length)
        self.schedule.step()
        self.inventory.remove_null_projects()
        self.time += 1

    def run_model(self, step_count: int):
        for i in range(step_count):
            self.step()
