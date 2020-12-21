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


def safe_mean(x):
    if len(x) > 0:
        return np.mean(x)
    else:
        return 0


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


def av_team_size(model):
    return safe_mean(
        [len(project.team.members)
         for project in model.inventory.projects.values()
         if project.progress >= 0]
    )


def av_success_prob(model):
    return safe_mean(
        [project.success_probability
         for project in model.inventory.projects.values()
         if project.progress >= 0]
    )


def av_worker_ovr(model):
    return safe_mean(
        [worker.skills.ovr for worker in model.schedule.agents]
    )


def av_team_ovr(model):
    return safe_mean(
        [project.team.team_ovr
         for project in model.inventory.projects.values()
         if project.progress >= 0]
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
            model=self
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
                "TrainingWorkers": training_workers,
                "AverageTeamSize": av_team_size,
                "AverageSuccessProbability": av_success_prob,
                "AverageWorkerOvr": av_worker_ovr,
                "AverageTeamOvr": av_team_ovr}
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
