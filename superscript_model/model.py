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
                     TRAINING_MODE,
                     TARGET_TRAINING_LOAD,
                     BUDGET_FUNCTIONALITY_FLAG,
                     UNITS_PER_FTE,
                     DEPARTMENTAL_WORKLOAD,
                     PEER_ASSESSMENT_SUCCESS_MEAN,
                     PEER_ASSESSMENT_SUCCESS_STDEV,
                     PEER_ASSESSMENT_FAIL_MEAN,
                     PEER_ASSESSMENT_FAIL_STDEV,
                     PEER_ASSESSMENT_WEIGHT)


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

    return model.inventory.success_history.get(
        model.schedule.steps - 1, 0.0
    )


def number_failed_projects(model):
    return model.inventory.fail_history.get(
        model.schedule.steps - 1, 0.0
    )


def on_training(model):
    return sum(
        [1 for worker in model.schedule.agents
         if worker.training_remaining > 0]
    )


def no_projects(model):
    return sum(
        [1 for worker in model.schedule.agents
         if worker.contributions.get_units_contributed(worker.now) == 0]
    )


def on_projects(model):
    return sum(
        [1 for worker in model.schedule.agents
         if ((worker.contributions.get_units_contributed(worker.now) > 0)
             and worker.training_remaining == 0)]
    )


def training_load(model):
    return on_training(model) / model.worker_count


def project_load(model):

    project_units = sum([
        worker.contributions.get_units_contributed(worker.now)
        for worker in model.schedule.agents
        if ((worker.contributions.get_units_contributed(worker.now) > 0)
            and worker.training_remaining == 0)
    ])
    return project_units / (model.worker_count * UNITS_PER_FTE)


def departmental_load(model):
    return DEPARTMENTAL_WORKLOAD


def slack(model):
    return 1 - departmental_load(model) - project_load(model) - training_load(model)


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


def worker_turnover(model):
    return model.worker_turnover.get(
        model.schedule.steps - 2, 0.0
    )


class SuperScriptModel(Model):

    def __init__(self, worker_count=WORKER_COUNT,
                 department_count=DEPARTMENT_COUNT,
                 new_projects_per_timestep=NEW_PROJECTS_PER_TIMESTEP,
                 project_length=PROJECT_LENGTH,
                 training_on=TRAINING_ON,
                 training_mode=TRAINING_MODE,
                 target_training_load=TARGET_TRAINING_LOAD,
                 budget_functionality_flag=BUDGET_FUNCTIONALITY_FLAG,
                 peer_assessment_success_mean=PEER_ASSESSMENT_SUCCESS_MEAN,
                 peer_assessment_success_stdev=PEER_ASSESSMENT_SUCCESS_STDEV,
                 peer_assessment_fail_mean=PEER_ASSESSMENT_FAIL_MEAN,
                 peer_assessment_fail_stdev=PEER_ASSESSMENT_FAIL_STDEV,
                 peer_assessment_weight=PEER_ASSESSMENT_WEIGHT):

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
        self.training_mode = training_mode
        self.target_training_load = target_training_load
        self.trainer = Trainer(self)

        self.peer_assessment_success_mean = peer_assessment_success_mean
        self.peer_assessment_success_stdev = peer_assessment_success_stdev
        self.peer_assessment_fail_mean = peer_assessment_fail_mean
        self.peer_assessment_fail_stdev = peer_assessment_fail_stdev
        self.peer_assessment_weight = peer_assessment_weight

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
        self.worker_turnover = dict()
        self.time = 0  # replace with schedule.steps
        self.running = True

        self.datacollector = DataCollector(
            model_reporters={
                "ActiveProjects": active_project_count,
                "RecentSuccessRate": recent_success_rate,
                "SuccessfulProjects": number_successful_projects,
                "FailedProjects": number_failed_projects,
                "WorkersOnProjects": on_projects,
                "WorkersWithoutProjects": no_projects,
                "WorkersOnTraining": on_training,
                "AverageTeamSize": av_team_size,
                "AverageSuccessProbability": av_success_prob,
                "AverageWorkerOvr": av_worker_ovr,
                "AverageTeamOvr": av_team_ovr,
                "WorkerTurnover": worker_turnover,
                "ProjectLoad": project_load,
                "TrainingLoad": training_load,
                "DeptLoad": departmental_load,
                "Slack": slack}
        )

    def step(self):

        self.trainer.update_skill_quartiles()
        self.inventory.create_projects(self.new_projects_per_timestep,
                                       self.time, self.project_length)
        self.schedule.step()
        self.trainer.train()
        self.inventory.remove_null_projects()
        self.time += 1

        self.datacollector.collect(self)
        assert (on_projects(self)
                + no_projects(self)
                + on_training(self)
                == self.worker_count)

    def run_model(self, step_count: int):
        for i in range(step_count):
            self.step()
