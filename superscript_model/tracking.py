"""
SuperScript data collection module.
===================================

Methods and objects used to for data collection during simulation.

SSDataCollector: subclass of Mesa.datacollection.DataCollector. Uses
    the tracking functions defined in this file.

"""
import numpy as np
from mesa.datacollection import DataCollector

from .config import (UNITS_PER_FTE,
                     DEPARTMENTAL_WORKLOAD)


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


def number_null_projects(model):
    return model.inventory.null_count


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


def project_load(model, units_per_fte=UNITS_PER_FTE):

    project_units = sum([
        worker.contributions.get_units_contributed(worker.now)
        for worker in model.schedule.agents
        if ((worker.contributions.get_units_contributed(worker.now) > 0)
            and worker.training_remaining == 0)
    ])
    return project_units / (model.worker_count * units_per_fte)


def departmental_load(model):
    return model.departmental_workload


def slack(model):
    return (1
            - departmental_load(model)
            - project_load(model)
            - training_load(model))


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
        model.schedule.steps - 1, 0.0
    )


def projects_per_worker(model):

    project_count = 0
    for worker in model.schedule.agents:

        if worker.now in worker.contributions.per_skill_contributions.keys():
            project_count += (
                len(set.union(*[
                    set(projects)
                    for projects
                    in (worker.contributions
                              .per_skill_contributions[worker.now]
                              .values())
                ]))
            )
    return project_count / model.worker_count


def worker_ovr(worker):
    return worker.skills.ovr


def worker_hard_skills(worker):
    return worker.skills.hard_skills


def worker_training_tracker(worker):
    return worker.skills.training_tracker


def worker_skill_decay_tracker(worker):
    return worker.skills.skill_decay_tracker


def worker_peer_assessment_tracker(worker):
    return worker.skills.peer_assessment_tracker


class SSDataCollector(DataCollector):
    """Data collection class.

    Uses functions defined above to track model-level,
    agent-level and table data.

    Tables are used to track project and team specific data.

    Functions for model and agent reporters take model and agents
    instances respectively.
    """
    def __init__(self):

        model_reporters = {
            "ActiveProjects": active_project_count,
            "RecentSuccessRate": recent_success_rate,
            "SuccessfulProjects": number_successful_projects,
            "FailedProjects": number_failed_projects,
            "NullProjects": number_null_projects,
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
            "Slack": slack,
            "ProjectsPerWorker": projects_per_worker
            }
        agent_reporters = {
            "now": "now",
            "contributes": "contributes_now",
            "ovr": worker_ovr,
            "hard_skills": worker_hard_skills,
            "training": worker_training_tracker,
            "skill_decay": worker_skill_decay_tracker,
            "peer_assessment": worker_peer_assessment_tracker
        }
        tables = {
            "Projects": {"project_id": [],
                         "prob": [],
                         "risk": [],
                         "budget": [],
                         "null": [],
                         "success": [],
                         "maximum_offset": [],
                         "realised_offset": [],
                         "start_time": [],
                         "ovr_prob_cpt": [],
                         "skill_balance_prob_cpt": [],
                         "creativity_match_prob_cpt": [],
                         "risk_prob_cpt": [],
                         "chemistry_prob_cpt": [],
                         "team_budget": [],
                         "team_ovr": [],
                         "team_creativity_match": [],
                         "team_size": []
                         }
        }
        super().__init__(model_reporters=model_reporters,
                         agent_reporters=agent_reporters,
                         tables=tables)
