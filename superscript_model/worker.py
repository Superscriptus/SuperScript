from mesa import Agent
from interface import Interface, implements
import json

from .project import Project
from .utilities import Random
from .organisation import Department
from .config import (HARD_SKILLS,
                     SOFT_SKILLS,
                     MAX_SKILL_LEVEL,
                     MIN_SOFT_SKILL_LEVEL,
                     P_HARD_SKILL,
                     WORKER_OVR_MULTIPLIER,
                     PRINT_DECIMALS_TO,
                     UNITS_PER_FTE,
                     WORKER_SUCCESS_HISTORY_LENGTH,
                     WORKER_SUCCESS_HISTORY_THRESHOLD)


class Worker(Agent):

    def __init__(self, worker_id: int,
                 model, department=Department(0)):

        super().__init__(worker_id, model)
        self.worker_id = worker_id
        self.skills = SkillMatrix()
        self.department = department
        self.department.number_of_workers += 1

        self.strategy = AllInStrategy('All-In')
        self.leads_on = dict()
        self.contributions = WorkerContributions(self)

    @property
    def contributes(self):
        return self.contributions.get_contributions()

    @property
    def recent_success_rate(self):
        return self.contributions.get_success_rate()

    @property
    def now(self):
        return self.model.schedule.steps

    @property
    def training_horizon(self):
        return self.department.trainer.training_length

    def assign_as_lead(self, project):
        self.leads_on[project.project_id] = project

    def remove_as_lead(self, project):
        del self.leads_on[project.project_id]

    def step(self):

        if self.is_free(self.now, self.training_horizon):
            self.department.trainer.train(self)

        """Dict can be updated during loop (one other?)"""
        projects = list(self.leads_on.values())

        for project in projects:
            if project in self.leads_on.values():
                project.advance()

    def get_skill(self, skill, hard_skill=True):
        if hard_skill:
            return self.skills.hard_skills[skill]
        else:
            return self.skills.soft_skills[skill]

    def is_free(self, start, length):
        return self.contributions.is_free_over_period(start, length)

    def replace(self):
        # ensure to reduce number of workers in dept by 1
        pass

    def bid(self, project):
        return self.strategy.bid(project, self)


class WorkerContributions:
    """Class that logs current and future contributions to projects."""
    def __init__(self, worker, units_per_full_time=UNITS_PER_FTE,
                 success_history_length=WORKER_SUCCESS_HISTORY_LENGTH,
                 success_history_threshold=WORKER_SUCCESS_HISTORY_THRESHOLD):
        self.contributes = dict()
        self.worker = worker
        self.units_per_full_time = units_per_full_time
        self.success_history = dict()

    def get_contributions(self):
        return self.contributes

    def add_contribution(self, project, skill):

        for time_offset in range(project.length):
            time = project.start_time + time_offset

            if time not in self.contributes.keys():
                self.contributes[time] = {
                    skill: []
                    for skill in self.worker.skills.hard_skills.keys()
                }
            (self.contributes[time][skill]
             .append(project.project_id))

    def get_units_contributed(self, time, skill):
        if time not in self.contributes.keys():
            return 0
        elif skill not in self.contributes[time].keys():
            return 0
        else:
            return len(self.contributes[time][skill])

    def contributes_less_than_full_time(self, start, length):

        for t in range(length):

            time = start + t
            contributes_at_time = 0
            for skill in self.worker.skills.hard_skills.keys():
                contributes_at_time += self.get_units_contributed(time, skill)

            if contributes_at_time >= self.units_per_full_time:
                return False

        return True

    def get_remaining_units(self, start, length):

        remaining_units = []
        for t in range(length):

            time = start + length
            contributes_at_time = 0
            for skill in self.worker.skills.hard_skills.keys():
                contributes_at_time += self.get_units_contributed(time, skill)

            remaining_units.append(
                self.units_per_full_time - contributes_at_time
            )
        return min(remaining_units)

    def get_success_rate(self):
        return self.success_history

    def is_free_over_period(self, start, length):
        if ((self.get_remaining_units(start, length)
                == self.units_per_full_time)
            and (self.worker.department
                     .is_workload_satisfied(
                        start, length)
                )):
            return True
        else:
            return False

class WorkerStrategyInterface(Interface):

    def bid(self, project: Project, worker: Worker) -> bool:
        pass

    def accept(self, project: Project) -> bool:
        pass


class AllInStrategy(implements(WorkerStrategyInterface)):

    def __init__(self, name: str):
        self.name = name

    def bid(self, project: Project, worker: Worker) -> bool:

        if (worker.department.is_workload_satisfied(
                project.start_time, project.length)
            and
            worker.contributions.contributes_less_than_full_time(
                project.start_time, project.length)):
            return True
        else:
            return False

    def accept(self, project: Project) -> bool:
        return True


class SkillMatrix:

    def __init__(self,
                 hard_skills=HARD_SKILLS,
                 soft_skills=SOFT_SKILLS,
                 max_skill=MAX_SKILL_LEVEL,
                 min_soft_skill=MIN_SOFT_SKILL_LEVEL,
                 hard_skill_probability=P_HARD_SKILL,
                 round_to=PRINT_DECIMALS_TO,
                 ovr_multiplier=WORKER_OVR_MULTIPLIER):

        self.hard_skills = dict(zip(hard_skills,
                                    [0.0 for s in hard_skills]))
        self.soft_skills = dict(
            zip(soft_skills,
                [Random.uniform(min_soft_skill, max_skill)
                 for s in soft_skills]
                )
        )
        self.max_skill = max_skill
        self.hard_skill_probability = hard_skill_probability
        self.ovr_multiplier = ovr_multiplier
        self.round_to = round_to

        while sum(self.hard_skills.values()) == 0.0:
            self.assign_hard_skills()

    def assign_hard_skills(self):

        for key in self.hard_skills.keys():
            if Random.uniform() <= self.hard_skill_probability:
                self.hard_skills[key] = Random.uniform(
                    0.0, self.max_skill)

    def to_string(self):

        output = {
            "Worker OVR":
                round(self.ovr, self.round_to),
            "Hard skills":
                [round(s, self.round_to)
                 for s in self.hard_skills.values()],
            "Soft skills":
                [round(s, self.round_to)
                 for s in self.soft_skills.values()],
            "Hard skill probability":
                self.hard_skill_probability,
            "OVR multiplier": self.ovr_multiplier}

        return json.dumps(output, indent=4)

    @property
    def ovr(self):

        return (sum([s for s in
                     self.hard_skills.values()
                     if s > 0.0])
                / sum([1 for s in
                       self.hard_skills.values()
                       if s > 0.0])
                ) * self.ovr_multiplier
