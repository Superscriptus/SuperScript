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
                     WORKER_SUCCESS_HISTORY_THRESHOLD,
                     SKILL_DECAY_FACTOR)


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
        self.history = WorkerHistory()
        self.training_remaining = 0

    @property
    def contributes(self):
        return self.contributions.get_contributions()

    @property
    def now(self):
        return self.model.schedule.steps

    @property
    def training_horizon(self):
        return self.model.trainer.training_length

    def assign_as_lead(self, project):
        self.leads_on[project.project_id] = project

    def remove_as_lead(self, project):
        del self.leads_on[project.project_id]

    def step(self):

        self.training_remaining -= 1  # refactor: self.model.trainer.advance_training()
        self.training_remaining = max(self.training_remaining, 0)

        """Dict can be updated during loop (one other?)"""
        projects = list(self.leads_on.values())

        for project in projects:
            if project in self.leads_on.values():
                project.advance()

        if self.is_free(self.now, self.training_horizon):  # refactor: conditional to Trainer method
            self.model.trainer.train(self)

        self.skills.decay(self)

    def get_skill(self, skill, hard_skill=True):
        if hard_skill:
            return self.skills.hard_skills[skill]
        else:
            return self.skills.soft_skills[skill]

    def is_free(self, start, length):
        return self.contributions.is_free_over_period(start, length)

    def replace(self):

        self.department.number_of_workers -= 1
        self.model.new_workers += 1

        if self.now in self.model.worker_turnover.keys():
            self.model.worker_turnover[self.now] += 1
        else:
            self.model.worker_turnover[self.now] = 1

        w = Worker(
            self.model.worker_count + self.model.new_workers,
            self.model, self.department
        )
        self.model.schedule.add(w)
        self.model.grid.replace_worker(self, w)
        self.model.schedule.remove(self)

    def bid(self, project):
        return self.strategy.bid(project, self)

    def individual_chemistry(self, project):
        chemistry = 0

        if (len(set(self.skills.top_two)
                .intersection(project.required_skills)) > 0):
            chemistry += 1

        chemistry += self.history.momentum()
        chemistry += project.risk >= 0.1 * self.skills.ovr
        return chemistry


class WorkerContributions:
    """Class that logs current and future contributions to projects."""
    def __init__(self, worker, units_per_full_time=UNITS_PER_FTE):
        self.per_skill_contributions = dict()
        self.total_contribution = dict()
        self.worker = worker
        self.units_per_full_time = units_per_full_time

    def get_contributions(self, time=None):
        if time is None:
            return self.per_skill_contributions
        elif time in self.per_skill_contributions.keys():
            return self.per_skill_contributions[time]
        else:
            return {}

    def get_skill_units_contributed(self, time, skill):
        contributions = self.get_contributions(time)

        if skill in contributions.keys():
            return contributions[skill]
        else:
            return 0

    def add_contribution(self, project, skill):

        for time_offset in range(project.length):
            time = project.start_time + time_offset

            if time not in self.per_skill_contributions.keys():
                self.per_skill_contributions[time] = {
                    skill: []
                    for skill in self.worker.skills.hard_skills.keys()
                }
            (self.per_skill_contributions[time][skill]
             .append(project.project_id))

            if time not in self.total_contribution.keys():
                self.total_contribution[time] = 1
            else:
                self.total_contribution[time] += 1

    def get_units_contributed(self, time):
        if time not in self.total_contribution.keys():
            return 0
        else:
            return self.total_contribution[time]

    def contributes_less_than_full_time(self, start, length):

        for t in range(length):

            time = start + t
            contributes_at_time = self.get_units_contributed(time)
            if contributes_at_time >= self.units_per_full_time:
                return False

        return True

    def get_remaining_units(self, start, length):

        remaining_units = []
        for t in range(length):

            time = start + t
            contributes_at_time = self.get_units_contributed(time)
            remaining_units.append(
                self.units_per_full_time - contributes_at_time
            )
        return min(remaining_units)

    def is_free_over_period(self, start, length):
        if ((self.get_remaining_units(start, length)
                == self.units_per_full_time)
            and (self.worker.department
                     .is_workload_satisfied(
                        start, length))):
            return True
        else:
            return False


class WorkerHistory:
    """Class to track recent worker's success rate."""
    def __init__(self, success_history_length=WORKER_SUCCESS_HISTORY_LENGTH,
                 success_history_threshold=WORKER_SUCCESS_HISTORY_THRESHOLD):

        self.success_history_length = success_history_length
        self.success_history_threshold = success_history_threshold
        self.success_history = []

    def record(self, success):
        self.success_history.append(success)
        if len(self.success_history) > self.success_history_length:
            self.success_history.pop(0)

    def get_success_rate(self):
        if len(self.success_history) == 0:
            return 0
        else:
            return sum(self.success_history) / len(self.success_history)

    def momentum(self):
        return self.get_success_rate() >= self.success_history_threshold


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
                 ovr_multiplier=WORKER_OVR_MULTIPLIER,
                 skill_decay_factor=SKILL_DECAY_FACTOR):

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
        self.skill_decay_factor = skill_decay_factor
        self.round_to = round_to

        while sum(self.hard_skills.values()) == 0.0:
            self.assign_hard_skills()

    @property
    def ovr(self):

        return (sum([s for s in
                     self.hard_skills.values()
                     if s > 0.0])
                / sum([1 for s in
                       self.hard_skills.values()
                       if s > 0.0])
                ) * self.ovr_multiplier

    @property
    def top_two(self):
        ranked_skills = {
            k: v for k, v in sorted(
            self.hard_skills.items(),
            reverse=True,
            key=lambda item: item[1]
        )}
        return list(ranked_skills.keys())[:2]

    def assign_hard_skills(self):

        for key in self.hard_skills.keys():
            if Random.uniform() <= self.hard_skill_probability:
                self.hard_skills[key] = Random.uniform(
                    0.0, self.max_skill)

    def decay(self, worker):

        for skill in self.hard_skills.keys():

            units = (
                worker.contributions
                      .get_skill_units_contributed(worker.now, skill)
            )
            if units == 0:
                self.hard_skills[skill] *= self.skill_decay_factor

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
