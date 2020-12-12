from mesa import Agent
from interface import Interface, implements
import json

from .project import Project
from .utilities import Random
from .config import (HARD_SKILLS,
                     SOFT_SKILLS,
                     MAX_SKILL_LEVEL,
                     P_HARD_SKILL,
                     WORKER_OVR_MULTIPLIER,
                     PRINT_DECIMALS_TO)


class WorkerStrategyInterface(Interface):

    def bid(self, project: Project) -> bool:
        pass

    def accept(self, project: Project) -> bool:
        pass


class AllInStrategy(implements(WorkerStrategyInterface)):

    def __init__(self, name: str):
        self.name = name

    def bid(self, project: Project) -> bool:
        return True

    def accept(self, project: Project) -> bool:
        return True


class Worker(Agent):

    def __init__(self, worker_id: int, model):

        self.worker_id = worker_id
        self.skills = SkillMatrix()
        super().__init__(worker_id, model)
        self.strategy = AllInStrategy('All-In')
        self.leads_on = dict()
        self.contributes = dict()

    def assign_as_lead(self, project):
        self.leads_on[project.project_id] = project

    def remove_as_lead(self, project):
        del self.leads_on[project.project_id]

    def add_contribution(self, project, skill):

        for time_offset in range(project.length):
            time = project.start_time + time_offset

            if time not in self.contributes.keys():
                self.contributes[time] = {
                    skill: []
                    for skill in self.skills.hard_skills.keys()
                }
            (self.contributes[time][skill]
             .append(project.project_id))

    def step(self):
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


class SkillMatrix:

    def __init__(self,
                 hard_skills=HARD_SKILLS,
                 soft_skills=SOFT_SKILLS,
                 max_skill=MAX_SKILL_LEVEL,
                 hard_skill_probability=P_HARD_SKILL,
                 round_to=PRINT_DECIMALS_TO,
                 ovr_multiplier=WORKER_OVR_MULTIPLIER):

        self.hard_skills = dict(zip(hard_skills,
                                    [0.0 for s in hard_skills]))
        self.soft_skills = dict(
            zip(soft_skills,
                [Random.uniform(0.0, max_skill) for s in hard_skills]
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


