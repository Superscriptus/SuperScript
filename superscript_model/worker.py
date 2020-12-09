from mesa import Agent
from interface import Interface, implements
import json

from .project import Project
from .utilities import Random


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

    def assign_as_lead(self, project):
        self.leads_on[project.project_id] = project

    def remove_as_lead(self, project):
        del self.leads_on[project.project_id]

    def step(self):
        """Dict can be updated during loop (one other?)"""
        projects = list(self.leads_on.values())

        for project in projects:
            if project in self.leads_on.values():
                project.advance()


class SkillMatrix:

    def __init__(self,
                 hard_skills=['A','B','C','D','E'],
                 soft_skills=['F','G','H','I','J'],
                 max_skill=5,
                 hard_skill_probability=0.8,
                 round_to=1,
                 ovr_multiplier=20):

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

        output = {"Hard skills": self.hard_skills,
                  "Soft skills": self.soft_skills,
                  "Hard skill probability":
                      self.hard_skill_probability,
                  "OVR multiplier": self.ovr_multiplier}

        return json.dumps(output)



