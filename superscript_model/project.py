import numpy as np
import json

from .function import FunctionFactory
from .utilities import Random
from .config import (MAXIMUM_TIMELINE_FLEXIBILITY,
                     PROJECT_LENGTH,
                     DEFAULT_START_OFFSET,
                     DEFAULT_START_TIME,
                     P_HARD_SKILL_PROJECT,
                     PER_SKILL_MAX_UNITS,
                     PER_SKILL_MIN_UNITS,
                     MIN_REQUIRED_UNITS,
                     MIN_SKILL_LEVEL,
                     MAX_SKILL_LEVEL,
                     MIN_PROJECT_CREATIVITY,
                     MAX_PROJECT_CREATIVITY,
                     RISK_LEVELS,
                     P_BUDGET_FLEXIBILITY,
                     MAX_BUDGET_INCREASE,
                     HARD_SKILLS)


class ProjectInventory:

    def __init__(self,
                 team_allocator,
                 timeline_flexibility='NoFlexibility',
                 max_timeline_flex = MAXIMUM_TIMELINE_FLEXIBILITY):

        self.projects = dict()
        self.index_total = 0
        self.team_allocator = team_allocator
        self.timeline_flexibility_func = (
            FunctionFactory.get(timeline_flexibility)
        )
        self.max_timeline_flex = max_timeline_flex
        self.success_calculator = SuccessCalculator()

    @property
    def active_count(self):
        return sum([1 for p
                    in self.projects.values()
                    if p.progress >= 0])

    def get_start_time_offset(self):

        p_vector = (
            self.timeline_flexibility_func
            .get_values(np.arange(self.max_timeline_flex + 1))
        )

        r = Random.uniform()
        for i in np.flip(np.arange(1, self.max_timeline_flex)):
            if r <= p_vector[i]:
                return i

        return 0
        # if r <= p_vector[4]:
        #     return 4
        # elif r <= p_vector[3]:
        #     return 3
        # elif r <= p_vector[2]:
        #     return 3
        # elif r <= p_vector[1]:
        #     return 3
        # else:
        #     return 0

    def create_projects(self, new_projects_count,
                        time, length):

        new_projects = []
        for i in range(new_projects_count):
            p = Project(
                self, self.index_total + i,
                project_length=length,
                start_time_offset=self.get_start_time_offset(),
                start_time=time
            )
            new_projects.append(p)

        new_projects = self.rank_projects(new_projects)
        for p in new_projects:
            self.team_allocator.allocate_team(p)
            self.success_calculator.calculate_success_probability(p)
            self.add_project(p)

    @staticmethod
    def rank_projects(project_list):
        project_list.sort(reverse=True, key=lambda x: (x.risk, x.creativity))
        return project_list

    def add_project(self, project):

        if project.project_id not in self.projects.keys():
            self.projects[project.project_id] = project
            self.index_total += 1
        else:
            raise KeyError('Project ID %d already exists in inventory.'
                           % project.project_id)

    def delete_project(self, project_id):
        try:
            del self.projects[project_id]
        except KeyError:
            print('Project ID %d not in inventory.' % project_id)
            raise

    def advance_projects(self):
        """ Allows projects to be deleted/terminated during loop"""
        project_ids = list(self.projects.keys())

        for pid in project_ids:
            if pid in self.projects.keys():
                self.projects[pid].advance()


class Project:

    def __init__(self,
                 inventory: ProjectInventory,
                 project_id=42,
                 project_length=PROJECT_LENGTH,
                 start_time_offset=DEFAULT_START_OFFSET,
                 start_time=DEFAULT_START_TIME):

        self.inventory = inventory
        self.project_id = project_id
        self.length = project_length
        self.progress = 0 - start_time_offset
        self.start_time = start_time + start_time_offset
        self.team = None
        self.requirements = ProjectRequirements()
        self.success_probability = 0.0

    def advance(self):
        self.progress += 1
        if self.progress == self.length:
            self.terminate()

    def terminate(self):
        if self.team is not None:
            self.team.remove_lead(self)
            self.team = None

        self.inventory.delete_project(self.project_id)

    @property
    def required_skills(self):
        return self.requirements.get_required_skills()

    @property
    def risk(self):
        return self.requirements.risk

    @property
    def creativity(self):
        return self.requirements.creativity

    def get_skill_requirement(self, skill):
        return self.requirements.hard_skills[skill]


class ProjectRequirements:

    def __init__(self,
                 p_hard_skill_required=P_HARD_SKILL_PROJECT,
                 per_skill_max=PER_SKILL_MAX_UNITS,
                 per_skill_min=PER_SKILL_MIN_UNITS,
                 min_skill_required=MIN_REQUIRED_UNITS,
                 hard_skills=HARD_SKILLS,
                 min_skill_level = MIN_SKILL_LEVEL,
                 max_skill_level = MAX_SKILL_LEVEL,
                 min_project_creativity=MIN_PROJECT_CREATIVITY,
                 max_project_creativity=MAX_PROJECT_CREATIVITY,
                 risk_levels=RISK_LEVELS,
                 p_budget_flexibility=P_BUDGET_FLEXIBILITY,
                 max_budget_increase=MAX_BUDGET_INCREASE):

        self.risk = Random.choice(risk_levels)
        self.creativity = Random.randint(min_project_creativity,
                                         max_project_creativity)
        self.flexible_budget = (
            True if Random.uniform() <= p_budget_flexibility else False
        )
        self.max_budget_increase = max_budget_increase

        self.p_hard_skill_required = p_hard_skill_required
        self.min_skill_required = min_skill_required
        self.per_skill_max = per_skill_max
        self.per_skill_min = per_skill_min
        self.min_skill_level = min_skill_level
        self.max_skill_level = max_skill_level

        self.hard_skills = dict(zip(hard_skills,
                                    [{
                                        'level': None,
                                        'units': 0}
                                        for s in hard_skills])
                                )
        self.total_skill_units = None

        max_assigned_units = 0
        while max_assigned_units < self.min_skill_required:

            self.assign_skill_requirements()
            max_assigned_units = max(
                [s['units'] for s in self.hard_skills.values()
                 if s['level'] is not None]
            )

    def select_non_zero_skills(self):

        n_skills = 0
        while n_skills == 0:

            non_zero_skills = [s for s in self.hard_skills.keys()
                               if Random.uniform() <= self.p_hard_skill_required]
            Random.shuffle([non_zero_skills])
            n_skills = len(non_zero_skills)

        self.total_skill_units = Random.randint(n_skills * self.per_skill_min + 1,
                                                n_skills * self.per_skill_max)
        return n_skills, non_zero_skills

    def assign_skill_requirements(self):

        n_skills, non_zero_skills = self.select_non_zero_skills()
        remaining_skill_units = self.total_skill_units
        for i, skill in enumerate(non_zero_skills):

            a = (remaining_skill_units
                 - (n_skills - (i + 1)) * self.per_skill_max)
            a = max(a, self.per_skill_min)

            b = (remaining_skill_units
                 - (n_skills - (i + 1)) * self.per_skill_min)
            b = min(b, self.per_skill_max)

            units = Random.randint(a, b)
            self.hard_skills[skill]['level'] = Random.randint(
                self.min_skill_level, self.max_skill_level
            )
            self.hard_skills[skill]['units'] = units
            remaining_skill_units -= units

    def get_required_skills(self):
        return [skill for skill
                in self.hard_skills.keys()
                if self.hard_skills[skill]['level'] is not None]

    def to_string(self):

        output = {
            'risk': self.risk,
            'creativity': self.creativity,
            'flexible_budget': self.flexible_budget,
            'max_budget_increase': self.max_budget_increase,
            'p_hard_skill_required': self.p_hard_skill_required,
            'min_skill_required': self.min_skill_required,
            'per_skill_cap': self.per_skill_max,
            'total_skill_units': self.total_skill_units,
            'hard_skills': self.hard_skills
        }
        return json.dumps(output, indent=4)


class SuccessCalculator:

    def __init__(self):
        self.probability_ovr = (
            FunctionFactory.get('SuccessProbabilityOVR')
        )
        self.probability_skill_balance = (
            FunctionFactory.get('SuccessProbabilitySkillBalance')
        )
        self.probability_creativity_match = (
            FunctionFactory.get('SuccessProbabilityCreativityMatch')
        )
        self.probability_risk = (
            FunctionFactory.get('SuccessProbabilityRisk')
        )

    def calculate_success_probability(self, project):

        if project.team is not None:
            ovr = project.team.team_ovr
            skill_balance = project.team.skill_balance
            creativity_match = project.team.creativity_match
            risk = project.risk
        else:
            ovr = 0.0
            skill_balance = 0.0
            creativity_match = 0.0
            risk = 0.0

        project.success_probability = (
            self.probability_ovr.get_values(ovr)
            + self.probability_skill_balance.get_values(skill_balance)
            + self.probability_creativity_match.get_values(creativity_match)
            + self.probability_risk.get_values(risk)
        ) / 100

    def determine_success(self):
        """To be called when settling project (terminate?)"""
        pass