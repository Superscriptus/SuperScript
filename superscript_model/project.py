import numpy as np
import json

from .function import FunctionFactory
from .utilities import Random


class ProjectInventory:

    def __init__(self,
                 team_allocator,
                 timeline_flexibility='NoFlexibility'):

        self.projects = dict()
        self.index_total = 0
        self.team_allocator = team_allocator
        self.timeline_flexibility_func = (
            FunctionFactory.get(timeline_flexibility)
        )

    @property
    def active_count(self):
        return sum([1 for p
                    in self.projects.values()
                    if p.progress >= 0])

    def get_start_time_offset(self):

        p_vector = (self.timeline_flexibility_func
                    .get_values(np.arange(5)))

        r = Random.uniform()
        if r <= p_vector[4]:
            return 4
        elif r <= p_vector[3]:
            return 3
        elif r <= p_vector[2]:
            return 3
        elif r <= p_vector[1]:
            return 3
        else:
            return 0

    def create_projects(self, new_projects_count):

        for i in range(new_projects_count):
            p = Project(
                self, self.index_total,
                start_time_offset=self.get_start_time_offset()
            )
            self.team_allocator.allocate_team(p)
            self.add_project(p)

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
                 project_length=5,
                 start_time_offset=0):

        self.inventory = inventory
        self.project_id = project_id
        self.length = project_length
        self.progress = 0 - start_time_offset
        self.team = None
        self.requirements = ProjectRequirements()

    def advance(self):
        self.progress += 1
        if self.progress == self.length:
            self.terminate()

    def terminate(self):
        if self.team is not None:
            self.team.remove_lead(self)
            self.team = None

        self.inventory.delete_project(self.project_id)


class ProjectRequirements:

    def __init__(self,
                 p_hard_skill_required=0.8,
                 per_skill_max=10,
                 per_skill_min=1,
                 min_skill_required=2,
                 hard_skills=['A','B','C','D','E'],
                 min_project_creativity=1,
                 max_project_creativity=5,
                 risk_levels=[5,10,25],
                 p_budget_flexibility=0.25,
                 max_budget_increase=0.25):

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

        self.hard_skills = dict(zip(hard_skills,
                                    [{
                                        'level': None,
                                        'units': 0}
                                        for s in hard_skills])
                                )
        self.total_skill_units = None

        max_assigned_units = 0
        while max_assigned_units < 2:

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
            self.hard_skills[skill]['level'] = Random.randint(1, 5)
            self.hard_skills[skill]['units'] = units
            remaining_skill_units -= units

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