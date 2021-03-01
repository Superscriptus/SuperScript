import numpy as np
import json
import pickle

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
                 max_timeline_flex=MAXIMUM_TIMELINE_FLEXIBILITY,
                 hard_skills=HARD_SKILLS,
                 social_network=None,
                 model=None,
                 save_flag=False,
                 load_flag=False):

        self.projects = dict()
        self.null_projects = dict()
        self.null_count = 0
        self.index_total = 0
        self.team_allocator = team_allocator
        self.timeline_flexibility_func = (
            FunctionFactory.get(timeline_flexibility)
        )
        self.max_timeline_flex = max_timeline_flex

        self.success_calculator = SuccessCalculator()
        self.success_history = dict()
        self.fail_history = dict()

        self.total_skill_requirement = dict(zip(
            hard_skills, [0 for s in hard_skills]
        ))
        self.social_network = social_network
        self.model = model
        self.skill_update_func = (FunctionFactory.get(
            'SkillUpdateByRisk'
        ) if self.model.update_skill_by_risk_flag
          else FunctionFactory.get('IdentityFunction'))

        self.save_flag = save_flag
        if self.save_flag:
            self.all_projects = {}

        self.load_flag = load_flag
        assert not self.save_flag & self.load_flag

        if self.load_flag:
            try:
                with open('project_file.pickle', 'rb') as ifile:
                    self.all_projects = pickle.load(ifile)
            except FileNotFoundError:
                print(
                    'Cannot load predefined projects: '
                    'project_file.pickle not found.'
                )
                self.load_flag = False

    @property
    def active_count(self):
        return sum([1 for p
                    in self.projects.values()
                    if p.progress >= 0])

    @property
    def top_two_skills(self):
        return list(self.total_skill_requirement.keys())[:2]

    def remove_null_projects(self):

        nulls = list(self.null_projects.keys())
        self.null_count = len(nulls)

        for project_id in nulls:
            self.log_project_data_collector(
                self.null_projects[project_id], null=True, success=False
            )
            self.delete_project(project_id)
            del self.null_projects[project_id]

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

    def determine_total_skill_requirements(self, projects):

        for skill in self.total_skill_requirement.keys():
            self.total_skill_requirement[skill] = sum([
                project.requirements.hard_skills[skill]['units']
                for project in projects
            ])
        self.rank_total_requirements()

    def rank_total_requirements(self):
        self.total_skill_requirement = {
            k: v for k, v in sorted(
                self.total_skill_requirement.items(),
                reverse=True,
                key=lambda item: item[1]
            )
        }

    def create_projects(self, new_projects_count,
                        time, length):

        auto_offset = (
            False if self.model.organisation_strategy == 'Basin'
            else True
        )

        if self.load_flag:
            new_projects = self.all_projects.get(time, [])
        else:
            new_projects = []
            for i in range(new_projects_count):
                p = Project(
                    self, self.index_total + i,
                    project_length=length,
                    start_time_offset=self.get_start_time_offset(),
                    start_time=time,
                    auto_offset=auto_offset
                )
                new_projects.append(p)

            self.determine_total_skill_requirements(new_projects)
            new_projects = self.rank_projects(new_projects)

        if self.save_flag:
            self.all_projects[time] = new_projects

        for p in new_projects:
            self.team_allocator.allocate_team(p)
            self.success_calculator.calculate_success_probability(p)
            self.add_project(p)

    def save_projects(self):
        if self.save_flag:
            with pickle.open('project_file.pickle', 'wb') as ofile:
                pickle.dump(self.all_projects, ofile)

    @staticmethod
    def rank_projects(project_list):
        project_list.sort(reverse=True, key=lambda x: (x.risk, x.creativity))
        return project_list

    def add_project(self, project):

        if project.team is None or project.team.lead is None:
            self.null_projects[project.project_id] = project

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

    def log_project_data_collector(self, project, null, success):

        next_row = {"project_id": project.project_id,
                    "prob": project.success_probability,
                    "risk": project.risk,
                    "budget": project.budget,
                    "null": null,
                    "success": success}

        self.model.datacollector.add_table_row(
            "Projects", next_row, ignore_missing=False
        )


class Project:

    def __init__(self,
                 inventory: ProjectInventory,
                 project_id=42,
                 project_length=PROJECT_LENGTH,
                 start_time_offset=DEFAULT_START_OFFSET,
                 start_time=DEFAULT_START_TIME,
                 auto_offset=True):

        self.inventory = inventory
        self.project_id = project_id
        self.length = project_length
        self.progress = 0 - start_time_offset
        self.start_time_offset = start_time_offset
        self.start_time = (start_time + start_time_offset
                           if auto_offset else start_time)
        self.team = None
        self.requirements = ProjectRequirements(
            budget_functionality_flag
            =self.inventory.model.budget_functionality_flag
        )
        self.success_probability = 0.0

    def advance(self):
        self.progress += 1
        if self.progress >= self.length:
            self.terminate()

    def terminate(self):

        success = (
            self.inventory.success_calculator.determine_success(self)
        )
        if self.team is not None:
            self.team.skill_update(
                success, self.inventory.skill_update_func
            )
            self.team.log_project_outcome(success)
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

    @property
    def chemistry(self):

        chemistry = np.mean(
            [member.individual_chemistry(self)
             for member in self.team.members.values()]
        )
        if self.inventory.social_network is not None:
            chemistry += (
                self.inventory.social_network
                    .get_team_historical_success_flag(self.team)
            )
        return chemistry

    @property
    def budget(self):
        return self.requirements.budget

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
                 max_budget_increase=MAX_BUDGET_INCREASE,
                 budget_functionality_flag=True):

        self.risk = Random.choice(risk_levels)
        self.creativity = Random.randint(min_project_creativity,
                                         max_project_creativity)
        self.flexible_budget = (
            True if Random.uniform() <= p_budget_flexibility else False
        )

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
        self.budget_functionality_flag = budget_functionality_flag
        self.budget = self.calculate_budget(self.flexible_budget,
                                            max_budget_increase)

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

    def calculate_budget(self, flexible_budget_flag,
                         max_budget_increase):

        if not self.budget_functionality_flag:
            return None

        budget = 0
        for skill in self.get_required_skills():
            budget += (self.hard_skills[skill]['units']
                       * self.hard_skills[skill]['level'])

        if flexible_budget_flag:
            budget *= max_budget_increase

        return budget

    def to_string(self):

        output = {
            'risk': self.risk,
            'creativity': self.creativity,
            'flexible_budget': self.flexible_budget,
            'budget': self.budget,
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
        self.probability_chemistry = (
            FunctionFactory.get('SuccessProbabilityChemistry')
        )
        self.ovr = 0.0
        self.skill_balance = 0.0
        self.creativity_match = 0.0
        self.risk = 0.0
        self.chemistry = 0.0

    def get_component_values(self, project):

        if project.team is not None:
            self.ovr = project.team.team_ovr
            self.skill_balance = project.team.skill_balance
            self.creativity_match = project.team.creativity_match
            self.risk = project.risk
            self.chemistry = project.chemistry
        else:
            self.ovr = 0.0
            self.skill_balance = 0.0
            self.creativity_match = 0.0
            self.risk = 0.0
            self.chemistry = 0.0

    def calculate_success_probability(self, project):

        if project.team is None or project.team.lead is None:
            project.success_probability = 0
        else:
            self.get_component_values(project)
            probability = (
                self.probability_ovr.get_values(self.ovr)
                + self.probability_skill_balance.get_values(
                              self.skill_balance)
                + self.probability_creativity_match.get_values(
                              self.creativity_match)
                + self.probability_risk.get_values(self.risk)
                + self.probability_chemistry.get_values(self.chemistry)
            ) / 100
            project.success_probability = max(0, probability)

    def determine_success(self, project):

        success = Random.uniform() <= project.success_probability
        time = project.inventory.model.schedule.steps

        if success:
            if time in project.inventory.success_history.keys():
                project.inventory.success_history[time] += 1
            else:
                project.inventory.success_history[time] = 1
        else:
            if time in project.inventory.fail_history.keys():
                project.inventory.fail_history[time] += 1
            else:
                project.inventory.fail_history[time] = 1

        project.inventory.log_project_data_collector(
            project, null=False, success=success
        )

        return success

    def to_string(self, project):
        self.get_component_values(project)
        output = {
            'ovr (value, prob): ': (
                self.ovr,
                self.probability_ovr.get_values(self.ovr)
            ),
            'skill balance (value, prob): ': (
                self.skill_balance,
                self.probability_skill_balance.get_values(self.skill_balance)
            ),
            'creativity match (value, prob): ': (
                self.creativity_match,
                self.probability_creativity_match.get_values(
                    self.creativity_match)
            ),
            'risk (value, prob): ': (
                self.risk,
                self.probability_risk.get_values(self.risk)
            ),
            'chemistry (value, prob): ': (
                self.chemistry,
                self.probability_chemistry.get_values(self.chemistry)
            )
        }
        return json.dumps(output, indent=4)
