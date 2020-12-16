from interface import Interface, implements
from itertools import combinations
import json

from .project import Project
from .utilities import Random
from .config import (TEAM_OVR_MULTIPLIER,
                     MIN_TEAM_SIZE,
                     MAX_TEAM_SIZE,
                     PRINT_DECIMALS_TO,
                     MAX_SKILL_LEVEL,
                     MIN_SOFT_SKILL_LEVEL,
                     SOFT_SKILLS,
                     DEPARTMENTAL_WORKLOAD,
                     WORKLOAD_SATISFIED_TOLERANCE,
                     UNITS_PER_FTE)


class Team:

    def __init__(self, project, members,
                 lead, round_to=PRINT_DECIMALS_TO,
                 soft_skills=SOFT_SKILLS):
        self.project = project
        self.members = members
        self.lead = lead
        self.assign_lead(self.project)
        self.round_to = round_to
        self.soft_skills = soft_skills  # used by compute_creativity_match()
        # currently this is automatic, but could be handled by TeamAllocator:
        self.contributions = self.determine_member_contributions()
        self.team_ovr = self.compute_ovr()
        self.skill_balance = self.compute_skill_balance()
        self.creativity_match = self.compute_creativity_match()

    def assign_lead(self, project):
        if self.lead is not None:
            self.lead.assign_as_lead(project)

    def remove_lead(self, project):
        self.lead.remove_as_lead(project)
        self.lead = None

    def compute_ovr(self, multiplier=TEAM_OVR_MULTIPLIER):

        skill_count = 0
        ovr = 0
        for skill in self.project.required_skills:

            workers = self.contributions[skill]
            for worker_id in workers:
                ovr += self.members[worker_id].get_skill(skill)
                skill_count += 1

        if skill_count > 0:
            return multiplier * ovr / skill_count
        else:
            return 0.0

    def rank_members_by_skill(self, skill):

        ranked_members = {
            member[0]: member[1].get_skill(skill)
            for member in self.members.items()
        }
        return {
            k: v for k, v in sorted(ranked_members.items(),
                                    reverse=True,
                                    key=lambda item: item[1])
        }

    def determine_member_contributions(self):

        contributions = dict()
        for skill in self.project.required_skills:

            ranked_members = self.rank_members_by_skill(skill)
            skill_requirement = self.project.get_skill_requirement(skill)

            contributions[skill] = [
                m[0] for m in ranked_members.items()
            ][:skill_requirement['units']]

        self.assign_contributions_to_members(contributions)

        return contributions

    def assign_contributions_to_members(self, contributions):

        # for skill in contributions.keys():
        #     for member_id in contributions[skill]:
        #         self.members[member_id].add_contribution(
        #             self.project, skill
        #         )
        for member_id in self.members.keys():

            units_contributed_by_member = 0
            for skill in contributions.keys():

                if member_id in contributions[skill]:
                    self.members[member_id].add_contribution(
                        self.project, skill
                    )
                    units_contributed_by_member += 1

            (self.members[member_id]
             .department.update_supplied_units(
                member_id, units_contributed_by_member, self.project
            ))

    def compute_skill_balance(self):

        skill_balance = 0
        number_with_negative_differences = 0
        for skill in self.project.required_skills:

            required_units = (
                self.project.get_skill_requirement(skill)['units']
            )
            required_level = (
                self.project.get_skill_requirement(skill)['level']
            )
            worker_skills = [
                self.members[worker_id].get_skill(skill)
                for worker_id in self.contributions[skill]
            ]
            skill_mismatch = (
                (sum(worker_skills) / required_units) - required_level
            ) if required_units > 0 else 0

            if skill_mismatch < 0:
                skill_balance += skill_mismatch ** 2
                number_with_negative_differences += 1

        if number_with_negative_differences > 0:
            return skill_balance / number_with_negative_differences
        else:
            return 0

    def compute_creativity_match(self,
                                 max_skill_level=MAX_SKILL_LEVEL,
                                 min_skill_level=MIN_SOFT_SKILL_LEVEL):

        creativity_level = 0
        number_of_existing_skills = 0
        max_distance = max_skill_level - min_skill_level
        if len(self.members.keys()) > 1:
            max_distance /= (len(self.members.keys()) - 1)

        for skill in self.soft_skills:

            worker_skills = [
                member.get_skill(
                    skill, hard_skill=False
                )
                for member in self.members.values()
            ]

            pairs = list(combinations(worker_skills, 2))
            if len(pairs) > 0:
                creativity_level += (
                    sum([((p[1] - p[0]) / max_distance) ** 2
                         for p in pairs])
                    / len(pairs)
                )
                number_of_existing_skills += 1

        if number_of_existing_skills > 0:
            creativity_level /= number_of_existing_skills
        else:
            creativity_level = 0

        creativity_level = (creativity_level * max_distance) + 1
        return (self.project.creativity - creativity_level) ** 2

    def to_string(self):

        output = {
            'project': self.project.project_id,
            'members': list(self.members.keys()),
            'lead': self.lead.worker_id,
            'success_probability': round(
                self.project.success_probability, self.round_to
            ),
            'team_ovr': round(self.team_ovr, self.round_to),
            'skill_balance': round(self.skill_balance, self.round_to),
            'creativity_match': round(
                self.creativity_match, self.round_to
            ),
            'skill_contributions': self.contributions
        }
        return json.dumps(output, indent=4)


class OrganisationStrategyInterface(Interface):

    def invite_bids(self, project:Project) -> list:
        pass

    def select_team(self, project: Project,
                    bid_pool=None) -> Team:
        pass


class RandomStrategy(implements(OrganisationStrategyInterface)):

    def __init__(self, model,
                 min_team_size=MIN_TEAM_SIZE,
                 max_team_size=MAX_TEAM_SIZE):
        self.model = model
        self.min_team_size = min_team_size
        self.max_team_size = max_team_size

    def invite_bids(self, project: Project) -> list:

        bid_pool = [
            worker for worker in self.model.schedule.agents
            if worker.bid(project)
        ]
        return bid_pool

    def select_team(self, project: Project,
                    bid_pool=None) -> Team:

        size = Random.randint(self.min_team_size,
                              self.max_team_size)
        bid_pool = (self.model.schedule.agents
                    if bid_pool is None else bid_pool)

# Check this functionality...
        if size > len(bid_pool):
            print("Cannot select %d workers from bid_pool of size %d"
                  % (size, len(bid_pool)))
            workers = {}
            lead = None
        else:
            workers = {worker.worker_id: worker
                       for worker in
                       Random.choices(bid_pool, size)}
            lead = Random.choice(list(workers.values()))

        return Team(project, workers, lead)


class TeamAllocator:

    def __init__(self, model):
        self.model = model
        self.strategy = RandomStrategy(model)

    def allocate_team(self, project: Project):

        bid_pool = self.strategy.invite_bids(project)
        project.team = self.strategy.select_team(
            project, bid_pool=bid_pool
        )


class Department:

    def __init__(self, dept_id, workload=DEPARTMENTAL_WORKLOAD,
                 units_per_full_time=UNITS_PER_FTE,
                 tolerance=WORKLOAD_SATISFIED_TOLERANCE):

        self.dept_id = dept_id
        self.number_of_workers = 0
        self.workload = workload
        self.units_per_full_time = units_per_full_time
        self.tolerance = tolerance
        self.units_supplied_to_projects = dict()

    def update_supplied_units(self, worker_id,
                              units_contributed, project):

        for time_offset in range(project.length):
            time = project.start_time + time_offset

            if time not in self.units_supplied_to_projects.keys():
                self.units_supplied_to_projects[time] = dict()

            self.add_worker_units(worker_id, units_contributed, time)

    def add_worker_units(self, worker_id, units, time):

        if worker_id not in self.units_supplied_to_projects[time].keys():
            self.units_supplied_to_projects[time][worker_id] = units
        else:
            self.units_supplied_to_projects[time][worker_id] += units

    def is_workload_satisfied(self, start, length):

        total_units_dept_can_supply = (
                self.number_of_workers * self.units_per_full_time
        )
        departmental_workload_units = (
            total_units_dept_can_supply * self.workload
        )

        for t in range(length):

            time = start + t
            total_supplied_units = sum(
                self.units_supplied_to_projects[time].values()
            ) if time in self.units_supplied_to_projects else 0

            if (total_supplied_units
                    >= (total_units_dept_can_supply
                        - departmental_workload_units
                        - self.tolerance)):
                return False

        return True

    def to_string(self):
        output = {
            'dept_id': self.dept_id,
            'number_of_workers': self.number_of_workers,
            'workload': self.workload,
            'units_per_full_time': self.units_per_full_time,
            'tolerance': self.tolerance
        }
        return json.dumps(output, indent=4)

