from interface import Interface, implements
import json

from .project import Project
from .utilities import Random
from .config import (TEAM_OVR_MULTIPLIER,
                     MIN_TEAM_SIZE,
                     MAX_TEAM_SIZE,
                     PRINT_DECIMALS_TO)


class Team:

    def __init__(self, project, members,
                 lead, round_to=PRINT_DECIMALS_TO):
        self.project = project
        self.members = members
        self.lead = lead
        self.assign_lead(self.project)
        self.round_to = round_to
        # currently this is automatic, but could be handled by TeamAllocator:
        self.contributions = self.determine_member_contributions()
        self.team_ovr = self.compute_ovr()
        self.skill_balance = self.compute_skill_balance()

    def assign_lead(self, project):
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

        for skill in contributions.keys():
            for member_id in contributions[skill]:
                self.members[member_id].add_contribution(
                    self.project, skill
                )

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
            supplied_units = sum([s for s in worker_skills if s >= required_level])

            if supplied_units < required_units:
                skill_balance += (supplied_units - required_units) ** 2
                number_with_negative_differences += 1

        if number_with_negative_differences > 0:
            return skill_balance / number_with_negative_differences
        else:
            return 0

    def to_string(self):

        output = {
            'project': self.project.project_id,
            'members': list(self.members.keys()),
            'lead': self.lead.worker_id,
            'team_ovr': round(self.team_ovr, self.round_to),
            'skill_balance': round(self.skill_balance, self.round_to),
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
            if worker.strategy.bid(project)
        ]
        return bid_pool

    def select_team(self, project: Project,
                    bid_pool=None) -> Team:

        size = Random.randint(self.min_team_size,
                              self.max_team_size)
        bid_pool = (self.model.schedule.agents
                    if bid_pool is None else bid_pool)

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
