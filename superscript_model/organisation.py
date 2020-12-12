from interface import Interface, implements

from .project import Project
from .utilities import Random


class Team:

    def __init__(self, project, members, lead):
        self.project = project
        self.members = members
        self.lead = lead
        self.assign_lead(self.project)
        # do we need to store contributions here?
        # currently this is automatic, but could be handled by TeamAllocator
        self.contributions = self.determine_member_contributions()
        self.team_ovr = self.compute_ovr()

    def assign_lead(self, project):
        self.lead.assign_as_lead(project)

    def remove_lead(self, project):
        self.lead.remove_as_lead(project)
        self.lead = None

    def compute_ovr(self):
        self.project.required_skills
        return None

    def rank_members_by_skill(self, skill):

        ranked_members = {
            member[0]: member[1].get_skill(skill)
            for member in self.members.items()
        }
        return {
            k: v for k, v in sorted(ranked_members.items(),
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


class OrganisationStrategyInterface(Interface):

    def invite_bids(self, project:Project) -> list:
        pass

    def select_team(self, project: Project,
                    bid_pool=None) -> Team:
        pass


class RandomStrategy(implements(OrganisationStrategyInterface)):

    def __init__(self, model):
        self.model = model

    def invite_bids(self, project:Project) -> list:

        bid_pool = [
            worker for worker in self.model.schedule.agents
            if worker.strategy.bid(project)
        ]
        return bid_pool

    def select_team(self, project: Project,
                    bid_pool=None) -> Team:

        size = Random.randint(3, 7)
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
