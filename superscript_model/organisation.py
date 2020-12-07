from interface import Interface, implements

from .project import Project
from .utilities import Random


class Team:

    def __init__(self, project, members, lead):
        self.team_ovr = None
        self.members = members
        self.lead = lead
        self.assign_lead(project)

    def assign_lead(self, project):
        self.lead.assign_as_lead(project)

    def remove_lead(self, project):
        self.lead.remove_as_lead(project)


class OrganisationStrategyInterface(Interface):

    def select_team(self, project: Project,
                      bid_pool=None) -> Team:
        pass


class RandomStrategy(implements(OrganisationStrategyInterface)):

    def __init__(self, model):
        self.model = model

    def select_team(self, project: Project,
                    bid_pool=None) -> Team:

        size = Random.randint(3, 7)
        bid_pool = (self.model.schedule.agents
                    if bid_pool is None else bid_pool)

        workers = Random.choices(bid_pool, size)
        lead = Random.choice(workers)

        return Team(project, workers, lead)


class TeamAllocator:

    def __init__(self, model):
        self.model = model
        self.strategy = RandomStrategy(model)

    def invite_bids(self):
        # NOT IMPLEMENTED
        pass

    def allocate_team(self, project: Project):

        project.team = self.strategy.select_team(project,
                                                 bid_pool=None)
