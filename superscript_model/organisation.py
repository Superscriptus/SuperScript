from interface import Interface, implements

from .project import Project
from .utilities import Random


class Team:

    def __init__(self, project, members, lead):
        self.project = project
        self.members = members
        self.lead = lead
        self.assign_lead(self.project)

        # update all unit tests
        self.contributions = self.determine_contributions() # do we need to store contributions here?
        #check if team is viable: can it meet the project requirements? If not? (handle in TeamAllocator)
        # Introduce time tracking...(count steps())
        self.team_ovr = self.compute_ovr()
        #self.set_contributions() #Log contributions (and when they are required) in worker
                                  # so they can bid based on their availability

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

    def determine_contributions(self):

        contributions = dict()
        for skill in self.project.required_skills:

            ranked_members = self.rank_members_by_skill(skill)
            skill_requirement = self.project.get_skill_requirement(skill)

            top_members = list(
                ranked_members.items()
            )[:skill_requirement['units']]

            contributions[skill] = [
                m[0] for m in top_members
                if m[1] >= skill_requirement['level']
            ]
        return contributions


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

        workers = {worker.worker_id: worker
                   for worker in
                   Random.choices(bid_pool, size)}
        lead = Random.choice(list(workers.values()))

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
