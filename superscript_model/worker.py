from mesa import Agent
from interface import Interface, implements

from .project import Project


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
        super().__init__(worker_id, model)
        self.strategy = AllInStrategy('All-In')
        self.leads_on = dict()

    def assign_as_lead(self, project):
        self.leads_on[project.project_id] = project

    def remove_as_lead(self, project):
        del self.leads_on[project.project_id]

    def step(self):
        projects = list(self.leads_on.values())

        for project in projects:
            if project in self.leads_on.values():
                project.advance()
