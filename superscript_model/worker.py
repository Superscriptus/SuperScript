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

    def __init__(self, worker_id: int):

        self.worker_id = worker_id
