from mesa import Agent
from interface import Interface, implements


class WorkerStrategyInterface(Interface):

    def bid(self):
        pass

    def accept(self):
        pass


class AllInStrategy(implements(WorkerStrategyInterface)):

    def __init__(self, name: str):
        self.name = name

    def bid(self):
        pass

    def accept(self):
        pass


class Worker(Agent):

    def __init__(self, worker_id: int):

        self.worker_id = worker_id
