from mesa import Agent
from interface import Interface


class WorkerStrategyInterface(Interface):

    def bid(self):
        pass

    def accept(self):
        pass


class Worker(Agent):

    def __init__(self, worker_id: int):

        self.worker_id = worker_id
