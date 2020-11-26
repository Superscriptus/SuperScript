from mesa import Agent


class Worker(Agent):

    def __init__(self, worker_id: int):

        self.worker_id = worker_id
