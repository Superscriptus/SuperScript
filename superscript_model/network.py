from itertools import combinations
from mesa.space import NetworkGrid

from .config import HISTORICAL_SUCCESS_RATIO_THRESHOLD


class SocialNetwork(NetworkGrid):

    def __init__(self, model, G,
                 success_threshold=HISTORICAL_SUCCESS_RATIO_THRESHOLD):

        self.model = model
        self.G = G
        self.success_threshold = success_threshold

        for worker in self.model.schedule.agents:
            self.G.add_node(worker.worker_id,
                            department=worker.department.dept_id)

        super().__init__(self.G)
        for worker, node in zip(self.model.schedule.agents, self.G.nodes()):
            self.place_agent(worker, node)

    def remove_from_graph(self, worker):
        self._remove_agent(worker, worker.worker_id)
        self.G.remove_node(worker.worker_id)

    def add_to_graph(self, worker):
        self.G.add_node(worker.worker_id)
        self.place_agent(worker, worker.worker_id)

    def add_team_edges(self, team):

        pairs = list(combinations(team.members.keys(), 2))
        for pair in pairs:

            if (pair[0], pair[1]) not in self.G.edges():
                self.G.add_edge(pair[0], pair[1], weight=1)
            else:
                self.G[pair[0]][pair[1]]['weight'] += 1

    def get_team_historical_success_flag(self, team):

        pairs = list(combinations(team.members.keys(), 2))
        success_ratio = 0
        for pair in pairs:
            if (pair[0], pair[1]) in self.G.edges():
                success_ratio += 1

        success_ratio = (
            success_ratio / len(pairs) if len(pairs) > 0 else 0.0
        )
        if success_ratio >= self.success_threshold:
            return True
        else:
            return False

