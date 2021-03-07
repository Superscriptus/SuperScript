import pickle
import networkx as nx
from itertools import combinations
from mesa.space import NetworkGrid

from .config import HISTORICAL_SUCCESS_RATIO_THRESHOLD


class SocialNetwork(NetworkGrid):

    def __init__(self, model, G,
                 success_threshold=HISTORICAL_SUCCESS_RATIO_THRESHOLD):

        self.model = model
        self.G = G
        self.success_threshold = success_threshold

    def initialise(self):

        for worker in self.model.schedule.agents:
            self.G.add_node(worker.worker_id,
                            department=worker.department.dept_id)
        super().__init__(self.G)

        for worker, node in zip(self.model.schedule.agents, self.G.nodes()):
            self.place_agent(worker, node)

    def replace_worker(self, old_worker, new_worker):

        self._remove_agent(old_worker, old_worker.worker_id)
        self.G = nx.relabel_nodes(
            self.G,
            {old_worker.worker_id: new_worker.worker_id},
            copy=False
        )
        self.place_agent(new_worker, new_worker.worker_id)

    # def remove_from_graph(self, worker):
    #     self._remove_agent(worker, worker.worker_id)
    #     self.G.remove_node(worker.worker_id)
    #
    # def add_to_graph(self, worker):
    #     self.G.add_node(worker.worker_id,
    #                     department=worker.department.dept_id)
    #     self.place_agent(worker, self.G.nodes(worker.worker_id))

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

    def save(self):
        nx.write_gpickle(
            self.G,
            self.model.io_dir
            + 'network_timestep_%d.gpickle' % self.model.time
        )
