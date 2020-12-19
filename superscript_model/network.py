import networkx as nx
from itertools import combinations
import matplotlib.pyplot as plt

from mesa.space import NetworkGrid


class InteractionNetwork(NetworkGrid):

    def __init__(self, model, G):

        self.model = model
        self.G = G #nx.Graph()
        for worker in self.model.schedule.agents:
            self.G.add_node(worker.worker_id,
                            department=worker.department.dept_id)

        super().__init__(self.G)
        for worker, node in zip(self.model.schedule.agents, self.G.nodes()):
            self.place_agent(worker, node)

        #for worker in self.model.schedule.agents:
        #    self.remove_from_graph(worker)

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
                print("edge not found:", pair)
                self.G.add_edge(pair[0], pair[1], weight=1)
            else:
                print('updating edge weight:', pair)
                self.G[pair[0]][pair[1]]['weight'] += 1

        # nx.draw(self.G)
        # plt.show()