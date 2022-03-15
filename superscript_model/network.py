"""
SuperScript network module
===========

Classes:
    SocialNetwork
        Class for tracking social network of collaborations
"""

import networkx as nx
from itertools import combinations
from mesa.space import NetworkGrid

from .config import HISTORICAL_SUCCESS_RATIO_THRESHOLD


class SocialNetwork(NetworkGrid):
    """Social network class.

    Is a subclass of mesa.space.NetworkGrid

    Edges in this network count the number of successful collaborations
    between each pair of workers.
    ...

    Attributes:
        model: model.SuperScriptModel
            Reference to main model.
        G: nx.Graph
            Networkx x graph object, created in main model.
        success_threshold: float
            Fraction of team members who need to have worked before on
            successful project for get_team_historical_success_flag to
            return True.
    """

    def __init__(self, model, G,
                 success_threshold=HISTORICAL_SUCCESS_RATIO_THRESHOLD):

        self.model = model
        self.G = G
        self.success_threshold = success_threshold

    def initialise(self):
        """Create the network.

        Adds all workers as nodes in the network.
        Calls base class constructor and calls base class method
        to place each worker on their node.
        """

        for worker in self.model.schedule.agents:
            self.G.add_node(
                worker.worker_id,
                department=worker.department.dept_id
            )
        super().__init__(self.G)

        for worker, node in zip(self.model.schedule.agents, self.G.nodes()):
            self.place_agent(worker, node)

    def replace_worker(self, old_worker, new_worker):
        """Replaces worker in the network (called when a worker is
        replaced due to inactivity.

        Args:
            old_worker: worker.Worker
                Worker being replace.
            new_worker: worker.Worker
                Worker replacing old worker.
        """

        self._remove_agent(old_worker, old_worker.worker_id)
        self.G = nx.relabel_nodes(
            self.G,
            {old_worker.worker_id: new_worker.worker_id},
            copy=False
        )
        self.place_agent(new_worker, new_worker.worker_id)

    def add_team_edges(self, team):
        """Adds all edges between pairs of workers in this team.

        If an edge already exists, its weight is incremented by 1.

        Args:
            team: organisation.Team
                Provides list of members that have successfully
                collaborated on a project and are to be added
                to the network.
        """

        pairs = list(combinations(team.members.keys(), 2))
        for pair in pairs:

            if (pair[0], pair[1]) not in self.G.edges():
                self.G.add_edge(pair[0], pair[1], weight=1)
            else:
                self.G[pair[0]][pair[1]]['weight'] += 1

    def get_team_historical_success_flag(self, team):
        """Determine is team meets required threshold for previous
        successful collaboration.

        Args:
            team: organisation.Team

        Returns:
            bool: True if more than self.success_threshold of possible
                  edges are realised in this network.
        """

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

    def track(self):
        """
        Save network for later analysis.

        Note: this updated method uses a more concise format to save space on
        disk and reduce loading times in the Streamlit application, such that
        the network state on each timestep can be represented. Here, the
        initial network is stored at the end of the first timetsep, and a
        then network diff is stored in a json dictionary for each subsequent
        timestep and saved to disk at the end of the simulation.
        """

        if self.model.save_network_flag:
            if self.model.time == 0:

                nx.write_multiline_adjlist(
                    self.G,
                    self.model.io_dir
                    + '/network_timestep_%d.gpickle' % self.model.time
                )
            else:
                # store network diff...(add old net)
                # save at end of simulation
                pass
