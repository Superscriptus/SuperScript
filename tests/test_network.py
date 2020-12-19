import networkx as nx
import unittest
from unittest.mock import patch

from superscript_model.network import SocialNetwork
from superscript_model.model import SuperScriptModel
from superscript_model.worker import Worker
from superscript_model.organisation import Team
from superscript_model.project import Project
from superscript_model.config import HISTORICAL_SUCCESS_RATIO_THRESHOLD


class TestProject(unittest.TestCase):

    @patch('superscript_model.organisation.TeamAllocator')
    def test_init(self, mock_allocator):

        model = SuperScriptModel(100)
        network = SocialNetwork(model, model.G)
        self.assertEqual(network.success_threshold,
                         HISTORICAL_SUCCESS_RATIO_THRESHOLD)
        self.assertIsInstance(network.G, nx.Graph)

    def test_initialise(self):

        model = SuperScriptModel(100)
        network = SocialNetwork(model, model.G)
        network.initialise()
        self.assertEqual(len(network.G.nodes()), 100)

    def test_replace_worker(self):

        model = SuperScriptModel(100)
        w = model.schedule.agents[0]
        self.assertTrue(w.worker_id in model.grid.G.nodes)
        self.assertFalse(101 in model.grid.G.nodes)
        w.replace()
        self.assertFalse(w.worker_id in model.grid.G.nodes)
        self.assertTrue(101 in model.grid.G.nodes)

    def test_add_team_edges(self):

        model = SuperScriptModel(10)

        model.inventory.create_projects(1, 0, 5)
        project = model.inventory.projects[0]
        team = Team(project,
                    members={w.worker_id: w
                             for w in model.schedule.agents[:3]},
                    lead=model.schedule.agents[0]
                    )
        model.grid.add_team_edges(team)
        self.assertTrue((0, 1) in model.grid.G.edges())
        self.assertTrue((0, 2) in model.grid.G.edges())
        self.assertTrue((2, 1) in model.grid.G.edges())
