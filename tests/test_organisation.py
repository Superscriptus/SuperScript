import unittest

from mesa import Model
from superscript_model.model import SuperScriptModel
from superscript_model.project import ProjectInventory, Project
from superscript_model.organisation import (Team,
                                            OrganisationStrategyInterface,
                                            RandomStrategy)


class TestTeam(unittest.TestCase):

    def test_init(self):

        team = Team(members=[],lead=None)
        self.assertTrue(team.team_ovr is None)
        self.assertEqual(len(team.members), 0)


class TestOrganisationStrategyInterface(unittest.TestCase):

    def test_interface(self):
        self.assertRaises(TypeError, lambda: OrganisationStrategyInterface())


class TestRandomStrategy(unittest.TestCase):

    def test_init(self):
        strategy = RandomStrategy(SuperScriptModel(42))
        self.assertIsInstance(strategy.model, Model)

    def test_allocate_team(self):

        strategy = RandomStrategy(SuperScriptModel(42))
        inventory = ProjectInventory()
        inventory.create_projects(1)
        team = strategy.allocate_team(inventory.projects[0],
                                      bid_pool=None)

        self.assertIsInstance(team, Team)
        self.assertTrue(set(team.members).issubset(strategy.model.schedule.agents))
        self.assertTrue(team.lead in team.members)
