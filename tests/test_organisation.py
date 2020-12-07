import unittest
from unittest.mock import patch
from .test_worker import implements_interface

from mesa import Model
from superscript_model.model import SuperScriptModel
from superscript_model.project import ProjectInventory, Project
from superscript_model.organisation import (Team,
                                            OrganisationStrategyInterface,
                                            RandomStrategy,
                                            TeamAllocator)


class TestTeam(unittest.TestCase):

    @patch('superscript_model.project.Project')
    @patch('superscript_model.worker.Worker')
    def test_init(self, mock_worker, mock_project):

        team = Team(mock_project, members=[mock_worker],lead=mock_worker)
        self.assertTrue(team.team_ovr is None)
        self.assertEqual(len(team.members), 1)
        self.assertEqual(mock_worker.assign_as_lead.call_count, 1)

    @patch('superscript_model.project.Project')
    @patch('superscript_model.worker.Worker')
    def test_remove_lead(self, mock_worker, mock_project):

        team = Team(mock_project, members=[mock_worker], lead=mock_worker)
        team.remove_lead(mock_project)
        self.assertEqual(mock_worker.remove_as_lead.call_count, 1)
        self.assertIs(team.lead, None)


class TestOrganisationStrategyInterface(unittest.TestCase):

    def test_interface(self):
        self.assertRaises(TypeError, lambda: OrganisationStrategyInterface())


class TestRandomStrategy(unittest.TestCase):

    def test_init(self):
        strategy = RandomStrategy(SuperScriptModel(42))
        self.assertIsInstance(strategy.model, Model)

    def test_select_team(self):

        strategy = RandomStrategy(SuperScriptModel(42))
        inventory = ProjectInventory()
        inventory.create_projects(1)
        team = strategy.select_team(inventory.projects[0],
                                    bid_pool=None)

        self.assertIsInstance(team, Team)
        self.assertTrue(set(team.members).issubset(strategy.model.schedule.agents))
        self.assertTrue(team.lead in team.members)


class TestTeamAllocator(unittest.TestCase):

    def test_init(self):
        allocator = TeamAllocator(SuperScriptModel(42))
        self.assertTrue(implements_interface(allocator.strategy,
                                             OrganisationStrategyInterface))

    @patch('superscript_model.project.Project')
    def test_allocate_team(self, mock_project):

        allocator = TeamAllocator(SuperScriptModel(42))
        allocator.allocate_team(mock_project)
        self.assertIsInstance(mock_project.team, Team)


