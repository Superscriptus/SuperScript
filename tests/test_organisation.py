import unittest
from unittest.mock import patch
from .test_worker import implements_interface

from mesa import Model
from superscript_model.model import SuperScriptModel
from superscript_model.worker import Worker
from superscript_model.project import ProjectInventory, Project
from superscript_model.organisation import (Team,
                                            OrganisationStrategyInterface,
                                            RandomStrategy,
                                            TeamAllocator)


class TestTeam(unittest.TestCase):

    @patch('superscript_model.project.Project')
    @patch('superscript_model.worker.Worker')
    def test_init(self, mock_worker, mock_project):

        team = Team(mock_project,
                    members={mock_worker.worker_id: mock_worker},
                    lead=mock_worker)
        self.assertEqual(team.team_ovr, 0.0)
        self.assertEqual(len(team.members), 1)
        self.assertIsInstance(team.members, dict)
        self.assertEqual(mock_worker.assign_as_lead.call_count, 1)

    @patch('superscript_model.project.Project')
    @patch('superscript_model.worker.Worker')
    def test_remove_lead(self, mock_worker, mock_project):

        team = Team(mock_project, members=[mock_worker], lead=mock_worker)
        team.remove_lead(mock_project)
        self.assertEqual(mock_worker.remove_as_lead.call_count, 1)
        self.assertIs(team.lead, None)

    @patch('superscript_model.model.Model')
    @patch('superscript_model.project.ProjectInventory')
    def test_assign_contributions_to_members(self,
                                             mock_inventory,
                                             mock_model):

        pid = 42
        p_len = 5
        p_start = 0
        worker = Worker(1, mock_model)
        project = Project(mock_inventory,
                          project_id=pid,
                          project_length=p_len,
                          start_time=p_start)
        team = Team(project,
                    members={worker.worker_id: worker},
                    lead=worker)

        self.assertEqual(len(worker.contributes.keys()), p_len)
        for skill in project.required_skills:
            self.assertEqual(
                worker.contributes[project.start_time][skill], [pid]
            )


class TestOrganisationStrategyInterface(unittest.TestCase):

    def test_interface(self):
        self.assertRaises(TypeError, lambda: OrganisationStrategyInterface())


class TestRandomStrategy(unittest.TestCase):

    def test_init(self):
        strategy = RandomStrategy(SuperScriptModel(42))
        self.assertIsInstance(strategy.model, Model)

    @patch('superscript_model.organisation.TeamAllocator')
    def test_select_team(self, mock_allocator):

        strategy = RandomStrategy(SuperScriptModel(42))
        inventory = ProjectInventory(mock_allocator)
        inventory.create_projects(1, time=0)
        team = strategy.select_team(inventory.projects[0],
                                    bid_pool=None)

        self.assertIsInstance(team, Team)
        self.assertTrue(set(team.members.values())
                        .issubset(strategy.model.schedule.agents))
        self.assertTrue(team.lead in team.members.values())

    @patch('superscript_model.project.Project')
    def test_invite_bids(self, mock_project):

        strategy = RandomStrategy(SuperScriptModel(42))
        bids = strategy.invite_bids(mock_project)
        self.assertEqual(len(bids), 42)



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


