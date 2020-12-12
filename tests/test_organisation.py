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

    @patch('superscript_model.model.Model')
    @patch('superscript_model.project.ProjectInventory')
    def test_to_string(self, mock_inventory, mock_model):

        worker = Worker(1, mock_model)
        project = Project(mock_inventory,
                          project_id=42,
                          project_length=5)

        team = Team(project,
                    members={worker.worker_id: worker},
                    lead=worker)

        self.assertIsInstance(team.to_string(), str)

    @patch('superscript_model.model.Model')
    @patch('superscript_model.project.ProjectInventory')
    def test_compute_ovr(self, mock_inventory, mock_model):

        workers = [Worker(i, mock_model) for i in range(5)]
        hard_skills = ['A', 'B','C', 'D', 'E']
        workers[0].skills.hard_skills = dict(zip(hard_skills,
                                                 [0.0, 3.9, 3.2, 4.1, 1.5]))
        workers[1].skills.hard_skills = dict(zip(hard_skills,
                                                 [4.1, 4.4, 2.1, 2.9, 0.4]))
        workers[2].skills.hard_skills = dict(zip(hard_skills,
                                                 [0.0, 0.0, 0.0, 0.5, 3.9]))
        workers[3].skills.hard_skills = dict(zip(hard_skills,
                                                 [0.0, 3.6, 2.5, 4.9, 5.0]))
        workers[4].skills.hard_skills = dict(zip(hard_skills,
                                                 [0.0, 0.0, 0.0, 2.9, 0.0]))
        project = Project(mock_inventory,
                          project_id=42,
                          project_length=5)
        project.requirements.hard_skills = {
            'A': {'units': 0, 'level': 3},
            'B': {'units': 2, 'level': 3},
            'C': {'units': 3, 'level': 3},
            'D': {'units': 4, 'level': 3},
            'E': {'units': 2, 'level': 3}
        }
        team = Team(project,
                    members={worker.worker_id: worker
                             for worker in workers},
                    lead=workers[0])

        self.assertEqual(round(team.compute_ovr(),2), 72.36)


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


