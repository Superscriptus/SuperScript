import unittest
from unittest.mock import patch
from .test_worker import implements_interface

from mesa import Model
from mesa.time import RandomActivation
from superscript_model.model import SuperScriptModel
from superscript_model.worker import Worker
from superscript_model.project import ProjectInventory, Project
from superscript_model.organisation import (Team,
                                            OrganisationStrategyInterface,
                                            RandomStrategy,
                                            TeamAllocator,
                                            Department,
                                            Trainer)

from superscript_model.config import (DEPARTMENTAL_WORKLOAD,
                                      UNITS_PER_FTE,
                                      WORKLOAD_SATISFIED_TOLERANCE,
                                      HARD_SKILLS,
                                      TRAINING_COMMENCES,
                                      TRAINING_LENGTH,
                                      MAX_SKILL_LEVEL)


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

        team = Team(mock_project,
                    members={mock_worker.worker_id: mock_worker},
                    lead=mock_worker)
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

        self.assertEqual(round(team.compute_ovr(), 2), 72.36)

    @patch('superscript_model.model.Model')
    @patch('superscript_model.project.ProjectInventory')
    def test_compute_skill_balance(self, mock_inventory, mock_model):

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

        self.assertEqual(round(team.compute_skill_balance(), 2), 0.16)

    @patch('superscript_model.model.Model')
    @patch('superscript_model.project.ProjectInventory')
    def test_compute_creativity_match(self, mock_inventory, mock_model):

        w1 = Worker(1, mock_model)
        w2 = Worker(2, mock_model)
        w3 = Worker(1, mock_model)
        w4 = Worker(2, mock_model)
        w5 = Worker(1, mock_model)
        w6 = Worker(2, mock_model)
        w7 = Worker(2, mock_model)
        project = Project(mock_inventory,
                          project_id=42,
                          project_length=5)
        team = Team(project,
                    members={w1.worker_id: w1,
                             w2.worker_id: w2,
                             w3.worker_id: w3,
                             w4.worker_id: w4,
                             w5.worker_id: w5,
                             w6.worker_id: w6,
                             w7.worker_id: w7},
                    lead=w1)
        self.assertTrue(team.compute_creativity_match() <= 16)
        self.assertTrue(team.compute_creativity_match() >= 0)

        soft_skills = ['F', 'G','H', 'I', 'J']
        w1.skills.soft_skills = dict(zip(soft_skills,
                                         [1.0, 1.0, 1.0, 1.0, 1.0]))
        w2.skills.soft_skills = dict(zip(soft_skills,
                                         [5.0, 5.0, 5.0, 5.0, 5.0]))
        project = Project(mock_inventory,
                          project_id=42,
                          project_length=5)
        project.requirements.creativity = 5
        team = Team(project,
                    members={w1.worker_id: w1,
                             w2.worker_id: w2},
                    lead=w1)
        self.assertEqual(team.creativity_match, 0)

    @patch('superscript_model.model.Model')
    @patch('superscript_model.project.ProjectInventory')
    def test_log_project_outcome(self, mock_inventory, mock_model):

        w1 = Worker(1, mock_model)
        w2 = Worker(2, mock_model)
        project = Project(mock_inventory,
                          project_id=42,
                          project_length=5)
        team = Team(project,
                    members={w1.worker_id: w1,
                             w2.worker_id: w2},
                    lead=w1
                    )
        team.log_project_outcome(success=True)
        self.assertEqual(w1.history.get_success_rate(), 1)
        self.assertEqual(w2.history.get_success_rate(), 1)
        team.log_project_outcome(success=False)
        self.assertEqual(w1.history.get_success_rate(), 0.5)
        self.assertEqual(w2.history.get_success_rate(), 0.5)


class TestOrganisationStrategyInterface(unittest.TestCase):

    def test_interface(self):
        self.assertRaises(TypeError, lambda: OrganisationStrategyInterface())


class TestRandomStrategy(unittest.TestCase):

    def test_init(self):
        strategy = RandomStrategy(SuperScriptModel(100))
        self.assertIsInstance(strategy.model, Model)

    @patch('superscript_model.organisation.TeamAllocator')
    @patch('builtins.print')
    @patch('superscript_model.model.SuperScriptModel')
    def test_select_team(self, mock_model, mock_print, mock_allocator):

        strategy = RandomStrategy(SuperScriptModel(100))
        inventory = ProjectInventory(mock_allocator, model=mock_model)
        inventory.create_projects(1, time=0, length=5)
        team = strategy.select_team(inventory.projects[0],
                                    bid_pool=None)

        self.assertIsInstance(team, Team)
        self.assertTrue(set(team.members.values())
                        .issubset(strategy.model.schedule.agents))
        self.assertTrue(team.lead in team.members.values())

        team = strategy.select_team(inventory.projects[0],
                                    bid_pool=[])
        self.assertEqual(team.members, {})
        self.assertIs(team.lead, None)
        #self.assertEqual(mock_print.call_count, 2)

    @patch('superscript_model.project.Project')
    def test_invite_bids(self, mock_project):

        mock_project.start_time = 0
        mock_project.length = 5
        strategy = RandomStrategy(SuperScriptModel(worker_count=100,
                                                   department_count=10))
        bids = strategy.invite_bids(mock_project)
        self.assertEqual(len(bids), 100)


class TestTeamAllocator(unittest.TestCase):

    def test_init(self):
        allocator = TeamAllocator(SuperScriptModel(100))
        self.assertTrue(implements_interface(allocator.strategy,
                                             OrganisationStrategyInterface))

    @patch('superscript_model.project.Project')
    def test_allocate_team(self, mock_project):

        mock_project.start_time = 0
        mock_project.length = 5
        mock_project.budget = 1000
        allocator = TeamAllocator(SuperScriptModel(42))
        allocator.allocate_team(mock_project)
        self.assertIsInstance(mock_project.team, Team)


class TestDepartment(unittest.TestCase):

    def test_init(self):
        dept = Department(0)
        self.assertEqual(dept.workload, DEPARTMENTAL_WORKLOAD)
        self.assertEqual(dept.units_per_full_time, UNITS_PER_FTE)
        self.assertEqual(dept.tolerance, WORKLOAD_SATISFIED_TOLERANCE)

    def test_is_workload_satisfied(self):
        dept = Department(0)
        dept.number_of_workers += 1
        dept.units_supplied_to_projects[0] = 8

        self.assertFalse(dept.is_workload_satisfied(0, 1))

    def test_to_string(self):
        dept = Department(0)
        self.assertIsInstance(dept.to_string(), str)

    @patch('superscript_model.model.Model')
    def test_add_training(self, mock_model):

        mock_model.trainer = Trainer(mock_model)
        dept = Department(0)
        worker = Worker(42, mock_model, department=dept)
        dept.add_training(worker, 5)
        #worker.model.trainer.train(worker)


class TestTrainer(unittest.TestCase):

    @patch('superscript_model.model.Model')
    def test_init(self, mock_model):
        trainer = Trainer(mock_model)
        self.assertEqual(trainer.hard_skills, HARD_SKILLS)
        self.assertEqual(trainer.max_skill_level, MAX_SKILL_LEVEL)
        self.assertEqual(trainer.training_length, TRAINING_LENGTH)
        self.assertIsInstance(trainer.skill_quartiles, dict)

    @patch('superscript_model.model.Model')
    def test_update_skill_quartiles(self, mock_model):
        mock_model.schedule = RandomActivation(mock_model)
        trainer = Trainer(mock_model)
        dept = Department(0)
        workers = []
        for i in range(5):
            w = Worker(i, mock_model, department=dept)
            w.skills.hard_skills = dict(
                zip(HARD_SKILLS, [i+1 for s in HARD_SKILLS])
            )
            mock_model.schedule.add(w)
        trainer.update_skill_quartiles()
        for skill in HARD_SKILLS:
            self.assertEqual(list(trainer.skill_quartiles[skill]),
                             [2, 3, 4])

    @patch('superscript_model.model.Model')
    @patch('superscript_model.organisation.TeamAllocator')
    def test_train(self, mock_allocator, mock_model):

        mock_model.schedule = RandomActivation(mock_model)
        mock_model.inventory = ProjectInventory(mock_allocator,
                                                model=mock_model)
        mock_model.inventory.total_skill_requirement = {
            'A': 10, 'B': 9
        }

        trainer = Trainer(mock_model)
        dept = Department(0)
        workers = []
        for i in range(5):
            w = Worker(i, mock_model, department=dept)
            w.skills.hard_skills = dict(
                zip(HARD_SKILLS, [i+1 for s in HARD_SKILLS])
            )
            workers.append(w)
            mock_model.schedule.add(w)

        mock_model.training_mode = 'all'

        trainer.update_skill_quartiles()
        mock_model.training_commences = 0
        for i in range(5):
            trainer.train()

        self.assertEqual(workers[0].skills.hard_skills['A'], 4)
        self.assertEqual(workers[0].skills.hard_skills['B'], 4)
        self.assertEqual(workers[0].skills.hard_skills['C'], 1)
        self.assertEqual(workers[1].skills.hard_skills['B'], 4)
        self.assertEqual(workers[2].skills.hard_skills['A'], 3)
        self.assertEqual(workers[4].skills.hard_skills['B'], 5)
