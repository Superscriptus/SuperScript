import unittest
from unittest.mock import patch
from .test_worker import implements_interface

from mesa import Model
from mesa.time import RandomActivation
from superscript_model.model import SuperScriptModel
from superscript_model.worker import Worker, SkillMatrix
from superscript_model.optimisation import OptimiserFactory
from superscript_model.project import ProjectInventory, Project
from superscript_model.organisation import (Team,
                                            OrganisationStrategyInterface,
                                            RandomStrategy,
                                            BasicStrategy,
                                            ParallelBasinhopping,
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

    def setUp(self):

        self.model = SuperScriptModel(
            worker_count=100,
            worker_strategy='AllIn',
            organisation_strategy='Random',
            budget_functionality_flag=False
        )

        self.project = Project(
            self.model.inventory,
            project_id=0,
            project_length=5,
            start_time=0)

        self.allocator = TeamAllocator(
            self.model,
            OptimiserFactory()
        )

    def test_init(self):

        workers = list(self.model.schedule.agents)[0:4]
        team = Team(
            self.project,
            members={
                agent.worker_id: agent
                for agent in workers
            },
            lead=workers[0]
        )

        self.assertEqual(len(team.members), 4)
        self.assertIsInstance(team.members, dict)
        self.assertIsNotNone(team.lead)

        team = Team(
            self.project,
            members={
                agent.worker_id: agent
                for agent in workers
            },
            lead=workers[0]
        )
        for skill in ['A', 'B', 'C', 'D', 'E']:
            team.contributions[skill] = []

        self.assertEqual(len(team.members), 4)
        self.assertIsInstance(team.members, dict)
        self.assertIsNotNone(team.lead)

    def test_remove_lead(self):

        self.allocator.allocate_team(self.project)
        team = self.project.team
        self.assertIsNotNone(team.lead)
        team.remove_lead()
        self.assertIsNone(team.lead)

    @patch('superscript_model.model.Model')
    @patch('superscript_model.organisation.TeamAllocator')
    def test_assign_contributions_to_members(self,
                                             mock_allocator,
                                             mock_model):

        pid = 42
        p_len = 5
        p_start = 0
        worker = Worker(1, mock_model)
        mock_model.p_budget_flexibility = 0.25
        project = Project(ProjectInventory(mock_allocator,
                                           model=mock_model),
                          project_id=pid,
                          project_length=p_len,
                          start_time=p_start)
        team = Team(project,
                    members={worker.worker_id: worker},
                    lead=worker)

        team.assign_contributions_to_members()
        self.assertEqual(len(worker.contributes.keys()), p_len)
        for skill in project.required_skills:
            self.assertEqual(
                worker.contributes[project.start_time][skill], [pid]
            )

    @patch('superscript_model.model.Model')
    @patch('superscript_model.organisation.TeamAllocator')
    def test_to_string(self, mock_allocator, mock_model):
        mock_model.p_budget_flexibility = 0.25
        worker = Worker(1, mock_model)
        project = Project(ProjectInventory(mock_allocator,
                                           model=mock_model),
                          project_id=42,
                          project_length=5)

        team = Team(project,
                    members={worker.worker_id: worker},
                    lead=worker)

        self.assertIsInstance(team.to_string(), str)

    @patch('superscript_model.model.Model')
    @patch('superscript_model.organisation.TeamAllocator')
    def test_compute_ovr(self, mock_allocator, mock_model):

        dept = Department(0, mock_model)
        workers = [Worker(i, mock_model, dept) for i in range(5)]
        hard_skills = ['A', 'B', 'C', 'D', 'E']
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
        mock_model.p_budget_flexibility = 0.25
        project = Project(
            ProjectInventory(
                mock_allocator,
                model=mock_model
            ),
            project_id=42,
            project_length=5
        )
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

        team.project.requirements.hard_skills = {
            'A': {'units': 0, 'level': 3},
            'B': {'units': 0, 'level': 3},
            'C': {'units': 0, 'level': 3},
            'D': {'units': 0, 'level': 3},
            'E': {'units': 0, 'level': 3}
        }
        self.assertEqual(round(team.compute_ovr(), 1), 0.0)

    @patch('superscript_model.model.Model')
    @patch('superscript_model.organisation.TeamAllocator')
    def test_compute_skill_balance(self, mock_allocator, mock_model):

        dept = Department(0, mock_model)
        workers = [Worker(i, mock_model, dept) for i in range(5)]
        hard_skills = ['A', 'B', 'C', 'D', 'E']
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
        mock_model.p_budget_flexibility = 0.25
        project = Project(ProjectInventory(mock_allocator,
                                           model=mock_model),
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
    @patch('superscript_model.organisation.TeamAllocator')
    def test_compute_creativity_match(self, mock_allocator, mock_model):

        w1 = Worker(1, mock_model)
        w2 = Worker(2, mock_model)
        w3 = Worker(1, mock_model)
        w4 = Worker(2, mock_model)
        w5 = Worker(1, mock_model)
        w6 = Worker(2, mock_model)
        w7 = Worker(2, mock_model)
        mock_model.p_budget_flexibility = 0.25
        project = Project(ProjectInventory(mock_allocator,
                                           model=mock_model),
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

        soft_skills = ['F', 'G', 'H', 'I', 'J']
        w1.skills.soft_skills = dict(zip(soft_skills,
                                         [1.0, 1.0, 1.0, 1.0, 1.0]))
        w2.skills.soft_skills = dict(zip(soft_skills,
                                         [5.0, 5.0, 5.0, 5.0, 5.0]))
        project = Project(ProjectInventory(mock_allocator,
                                           model=mock_model),
                          project_id=43,
                          project_length=5)
        project.requirements.creativity = 5
        team = Team(project,
                    members={w1.worker_id: w1,
                             w2.worker_id: w2},
                    lead=w1)
        self.assertEqual(team.creativity_match, 0)

    @patch('superscript_model.model.Model')
    @patch('superscript_model.organisation.TeamAllocator')
    def test_log_project_outcome(self, mock_allocator, mock_model):

        w1 = Worker(1, mock_model)
        w2 = Worker(2, mock_model)
        mock_model.p_budget_flexibility = 0.25
        project = Project(ProjectInventory(mock_allocator,
                                           model=mock_model),
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

    def test_skill_update(self):

        model = SuperScriptModel(
            worker_count=10,
            io_dir='tests/',
            load_projects=True,
            save_projects=False
        )
        project = model.inventory.get_loaded_projects_for_timestep(
            time=0
        )[0]

        w1 = model.schedule.agents[0]
        w2 = model.schedule.agents[1]

        for skill in ['A', 'B', 'C', 'D']:
            w1.skills.hard_skills[skill] = 1.5
            w2.skills.hard_skills[skill] = 2.2

        team = Team(project,
                    members={w1.worker_id: w1,
                             w2.worker_id: w2},
                    lead=w1
                    )
        get_skills = lambda w: {
            skill: w.get_skill(skill)
            for skill in project.required_skills
        }
        old_skills = {
            w: get_skills(w1)
            for w in [w1, w2]
        }
        team.skill_update(True, model.inventory.skill_update_func)
        team.skill_update(False, model.inventory.skill_update_func)

        for skill, workers in team.contributions.items():
            for worker_id in workers:
                w = team.members[worker_id]
                if old_skills[w][skill] > 0.0:
                    self.assertNotEqual(
                        w.get_skill(skill), old_skills[w][skill]
                    )


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
        strategy = RandomStrategy(
            SuperScriptModel(worker_count=100,
                             worker_strategy='AllIn',
                             budget_functionality_flag=False
                             )
        )
        mock_model.p_budget_flexibility = 0.25
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
        # self.assertEqual(mock_print.call_count, 2)

    @patch('superscript_model.project.Project')
    def test_invite_bids(self, mock_project):
        mock_project.start_time = 0
        mock_project.length = 5
        strategy = RandomStrategy(SuperScriptModel(worker_count=100,
                                                   department_count=10))
        bids = strategy.invite_bids(mock_project)
        self.assertEqual(len(bids), 100)


class TestBasicStrategy(unittest.TestCase):

    def test_init(self):
        strategy = BasicStrategy(
            SuperScriptModel(100),
            min_team_size=3,
            max_team_size=7
        )
        self.assertIsInstance(strategy.model, Model)
        self.assertEqual(strategy.min_team_size, 3)
        self.assertEqual(strategy.max_team_size, 7)

    def test_select_team(self):
        model = SuperScriptModel(
            worker_count=10,
            io_dir='tests/',
            load_projects=True,
            save_projects=False,
            organisation_strategy='Basic',
            worker_strategy='AllIn'
        )
        project = model.inventory.get_loaded_projects_for_timestep(
            time=0
        )[0]
        strategy = model.inventory.team_allocator.strategy

        team = strategy.select_team(
            project, bid_pool=None
        )

        self.assertIsInstance(team, Team)
        self.assertTrue(set(team.members.values())
                        .issubset(strategy.model.schedule.agents))
        if team.lead is not None:
            self.assertTrue(team.lead in team.members.values())

        team = strategy.select_team(project,
                                    bid_pool=[])
        self.assertEqual(team.members, {})
        self.assertIs(team.lead, None)

    @patch('superscript_model.project.Project')
    def test_invite_bids(self, mock_project):
        mock_project.start_time = 0
        mock_project.length = 5
        strategy = BasicStrategy(SuperScriptModel(worker_count=100,
                                                  department_count=10))
        bids = strategy.invite_bids(mock_project)
        self.assertEqual(len(bids), 100)

    def test_select_top_n(self):
        model = SuperScriptModel(
            worker_count=10,
            io_dir='tests/',
            load_projects=True,
            save_projects=False,
            organisation_strategy='Basic',
            worker_strategy='AllIn'
        )
        project = model.inventory.get_loaded_projects_for_timestep(
            time=0
        )[0]
        strategy = model.inventory.team_allocator.strategy

        strategy.max_team_size = 1
        team = strategy.select_top_n(
            bid_pool=model.schedule.agents, project=project
        )
        self.assertLessEqual(len(team.members), 1)

        strategy.max_team_size = 7
        strategy.min_team_size = 11
        team = strategy.select_top_n(
            bid_pool=model.schedule.agents, project=project
        )
        self.assertEqual(len(team.members), 0)


class TestParallelBasinhopping(unittest.TestCase):

    def setUp(self):
        self.model = model = SuperScriptModel(
            worker_count=10,
            io_dir='tests/',
            load_projects=True,
            save_projects=False,
            organisation_strategy='Basin',
            worker_strategy='AllIn'
        )
        self.project = (
            self.model.inventory
                .get_loaded_projects_for_timestep(
                time=0
            )[0]
        )
        self.strategy = self.model.inventory.team_allocator.strategy

    def test_init(self):
        strategy = ParallelBasinhopping(
            SuperScriptModel(
                worker_count=100,
                number_of_processors=1,
                number_of_basin_hops=0
            ),
            OptimiserFactory(),
            min_team_size=3,
            max_team_size=7
        )
        self.assertIsInstance(strategy.model, Model)
        self.assertEqual(strategy.min_team_size, 3)
        self.assertEqual(strategy.max_team_size, 7)
        self.assertEqual(strategy.niter, 0)
        self.assertEqual(strategy.num_proc, 1)

    def test_select_team(self):
        team = self.strategy.select_team(
            self.project, bid_pool=None
        )

        self.assertIsInstance(team, Team)
        self.assertTrue(
            set(team.members.values()).issubset(
                self.strategy.model.schedule.agents
            )
        )
        if team.lead is not None:
            self.assertTrue(team.lead in team.members.values())

    def test_invite_bids(self):

        bids = (
            self.strategy.invite_bids(self.project)
        )
        for value in bids.values():
            self.assertEqual(len(value), 10)


class TestTeamAllocator(unittest.TestCase):

    def test_init(self):
        allocator = TeamAllocator(
            SuperScriptModel(100), OptimiserFactory()
        )
        self.assertTrue(
            implements_interface(allocator.strategy,
                                 OrganisationStrategyInterface)
        )

    @patch('superscript_model.project.ProjectInventory')
    def test_allocate_team(self, mock_inventory):
        model = SuperScriptModel(
            worker_count=100,
            worker_strategy='AllIn',
            organisation_strategy='Random',
            budget_functionality_flag=False
        )
        mock_inventory.model = model
        project = Project(mock_inventory,
                          project_id=0,
                          project_length=5,
                          start_time=0)

        allocator = TeamAllocator(
            model,
            OptimiserFactory()
        )
        allocator.allocate_team(project)
        self.assertIsInstance(project.team, Team)


class TestDepartment(unittest.TestCase):

    @patch('superscript_model.model.Model')
    def test_init(self, mock_model):
        dept = Department(0, mock_model)
        self.assertEqual(dept.workload, DEPARTMENTAL_WORKLOAD)
        self.assertEqual(dept.units_per_full_time, UNITS_PER_FTE)
        self.assertEqual(dept.slack, WORKLOAD_SATISFIED_TOLERANCE)

    @patch('superscript_model.model.Model')
    def test_is_workload_satisfied(self, mock_model):
        dept = Department(0, mock_model)
        dept.number_of_workers += 1
        dept.units_supplied_to_projects[0] = 8

        self.assertFalse(dept.is_workload_satisfied(0, 1))

    @patch('superscript_model.model.Model')
    def test_to_string(self, mock_model):
        dept = Department(0, mock_model)
        self.assertIsInstance(dept.to_string(), str)

    @patch('superscript_model.model.Model')
    def test_add_training(self, mock_model):
        mock_model.trainer = Trainer(mock_model)
        dept = Department(0, mock_model)
        worker = Worker(42, mock_model, department=dept)
        dept.add_training(worker, 5)
        # worker.model.trainer.train(worker)

    def test_assign_work(self):

        model = SuperScriptModel(
            worker_count=2,
            department_count=1,
            departmental_workload=0.8,
            worker_strategy='AllIn',
            organisation_strategy='Basin',
            budget_functionality_flag=False
        )
        model.inventory.create_projects(1, time=0, length=5)
        model.inventory.projects[0].start_time = 0

        for worker in model.schedule.agents:
            for skill in ['A', 'B', 'C', 'D', 'E']:
                worker.contributions.add_contribution(model.inventory.projects[0], skill)

        model.departments[0].assign_work('verbose')

        self.assertLess(
            sum(worker.departmental_work_units for worker in model.schedule.agents) / 20,
            0.8
        )


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
        dept = Department(0, mock_model)
        workers = []
        for i in range(5):
            w = Worker(i, mock_model, department=dept)
            w.skills.hard_skills = dict(
                zip(HARD_SKILLS, [i + 1 for s in HARD_SKILLS])
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
        mock_model.inventory = ProjectInventory(
            mock_allocator, model=mock_model
        )
        mock_model.inventory.total_skill_requirement = {
            'A': 10, 'B': 9
        }

        trainer = Trainer(mock_model)
        dept = Department(0, mock_model)
        workers = []
        for i in range(5):
            w = Worker(i, mock_model, department=dept)
            w.skills.hard_skills = dict(
                zip(HARD_SKILLS, [i + 1 for s in HARD_SKILLS])
            )
            workers.append(w)
            mock_model.schedule.add(w)

        mock_model.training_mode = 'all'

        trainer.update_skill_quartiles()
        mock_model.training_commences = 0
        for i in range(5):
            trainer.train()

        self.assertEqual(workers[0].skills.hard_skills['A'], 4)
        self.assertEqual(workers[0].skills.hard_skills['B'], 1)
        self.assertEqual(workers[0].skills.hard_skills['C'], 1)
        self.assertEqual(workers[1].skills.hard_skills['B'], 2)
        self.assertEqual(workers[2].skills.hard_skills['A'], 3)
        self.assertEqual(workers[4].skills.hard_skills['B'], 5)

        for w in mock_model.schedule.agents:
            mock_model.schedule.remove(w)

        workers = []
        for i in range(5):
            w = Worker(i, mock_model, department=dept)
            w.skills.hard_skills = dict(
                zip(HARD_SKILLS, [i + 1 for s in HARD_SKILLS])
            )
            workers.append(w)
            mock_model.schedule.add(w)
        mock_model.training_mode = 'slots'
        mock_model.target_training_load = 0.2
        mock_model.worker_count = 5
        trainer.training_length = 1
        trainer.train()
        self.assertEqual(len(trainer.trainees), 1)

        mock_model.training_mode = 'not_implemented'
        trainer.train()

    @patch('superscript_model.model.Model')
    @patch('superscript_model.organisation.TeamAllocator')
    def test_training_boost(self, mock_allocator, mock_model):

        mock_model.schedule = RandomActivation(mock_model)
        mock_model.inventory = ProjectInventory(
            mock_allocator, model=mock_model
        )
        mock_model.inventory.total_skill_requirement = {
            'A': 10, 'B': 9
        }

        trainer = Trainer(mock_model)
        dept = Department(0, mock_model)
        workers = []
        for i in range(5):
            w = Worker(i, mock_model, department=dept)
            w.skills.hard_skills = dict(
                zip(HARD_SKILLS, [i + 1 for s in HARD_SKILLS])
            )
            workers.append(w)
            mock_model.schedule.add(w)

        mock_model.training_mode = 'all'

        trainer.update_skill_quartiles()
        trainer.training_boost()

        for skill in ['A', 'B', 'C', 'D', 'E']:
            self.assertEqual(workers[0].skills.hard_skills[skill], 4)
            self.assertEqual(workers[1].skills.hard_skills[skill], 4)
            self.assertEqual(workers[2].skills.hard_skills[skill], 3)
            self.assertEqual(workers[3].skills.hard_skills[skill], 4)
            self.assertEqual(workers[4].skills.hard_skills[skill], 5)
