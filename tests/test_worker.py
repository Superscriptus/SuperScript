import unittest
from unittest.mock import patch

import inspect
from mesa import Agent

from superscript_model.model import SuperScriptModel
from superscript_model.worker import (Worker,
                                      WorkerStrategyInterface,
                                      AllInStrategy,
                                      StakeStrategy,
                                      SkillMatrix)
from superscript_model.project import Project, ProjectInventory
from superscript_model.config import (HARD_SKILLS,
                                      SKILL_DECAY_FACTOR,
                                      WORKER_SUCCESS_HISTORY_LENGTH,
                                      REPLACE_AFTER_INACTIVE_STEPS)


def implements_interface(cls, interface):

    if not inspect.isclass(cls):
        cls = cls.__class__

    return (len(list(cls.interfaces())) == 1
            and list(cls.interfaces())[0] == interface)


class TestWorker(unittest.TestCase):

    @patch('superscript_model.model.Model')
    def test_init(self, mock_model):

        worker = Worker(42, mock_model)
        self.assertTrue(worker.worker_id == 42)
        self.assertIsInstance(worker, Agent)
        self.assertIsInstance(worker.skills, SkillMatrix)
        self.assertTrue(implements_interface(worker.strategy,
                                             WorkerStrategyInterface))

    @patch('superscript_model.model.Model')
    @patch('superscript_model.project.Project')
    def test_assign_as_lead(self, mock_project, mock_model):
        worker = Worker(42, mock_model)
        worker.assign_as_lead(mock_project)
        self.assertEqual(len(worker.leads_on.keys()), 1)

    @patch('superscript_model.model.Model')
    @patch('superscript_model.project.Project')
    def test_remove_as_lead(self, mock_project, mock_model):
        worker = Worker(42, mock_model)
        self.assertRaises(KeyError, worker.remove_as_lead, mock_project)

        worker.assign_as_lead(mock_project)
        self.assertEqual(len(worker.leads_on.keys()), 1)
        worker.remove_as_lead(mock_project)
        self.assertEqual(len(worker.leads_on.keys()), 0)

    @patch('superscript_model.model.Model')
    @patch('superscript_model.organisation.TeamAllocator')
    @patch('superscript_model.project.Project.terminate')
    def test_step(self, mock_terminate, mock_allocator, mock_model):

        mock_model.replace_after_inactive_steps = (
            REPLACE_AFTER_INACTIVE_STEPS
        )
        mock_model.p_budget_flexibility = 0.25
        worker = Worker(42, mock_model)
        project = Project(ProjectInventory(mock_allocator,
                                           model=mock_model),
                          project_id=42,
                          project_length=2)

        worker.assign_as_lead(project)
        self.assertEqual(project.progress, 0)
        worker.step()
        self.assertEqual(project.progress, 1)
        worker.step()
        self.assertEqual(mock_terminate.call_count, 1)

    @patch('superscript_model.model.Model')
    def test_get_skill(self, mock_model):

        worker = Worker(42, mock_model)

        skill_a = worker.get_skill('A', hard_skill=True)
        self.assertIsInstance(skill_a, float)
        self.assertTrue((skill_a >= 0.0) & (skill_a <= 5.0))

        skill_f = worker.get_skill('F', hard_skill=False)
        self.assertIsInstance(skill_f, float)
        self.assertTrue((skill_f >= 0.0) & (skill_f <= 5.0))

    @patch('superscript_model.model.Model')
    @patch('superscript_model.organisation.TeamAllocator')
    def test_get_units_contributed(self, mock_allocator, mock_model):
        worker = Worker(42, mock_model)
        mock_model.p_budget_flexibility = 0.25
        project = Project(ProjectInventory(mock_allocator,
                                           model=mock_model),
                          project_id=42,
                          project_length=2)

        worker.contributions.add_contribution(project, 'B')
        worker.contributions.add_contribution(project, 'C')
        worker.contributions.add_contribution(project, 'D')
        self.assertEqual(
            worker.contributions.get_units_contributed(0), 3
        )

    @patch('superscript_model.model.Model')
    @patch('superscript_model.organisation.TeamAllocator')
    def test_contributes_less_than_full_time(self, mock_allocator, mock_model):
        mock_model.p_budget_flexibility = 0.25
        worker = Worker(42, mock_model)
        worker.skills.hard_skills = dict(zip(HARD_SKILLS, [1, 2, 3, 4, 5]))
        project = Project(ProjectInventory(mock_allocator,
                                           model=mock_model),
                          project_id=42,
                          project_length=2)
        worker.contributions.add_contribution(project, 'B')
        self.assertTrue(
            worker.contributions.contributes_less_than_full_time(0, 1)
        )
        for i in range(10):
            worker.contributions.add_contribution(project, 'B')
        self.assertFalse(
            worker.contributions.contributes_less_than_full_time(0, 1)
        )

    @patch('superscript_model.model.Model')
    def test_degrade_unused_skills(self, mock_model):
        worker = Worker(42, mock_model)

        worker_skills = {
            skill: worker.get_skill(skill, hard_skill=True)
            for skill in HARD_SKILLS
        }

        worker.skills.decay(worker)
        for skill in HARD_SKILLS:
            self.assertEqual(worker.get_skill(skill),
                             SKILL_DECAY_FACTOR * worker_skills[skill])

    @patch('superscript_model.model.Model')
    @patch('superscript_model.organisation.TeamAllocator')
    def test_individual_chemistry(self, mock_allocator, mock_model):

        mock_model.p_budget_flexibility = 0.25
        worker = Worker(42, mock_model)
        worker.skills.hard_skills = dict(zip(HARD_SKILLS, [1, 2, 3, 4, 5]))
        project = Project(ProjectInventory(mock_allocator,
                                           model=mock_model),
                          project_id=42,
                          project_length=2)
        project.requirements.hard_skills = {'A': {'units': 3, 'level': 4}}
        project.requirements.risk = worker.skills.ovr
        self.assertEqual(worker.individual_chemistry(project), 1)
        project.requirements.hard_skills = {'D': {'units': 2, 'level': 5}}
        self.assertEqual(worker.individual_chemistry(project), 2)

    def test_replace(self):

        model = SuperScriptModel(100, department_count=10)
        self.assertEqual(model.departments[1].number_of_workers, 10)
        worker = model.schedule.agents[0]
        worker2 = model.schedule.agents[1]
        self.assertEqual(worker.department.number_of_workers, 10)
        worker.replace()
        worker2.leads_on[42] = None
        worker2.replace()
        self.assertEqual(worker.department.number_of_workers, 10)
        self.assertEqual(model.schedule.get_agent_count(), 100)

    def test_check_activity(self):

        model = SuperScriptModel(10, department_count=1)
        worker = model.schedule.agents[0]
        old_id = worker.worker_id
        worker.timesteps_inactive = 10
        worker.check_activity()
        self.assertNotEqual(
            old_id,
            model.schedule.agents[0]
        )

    @patch('superscript_model.model.Model')
    @patch('superscript_model.project.Project')
    def test_compute_worker_roi(self, mock_project, mock_model):
        worker = Worker(42, mock_model)
        worker.training_remaining = 1
        worker.compute_worker_roi()
        self.assertEqual(worker.roi, 5)


class TestWorkerHistory(unittest.TestCase):

    @patch('superscript_model.model.Model')
    def test_init(self, mock_model):
        worker = Worker(42, mock_model)
        self.assertEqual(worker.history.success_history_length,
                         WORKER_SUCCESS_HISTORY_LENGTH)
        self.assertTrue(len(
            worker.history.success_history) <= WORKER_SUCCESS_HISTORY_LENGTH)

    @patch('superscript_model.model.Model')
    def test_get_success_rate(self, mock_model):
        worker = Worker(42, mock_model)

        for i in range(5):
            worker.history.record(True)

        self.assertTrue(len(
            worker.history.success_history) <= WORKER_SUCCESS_HISTORY_LENGTH)
        self.assertEqual(worker.history.get_success_rate(), 1)


class TestWorkerStrategyInterface(unittest.TestCase):

    def test_interface(self):
        self.assertRaises(TypeError, lambda: WorkerStrategyInterface())


class TestAllInStrategy(unittest.TestCase):

    def test_init(self):

        self.assertTrue(implements_interface(AllInStrategy, WorkerStrategyInterface))

        strategy = AllInStrategy('test')
        self.assertEqual(strategy.name, 'test')

    @patch('superscript_model.model.Model')
    @patch('superscript_model.organisation.TeamAllocator')
    def test_bid(self, mock_allocator, mock_model):
        mock_model.p_budget_flexibility = 0.25
        mock_model.budget_functionality_flag = True
        inventory = ProjectInventory(
            mock_allocator,
            model=mock_model
        )
        worker = Worker(42, mock_model)
        strategy = AllInStrategy('test')
        self.assertTrue(strategy.bid(Project(
            inventory,
            project_id=42,
            project_length=5
        ), worker))
        inventory.create_projects(10, 0, 5)
        for project in inventory.projects.values():
            project.start_time = 0
            worker.contributions.add_contribution(project, 'A')
        self.assertFalse(strategy.bid(Project(
            inventory,
            project_id=0,
            project_length=5
        ), worker))

    @patch('superscript_model.model.SuperScriptModel')
    @patch('superscript_model.organisation.TeamAllocator')
    def test_accept(self, mock_allocator, mock_model):
        mock_model.p_budget_flexibility = 0.25
        project = Project(ProjectInventory(mock_allocator,
                                           model=mock_model),
                          project_id=42,
                          project_length=2)
        strategy = AllInStrategy('test')
        self.assertTrue(strategy.accept(project))


class TestStakeStrategy(unittest.TestCase):

    def test_init(self):
        self.assertTrue(implements_interface(StakeStrategy, WorkerStrategyInterface))
        strategy = StakeStrategy('test')
        self.assertEqual(strategy.name, 'test')

    @patch('superscript_model.model.Model')
    @patch('superscript_model.organisation.TeamAllocator')
    def test_bid(self, mock_allocator, mock_model):
        mock_model.p_budget_flexibility = 0.25
        mock_model.budget_functionality_flag = True
        inventory = ProjectInventory(
            mock_allocator,
            model=mock_model
        )
        worker = Worker(42, mock_model)
        strategy = StakeStrategy('test')

        inventory.create_projects(10, 0, 5)
        inventory.projects[1].requirements.risk = 1000
        self.assertFalse(
            strategy.bid(inventory.projects[1], worker)
        )
        inventory.projects[1].requirements.risk = 0
        self.assertTrue(
            strategy.bid(inventory.projects[1], worker)
        )
        for project in inventory.projects.values():
            project.start_time = 0
            worker.contributions.add_contribution(project, 'A')
        self.assertFalse(strategy.bid(Project(
            inventory,
            project_id=0,
            project_length=5
        ), worker))

    @patch('superscript_model.model.SuperScriptModel')
    @patch('superscript_model.organisation.TeamAllocator')
    def test_accept(self, mock_allocator, mock_model):
        mock_model.p_budget_flexibility = 0.25
        project = Project(ProjectInventory(mock_allocator,
                                           model=mock_model),
                          project_id=42,
                          project_length=2)
        strategy = StakeStrategy('test')
        self.assertTrue(strategy.accept(project))


class TestSkillMatrix(unittest.TestCase):

    def test_init(self):
        skills = SkillMatrix()
        self.assertEqual(skills.max_skill, 5)
        self.assertEqual(skills.hard_skill_probability, 0.8)
        self.assertEqual(skills.ovr_multiplier, 20)
        self.assertEqual(skills.round_to, 1)

        self.assertIsInstance(skills.hard_skills, dict)
        self.assertIsInstance(skills.soft_skills, dict)
        self.assertEqual(len(skills.hard_skills.keys()), 5)
        self.assertEqual(len(skills.soft_skills.keys()), 5)

    def test_assign_hard_skills(self):

        skills = SkillMatrix()
        skills.assign_hard_skills()
        self.assertTrue(sum(skills.hard_skills.values()) > 0.0)
        for skill in skills.hard_skills.values():
            self.assertTrue((skill >= 0.0) & (skill <= 5.0))

    def test_to_string(self):

        skills = SkillMatrix()
        self.assertIsInstance(skills.to_string(), str)

    def test_get_ovr(self):
        skills = SkillMatrix()
        skills.hard_skills = {'A': 0.0,
                              'B': 3.9,
                              'C': 3.2,
                              'D': 4.1,
                              'E': 1.5}
        self.assertEqual(skills.ovr, 63.5)