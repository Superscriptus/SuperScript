import unittest
from unittest.mock import patch

import inspect
from mesa import Agent

from superscript_model.worker import (Worker,
                                      WorkerStrategyInterface,
                                      AllInStrategy,
                                      SkillMatrix)
from superscript_model.project import Project, ProjectInventory


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

        worker = Worker(42, mock_model)
        project = Project(ProjectInventory(mock_allocator),
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

class TestWorkerStrategyInterface(unittest.TestCase):

    def test_interface(self):
        self.assertRaises(TypeError, lambda: WorkerStrategyInterface())


class TestAllInStrategy(unittest.TestCase):

    def test_init(self):

        self.assertTrue(implements_interface(AllInStrategy, WorkerStrategyInterface))

        strategy = AllInStrategy('test')
        self.assertEqual(strategy.name, 'test')

    def test_bid(self):

        strategy = AllInStrategy('test')
        self.assertTrue(strategy.bid(Project(42, 5)))

    def test_accept(self):

        strategy = AllInStrategy('test')
        self.assertTrue(strategy.accept(Project(42, 5)))


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