import unittest
from unittest.mock import patch

import inspect
from mesa import Agent

from superscript_model.worker import (Worker,
                                      WorkerStrategyInterface,
                                      AllInStrategy)
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



