import unittest
from unittest.mock import patch

import inspect
from mesa import Agent

from superscript_model.worker import (Worker,
                                      WorkerStrategyInterface,
                                      AllInStrategy)
from superscript_model.project import Project


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
        self.assertTrue(strategy.bid(Project(42)))

    def test_accept(self):

        strategy = AllInStrategy('test')
        self.assertTrue(strategy.accept(Project(42)))



