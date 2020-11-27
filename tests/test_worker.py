import unittest
from mesa import Agent

from superscript_model.worker import (Worker,
                                      WorkerStrategyInterface,
                                      AllInStrategy)
from superscript_model.project import Project


class TestWorker(unittest.TestCase):

    def test_init(self):

        worker = Worker(worker_id=42)
        self.assertTrue(worker.worker_id == 42)
        self.assertIsInstance(worker, Agent)


class TestWorkerStrategyInterface(unittest.TestCase):

    def test_interface(self):
        self.assertRaises(TypeError, lambda: WorkerStrategyInterface())


class TestAllInStrategy(unittest.TestCase):

    def test_init(self):

        self.assertEquals(len(list(AllInStrategy.interfaces())), 1)
        self.assertEquals(list(AllInStrategy.interfaces())[0],
                          WorkerStrategyInterface)

        strategy = AllInStrategy('test')
        self.assertEquals(strategy.name, 'test')

    def test_bid(self):

        strategy = AllInStrategy('test')
        self.assertTrue(strategy.bid(Project(42)))

    def test_accept(self):

        strategy = AllInStrategy('test')
        self.assertTrue(strategy.accept(Project(42)))



