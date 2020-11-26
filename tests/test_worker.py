import unittest

from superscript_model.worker import Worker
from mesa import Agent


class TestWorker(unittest.TestCase):

    def test_init(self):

        worker = Worker(worker_id=42)
        self.assertTrue(worker.worker_id == 42)
        self.assertIsInstance(worker, Agent)
