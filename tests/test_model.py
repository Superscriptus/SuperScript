import unittest
from unittest.mock import patch

from mesa import Model
from mesa.time import BaseScheduler
from superscript_model.model import SuperScriptModel
from superscript_model.config import (NEW_PROJECTS_PER_TIMESTEP,
                                      MIN_TEAM_SIZE,
                                      MAX_TEAM_SIZE)


class TestSuperScriptModel(unittest.TestCase):

    def test_init(self):

        model = SuperScriptModel(worker_count=100)
        self.assertEqual(model.worker_count, 100)
        self.assertIsInstance(model, Model)
        self.assertIsInstance(model.schedule, BaseScheduler)
        self.assertEqual(model.schedule.get_agent_count(), 100)

    @patch('superscript_model.model.RandomActivation.step')
    def test_step(self, mock_schedule):

        model = SuperScriptModel(worker_count=100)
        model.step()
        mock_schedule.assert_called_once()

    @patch('superscript_model.model.RandomActivation.step')
    def test_run_model(self, mock_step):
        model = SuperScriptModel(worker_count=1000,
                                 department_count=10)
        model.run_model(2)
        self.assertEqual(mock_step.call_count, 2)
        self.assertEqual(model.schedule.get_agent_count(), 1000)

    def test_integration(self):
        model = SuperScriptModel(worker_count=1000,
                                 department_count=10,
                                 budget_functionality_flag=False,
                                 worker_strategy='AllIn',
                                 organisation_strategy='Random'
                                 )
        model.trainer.training_commences = 0
        model.run_model(1)
        self.assertEqual(model.schedule.get_agent_count(), 1000)
        self.assertEqual(
            len(model.inventory.projects),
            NEW_PROJECTS_PER_TIMESTEP
        )
        model.run_model(1)
        self.assertEqual(model.schedule.get_agent_count(), 1000)
        self.assertEqual(
            len(model.inventory.projects),
            2 * NEW_PROJECTS_PER_TIMESTEP
        )

        model = SuperScriptModel(worker_count=1000,
                                 department_count=10,
                                 budget_functionality_flag=False,
                                 worker_strategy='AllIn',
                                 organisation_strategy='Random'
                                 )
        model.trainer.training_commences = 10
        model.run_model(2)
        self.assertEqual(model.schedule.get_agent_count(), 1000)
        self.assertEqual(
            len(model.inventory.projects),
            2 * NEW_PROJECTS_PER_TIMESTEP
        )

        for project in model.inventory.projects.values():
            if project.team.lead is not None:
                self.assertTrue(
                    project.team.size >= MIN_TEAM_SIZE
                )
                self.assertTrue(
                    project.team.size <= MAX_TEAM_SIZE
                )


if __name__ == '__main__':
    unittest.main()
