import unittest
from unittest.mock import patch
import numpy as np
import time
from .test_worker import implements_interface

from mesa import Model
from mesa.time import RandomActivation
from superscript_model.model import SuperScriptModel
from superscript_model.worker import Worker, SkillMatrix
from superscript_model.project import ProjectInventory, Project
from superscript_model.organisation import Team
from superscript_model.optimisation import (OptimiserFactory,
                                            Basinhopping,
                                            BHStep,
                                            BHConstraints,
                                            DummyReturn)

from superscript_model.config import (DEPARTMENTAL_WORKLOAD,
                                      UNITS_PER_FTE,
                                      WORKLOAD_SATISFIED_TOLERANCE,
                                      HARD_SKILLS,
                                      TRAINING_COMMENCES,
                                      TRAINING_LENGTH,
                                      MAX_SKILL_LEVEL)


class TestOptimiserFactory(unittest.TestCase):

    def test_get(self):

        model = SuperScriptModel(
            worker_count=100
        )
        project = Project(
            model.inventory,
            project_id=0,
            project_length=5,
            start_time=0)

        factory = OptimiserFactory()
        optimiser = factory.get(
            optimiser_name='ParallelBasinhopping',
            project=project,
            bid_pool=[],
            model=model
        )
        self.assertIsInstance(optimiser, Basinhopping)


class TestOptimiser(unittest.TestCase):

    def setUp(self):
        self.model = SuperScriptModel(
            worker_count=100,
        )
        self.project = Project(
            self.model.inventory,
            project_id=0,
            project_length=5,
            start_time=0)

        self.optimiser = OptimiserFactory().get(
            optimiser_name='ParallelBasinhopping',
            project=self.project,
            bid_pool=self.model.schedule.agents,
            model=self.model,
            save_flag=True,
            results_dir='model_development/experiments/tests/'
        )

    def test_init(self):

        self.assertEqual(
            self.optimiser.worker_ids,
            [agent.worker_id for agent in self.model.schedule.agents]
        )
        self.assertIsInstance(self.optimiser.constraints, list)
        self.assertIsInstance(self.optimiser.worker_unit_budgets, dict)

    def test_smart_guess(self):
        x = self.optimiser.smart_guess()
        self.assertEqual(
            len(x),
            len(self.model.schedule.agents) * 5
        )

        x = self.optimiser.smart_guess(time_limit=0)
        self.assertIsNone(x)

    def test_get_team(self):
        team = self.optimiser.get_team(x=None)
        self.assertIsNone(team)

        t = time.time()
        team = self.optimiser.get_team(self.optimiser.smart_guess())
        elapsed = time.time() - t
        if team is not None:
            self.assertIsInstance(team, Team)
        else:
            self.assertGreater(elapsed, 1)


        team = self.optimiser.get_team(
            np.zeros(len(self.model.schedule.agents) * 5))
        self.assertIsInstance(team, Team)
        self.assertIsNone(team.lead)

    def test_dummy_return(self):
        dummy = DummyReturn()
        self.assertEqual(dummy.fun, 0.0)
        self.assertEqual(dummy.x, None)

    def test_objective_func(self):
        value = self.optimiser.objective_func(
            self.optimiser.smart_guess()
        )
        self.assertLessEqual(value, 0.0)

    def test_test_constraints(self):
        self.assertFalse(
            self.optimiser.test_constraints(
                np.ones(len(self.model.schedule.agents) * 5),
                verbose=True
            )
        )

    def test_solve(self):

        elapsed, ret = self.optimiser.solve(
            guess=None,
            niter=0,
            repeat=0
        )
        self.assertEqual(elapsed, 0.0)
        self.assertIsInstance(ret, DummyReturn)

        self.optimiser.verbose = True
        elapsed, ret = self.optimiser.solve(
            guess=self.optimiser.smart_guess(),
            niter=0,
            repeat=0
        )
        self.optimiser.verbose = False

        self.assertGreater(elapsed, 0.0)
        self.assertLessEqual(ret.fun, 0.0)
        self.assertEqual(len(ret.x),
                         len(self.model.schedule.agents) * 5)

    def test_compute_distances_from_requirements(self):

        dists = self.optimiser.assign_dist_probs_from_requirements()
        self.assertIsInstance(dists, dict)
        self.assertEqual(
            set(dists.keys()),
            set([w.worker_id for w in self.optimiser.bid_pool])
        )

    def test_my_constraints(self):

        cons = BHConstraints(self.optimiser)
        self.assertEqual(cons.test, self.optimiser.test_constraints)

        self.assertIsInstance(
            cons.__call__(x_new=self.optimiser.smart_guess()),
            bool)

    def test_my_take_step(self):

        stepper = BHStep(self.optimiser)
        new_x = stepper.__call__(self.optimiser.smart_guess())
        self.assertEqual(
            len(new_x),
            len(self.model.schedule.agents) * 5
        )

        stepper.time_limit = 0
        old_x = self.optimiser.smart_guess()
        new_x = stepper.__call__(old_x)
        self.assertEqual(list(new_x), list(old_x))

        stepper.min_team_size = stepper.max_team_size + 1
        new_x = stepper.__call__(self.optimiser.smart_guess())
        self.assertEqual(
            len(new_x),
            len(self.model.schedule.agents) * 5
        )
