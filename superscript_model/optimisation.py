"""
SuperScript optimisation module
===========

This module handles the optimisation methods used for 'optimal' team
selection. Currently (v1.0) the only method implemented is parallel
basinhopping, but the factory method ensures that new optimisation
methods can easily be added in the future (see Roadmap in README.md).

An optimisation method consists of two component classes:
1. A runner
2. An optimiser

The runner.run() method is called by the team allocation strategy
(e.g. organisation.ParallelBasinhopping) and allows multiple
optimiser.solve() calls to be mapped across multiple cores/processes.

Classes:
    OptimiserFactory
        Returns an optimiser and a runner for team selection according
        to `optimiser_name` and `runner_name`.
    OptimiserInterface
        Defines interface methods for an optimiser, such that new
        optimisers can easily be added in the future.
    RunnerInterface
        Defines interface methods for a runner, such that a new runner
        could be added in the future (e.g. different parallelisation
        approach, or single core runner, although ParallelRunner has
        capability to run on single core by setting num_proc=1).
    DummyReturn
        Dummy return value (in format of scipy optimizer return) that
        is used when an optimisation fails (or timeouts).
    ParallelRunner
        Takes an optimiser and maps optimiser.solve() across
        multiple cores/processes for more efficient search.
    Basinhopping
        Basinhopping optimiser - conducts a series of basin hops with
        COBYLA linear optimiser at each hop.
    MyConstraints
        Constraints on the team that can be selected (in the format
        required for COBYLA).
    MyTakeStep
        Bespoke step method used in the basinhopping routine.
"""
from .organisation import Team
from .utilities import Random
from .config import MAX_TEAM_SIZE, MIN_TEAM_SIZE

from pathos.multiprocessing import ProcessingPool as Pool
from numpy import argmax
from interface import Interface, implements
from scipy.optimize import basinhopping
from scipy.spatial import minkowski_distance
import numpy as np
import pandas as pd
import time
import pickle


class OptimiserFactory:
    """Simple factory for supplying a runner and an optimiser.
    """
    @staticmethod
    def get_runner(
            runner_name, optimiser,
            num_proc=1,
            niter=0
    ):
        """Returns a runner object according to runner_name.

        Note:
            Currently (v1.0) only ParallelRunner implements this
            interface.

        Args:
            runner_name: str
                Which type of runner to create.
            optimiser: OptimiserInterface
                An optimiser for the runner to call.
            num_proc: int
                Number of processors to run on.
            niter: int
                Number of iterations (optimiser parameter).
        """
        if runner_name == "Parallel":
            return ParallelRunner(
                optimiser,
                num_proc,
                min_team_size=MIN_TEAM_SIZE,
                max_team_size=MAX_TEAM_SIZE
            )

    @staticmethod
    def get_optimiser(optimiser_name, project,
                      bid_pool, model, niter,
                      save_flag=False, results_dir=None):
        """Returns an optimiser object according to optimiser_name.

        Args:
            optimiser_name: str
                Which type of optimiser to create.
            project: project.Project
                The project for which a team is being allocated.
            bid_pool:
                The bid_pool from which to select the workers.
            model: model.SuperScriptModel
                Reference to main model.
            save_flag: bool (optional)
                Allows optimisation outputs to be saved to disk for
                development and benchmarking of the optimisation
                method.
            results_dir: str (optional)
                Folder in which to save outputs if activated.
        """
        if optimiser_name == 'Basinhopping':
            return Basinhopping(
                project,
                bid_pool,
                model,
                niter,
                save_flag=save_flag,
                results_dir=results_dir
            )


class OptimiserInterface(Interface):
    """Interface class for optimiser.

    Defines the required methods for an optimiser to solve the team
    allocation problem.
    """
    def team_size(self, x) -> int:
        """Returns size of team, calculated from solution vector x"""
        pass

    def solve(self, guess, niter, repeat=0) -> tuple:
        """Solves the optimisation problem.

        Args:
            guess: np.ndarray
                Initial guess of solution vector x
            niter: int
                Number of iterations to run optimiser.
            repeat: int
                Integer counter, used to identify which optimisation
                run this is when doing multiple optimisations in
                parallel, or doing repeated optimisation for
                benchmarking.

        Returns:
            tuple: (elapsed_time, res)
                (Runtime, Result object with attributes 'x' and 'fun')
                    res.x = optimised solution vector
                    res.fun = value of objective function at res.x
        """
        pass

    def get_team(self, x) -> Team:
        """Produces Team from solution vector x."""
        pass

    def objective_func(self, x) -> float:
        """Objective function to be minimised by the optimiser.

        As standard this would be the negative of the probability of
        project success, using the Team encoded in solution vector x.
        """
        pass

    def smart_guess(self) -> np.ndarray:
        """Returns a guess of the solution vector x, used
        for initial guess.

        Args:
            p: float
                Generic parameter.
            time_limit: int
                Maximum number of seconds before timeout.
        """
        pass


class RunnerInterface(Interface):
    """Interface class for runner.

        Defines the required methods for an runner, which calls the
        optimiser including the solve() method.
        """
    def run(self) -> (Team, float):
        """Returns a (locally) optimal Team and the probability of
        success for that team."""
        pass


class DummyReturn:
    """Dummy return value for when a Scipy.optimize method fails or
    timeouts.
    ...

    Attributes:
        fun: float
            Value of objective function.
        x: NoneType
            Solution vector.
    """
    def __init__(self):
        self.fun = 0.0
        self.x = None


class ParallelRunner(implements(RunnerInterface)):
    """Runs num_proc parallel optimisations using the supplied
    optimiser with Pathos.multiprocessing.

    Note:
         The bid_pool must be at least as long as max_team_size to
         work.

    Note:
        If niter=0 the optimiser's smart_guess method is used to
        return a guess solution. This is not mapped across multiple
        processes even if num_proc>1.

    Note:
        This method uses Pathos multiprocessing because the standard
        library uses Pickle to pass data to the new processes, and
        this was not working with some of the class methods and lambda
        functions (Pathos uses Dill instead of Pickle). By most
        accounts the Pathos implementation is better anyway. But there
        are likely other solutions.

        One thing to note is that this is blocking, so the processors
        must wait until the longest basinhopping is complete. For high
        values for `niter` and `num_proc` this can mean that many cores
        are sitting ideal for large parts of the simulation.

    ...

    Attributes:
        optimiser: OptimiserInterface
            Optimiser for runner to use.
        project: project.Project
            The project for which a team is being allocated.
        bid_pool:
            The bid_pool from which to select the workers.
        num_proc: int
            Number of processors to use.
        min_team_size: int
            Minimum size of team allowed.
        max_team_size: int
            Maximum size of team allowed.
    """

    def __init__(
            self, optimiser,
            num_proc,
            min_team_size=MIN_TEAM_SIZE,
            max_team_size=MAX_TEAM_SIZE
    ):

        self.opti = optimiser
        self.project = optimiser.project
        self.bid_pool = optimiser.bid_pool
        self.num_proc = num_proc
        self.min_team_size = min_team_size
        self.max_team_size = max_team_size

    def run(self):
        """Run optimiser across multiple core/processes.

        If the optimiser has niter=0, this method just returns the
        optimiser.smart_guess() for a quick and dirty approximation.

        Returns:
            (Team, float): tuple with best Team found and the
            corresponding probability of project success.
        """
        if len(self.bid_pool) < self.max_team_size:
            return Team(self.project, {}, None), 0.0

        elif self.opti.niter == 0:
            x = self.opti.smart_guess()
            return (
                self.opti.get_team(x),
                -self.opti.objective_func(x)
            )

        else:
            p = Pool(processes=self.num_proc)
            batch_results = p.map(
                self.opti.solve,
                [self.opti.smart_guess() for i in range(self.num_proc)],
                [self.opti.niter for i in range(self.num_proc)],
                range(self.num_proc)
            )

            p.close()
            p.join()
            p.clear()

            probs = [-r[1].fun for r in batch_results]
            team_x = [r[1].x for r in batch_results]

            return (
                self.opti.get_team(team_x[argmax(probs)]),
                max(probs)
            )


class Basinhopping(implements(OptimiserInterface)):
    """Basinhopping optimiser that uses COBYLA optimisation at
    each basin hopping step.

    Both the initial guess (smart_guess) and the BHStep methods select
    members based on how closely their skill set resembles the skill
    required by the project (using minkowski distance).

    Note:
        The solution vector `x` is a binary vector with one entry for
        each hard skill for each worker. So its length is:
        len(bid_pool) * 5

        If an entry is one it indicates that the worker contributes
        that skill to the project.
    ...

    Attributes:
        optimiser: OptimiserInterface
            Optimiser for runner to use.
        project: project.Project
            The project for which a team is being allocated.
        bid_pool:
            The bid_pool from which to select the workers.
        niter: int
            Number of iterations (basinhopping steps).
        exp_number: int
            Integer ID for the number of this experiment,
            used if running becnhmarking.
        skills: list
            Indicates which skills are 'hard' skills.
            By default: ['A', 'B', 'C', 'D', 'E']
        worker_ids: list
            List of worker_id numbers for the workers in the bid_pool
        worker_unit_budgets: dict
            Records how many units each worker has available to
            contribute to the project.
        constraints: dict
            Dictionary of contraints on the solutions vector, in the
            format required for COBYLA (scipy).
        min_team_size: int
            Minimum size of team allowed.
        max_team_size: int
            Maximum size of team allowed.
        save_flag: bool (optional)
            Allows optimisation outputs to be saved to disk for
            development and benchmarking of the optimisation
            method.
        results_dir: str (optional)
            Folder in which to save outputs if activated.
        verbose: bool
            Indicates whether to print details such as time taken to
            complete optimisation and constraint violation.
        smart_guess_timeout: int
            Number of seconds to spend trying to find a smart_guess
            that meets the constraints (else timeout).
        p_norm: int
            P-norm to use for minkowski distance between worker skills
            and project requirements (e.g. p_norm=2 is Euclidean
            distance).
        maxiter: int
            Maximum iterations (COBYLA parameter).
        catol: float
            Tolerance on constraints (COBYLA parameter).
        rhobeg: float
            Reasonable initial change (COBYLA parameter).
    """

    def __init__(
            self, project, bid_pool, model, niter,
            exp_number=None,
            verbose=False, save_flag=False,
            results_dir='model_development/experiments/optimisation/',
            min_team_size=MIN_TEAM_SIZE,
            max_team_size=MAX_TEAM_SIZE,
            smart_guess_timeout=1,
            p_norm=2,
            maxiter=100,
            catol=0.0,
            rhobeg=0.6
    ):

        self.project = project
        self.bid_pool = bid_pool
        self.model = model
        self.niter = niter
        self.exp_number = exp_number
        self.skills = ['A', 'B', 'C', 'D', 'E']
        self.worker_ids = [m.worker_id for m in bid_pool]
        self.worker_unit_budgets = self.get_worker_units_budgets()
        self.constraints = BHConstraints(self, verbose)
        self.min_team_size = min_team_size
        self.max_team_size = max_team_size
        self.save_flag = save_flag
        self.results_dir = results_dir
        self.verbose = verbose
        self.smart_guess_timeout = smart_guess_timeout
        self.p_norm = p_norm
        self.maxiter = maxiter
        self.catol = catol
        self.rhobeg = rhobeg

        if self.save_flag:
            with open(self.results_dir
                      + 'project_%d.json'
                      % self.exp_number, 'wb') as ofile:
                pickle.dump(self.project, ofile)
            with open(self.results_dir
                      + 'bid_pool_%d.json'
                      % self.exp_number, 'wb') as ofile:
                pickle.dump(self.bid_pool, ofile)

    def get_team(self, x):
        """Return Team that is encoded in solution vector x.

        Args:
            x: np.ndarray

        Returns:
            Team: the team represented by x
        """

        if x is None:
            return None

        contributions = dict()
        for skill in self.skills:
            contributions[skill] = []

        team_members = {}
        for wi, worker_id in enumerate(self.worker_ids):

            start = wi * 5
            in_team = False

            for si, skill in enumerate(self.skills):
                if x[start + si] > 0.5:
                    contributions[skill].append(worker_id)
                    in_team = True

            if in_team:
                team_members[worker_id] = self.bid_pool[wi]

        if len(team_members) == 0:
            return Team(self.project, {}, None)
        else:
            return Team(self.project,
                        team_members,
                        team_members[list(team_members.keys())[0]],
                        contributions=contributions)

    def objective_func(self, x):
        """Objective function that the optimiser tries to minimise.

        Because scipy's optimisers do minimisation (not maximisation),
        we take the negative of the probability.

        Args:
            x: np.ndarray
                Solution vector

        Returns:
            float: value of objective function for x ( always <=0).

        """
        test_team = self.get_team(x)

        self.project.team = test_team
        self.model.inventory.success_calculator.calculate_success_probability(
            self.project
        )

        return -self.project.success_probability

    def get_worker_units_budgets(self):
        """For each worker in the bid_pool, determine how many units
        they have available to contribute to the project.

        Returns:
            dict: record of unit availability.
        """
        return {
            worker.worker_id: worker.contributions.get_remaining_units(
                self.project.start_time, self.project.length
            )
            for worker in self.bid_pool
        }

    def adjusted_dept_unit_budget(
            self, x,
            dept_id,
            base_dept_unit_budgets,
            dept_members
    ):
        """Updates the available units in each department based on the
        current allocation in the solution vector x. This ensures that
        departmental unit budget will always be met.

        Args:
            x: np.ndarray
                Solution vector
            dept_id: int
                Identifier for the department
            base_dept_unit_budgets: dict
                Dictionary of current unit availability in each
                department.
            dept_members: dict
                Dictionary of workers belonging to each department.

        Returns:
            dict: updated copy of base_dept_unit_budgets
        """

        base = base_dept_unit_budgets[dept_id]
        for m in dept_members[dept_id]:
            start = self.bid_pool.index(m)
            base -= sum(np.round(x[start:start + 5]))

        return base

    def team_size(self, x):
        """Calculate how many workers in the team represented by x.

        Args:
            x: np.ndarray
                Solution vector

        Returns:
            int: number of workers with non-zero entries in x
        """

        size = 0
        for wi, worker_id in enumerate(self.worker_ids):
            start = wi * 5
            if sum(np.round(x[start:start + 5])) > 0:
                size += 1

        return size

    def solve(self, guess, niter, repeat=0):
        """Solve the optimisation problem by trying to minimise
        the objective function.

        Args:
            guess: np.ndarray
                Initial guess of solution vector x.
            niter: int
                Number of iterations (i.e. number of basin hops).
            repeat: int (optional)
                Integer ID for this run, used when doing multiple
                optimisations for benchmarking. Not required for
                normal simulations.
        """

        if guess is None:
            return 0.0, DummyReturn()

        minimizer_kwargs = {
            "method": 'COBYLA',
            'constraints': self.constraints.constraint_list,
            'options': {
                'maxiter': self.maxiter,
                'disp': False,
                'catol': self.catol,
                'rhobeg': self.rhobeg
            }
        }

        my_takestep = BHStep(self)

        start_time = time.time()
        res = basinhopping(self.objective_func, guess,
                           minimizer_kwargs=minimizer_kwargs,
                           niter=niter, seed=70470,
                           accept_test=self.constraints,
                           take_step=my_takestep)

        if (res.fun >= 0.0
                or sum(res.x) == 0
                or not self.constraints.test(res.x)):

            res.x = guess
            res.fun = self.objective_func(res.x)

        assert self.constraints.test(res.x)
        elapsed_time = time.time() - start_time

        if self.verbose:
            print("%d iterations took %.2f seconds" % (niter, elapsed_time))

        if self.save_flag:
            best_team = self.get_team(res.x)
            self.project.team = best_team
            self.model.inventory.success_calculator.calculate_success_probability(
                self.project
            )

            with open(self.results_dir
                      + 'best_team_project_%d_niter_%d_repeat_%d.json'
                      % (self.exp_number, niter, repeat), 'wb') as ofile:
                pickle.dump(best_team, ofile)

            with open(self.results_dir
                      + 'prob_summary_project_%d_niter_%d_repeat_%d.json'
                      % (self.exp_number, niter, repeat), 'wb') as ofile:
                pickle.dump(
                    self.model.inventory.success_calculator.to_string(self.project),
                    ofile
                )

        return elapsed_time, res

    def assign_dist_probs_from_requirements(
            self, project=None, workers=None
    ):
        """Assigns a probability for each worker based on the distance
        between their skill set and the skills required by the project.

        Args:
            project: project.Project (optional)
                Project to use requirements from.
            workers: list (optional)
                List of workers to assign probabilities to.

        Returns:
            dict: with a probability entry for each worker
        """

        if project is None:
            project = self.project
        if workers is None:
            workers = self.bid_pool

        worker_table = pd.DataFrame()
        worker_dict = {m.worker_id: m
                       for m in workers}
        worker_table['id'] = worker_dict.keys()

        for skill in project.required_skills:
            worker_table[skill] = [m.get_skill(skill) for m in workers]

        required_levels = [
            project.requirements.hard_skills[skill]['level']
            for skill in project.required_skills
        ]

        worker_table['distance'] = [
            minkowski_distance(
                row[project.required_skills],
                required_levels,
                self.p_norm
            )
            for ri, row in worker_table.iterrows()
        ]

        # Avoids division by zero when distance is zero
        probabilities = [
            (1 / d) if d > 0 else -1
            for d in worker_table.distance
        ]
        # Ensures that workers with distance = zero are assigned the
        # highest probability.
        worker_table['prob'] = [
            max(probabilities) + 1
            if p == -1 else p
            for p in probabilities
        ]

        if sum(worker_table['prob']) > 0:
            worker_table['prob'] /= sum(worker_table['prob'])

        worker_table.sort_values('prob', ascending=False, inplace=True)

        return dict(zip(worker_table.id, worker_table.prob))

    def smart_guess(self):
        """Guesses a solution vector by assigning randomly assigning
        members based on the distance probabilities computed by
        self.assign_dist_probs_from_requirements

        Guess always respects the constraints, unless it times out.

        Note:
              This method will try repeated guessing a solution until
              it finds one that respects all the constraints.If it
              times out it returns None.

        Returns:
            np.ndarray: guess of solution vector (None if timeout).
        """

        constraints_met = False
        timeout = time.time() + self.smart_guess_timeout

        while not constraints_met:
            x = np.zeros(5 * len(self.bid_pool))

            worker_dict = {m.worker_id: m
                           for m in self.bid_pool}

            p = list(
                self.assign_dist_probs_from_requirements(
                    workers=self.bid_pool
                ).values()
            )
            size = np.random.randint(
                self.min_team_size, self.max_team_size + 1
            )
            members = Random.weighted_choice(
                list(worker_dict.keys()), size, p=p
            )

            members = [worker_dict[wid] for wid in members]

            for m in members:
                start = self.bid_pool.index(m) * 5

                required_skill_count = len(
                    self.project.required_skills
                )

                add_skills = Random.choices(
                    self.project.required_skills,
                    min(
                        required_skill_count,
                        m.contributions.get_remaining_units(
                            self.project.start_time,
                            self.project.length
                        )
                    )
                )
                for skill in add_skills:
                    si = self.skills.index(skill)
                    x[start + si] = 1

            constraints_met = self.constraints.test(x)
            if time.time() > timeout:
                x = None
                break

        return x


class BHConstraints(object):
    """Constraints on the solution vector (x) for basinhopping
    optimiser, such as ensuring that team size is within the specified
    bounds, budget constraint is met, non-required skills are not
    included etc.

    Note:
        constraint_list is in the format required by Scipy's COBYLA
        optimiser (see the documentation).
    ...

    Attributes:
        optimiser: Basinhopping
            Optimiser to build constraints for.
        constraint_list: list
            Stores the constraints.
        verbose: bool
            Whether to print constraint violation when testing.
    """
    def __init__(self, optimiser, verbose=False):
        self.optimiser = optimiser
        self.constraint_list = self.build_constraints()
        self.verbose = verbose

    def __call__(self, **kwargs):
        """Used by scipy's basinhopping `accept_test` to determine
        if a new hop should be accepted.
        """
        x = kwargs["x_new"]
        return self.test(x)

    def build_constraints(self):
        """Builds constraint list in required format.

        Returns:
            list: of constraints
        """
        constraints = []

        for wi, worker in enumerate(self.optimiser.bid_pool):
            start = wi * 5

            constraints.append({
                'type': 'ineq',
                'fun': (
                    lambda x:
                    self.optimiser.worker_unit_budgets[
                        worker.worker_id
                    ]
                    - sum(np.round(x[start:start + 5]))
                ),
                'name': 'worker_unit_budget_%d' % worker.worker_id
            })

        base_dept_unit_budgets = {
            dept.dept_id: dept.get_remaining_unit_budget(
                self.optimiser.project.start_time,
                self.optimiser.project.length
            )
            for dept in set(
                [m.department for m in self.optimiser.bid_pool]
            )
        }
        dept_ids = base_dept_unit_budgets.keys()
        dept_members = {
            dept.dept_id: [
                m for m in self.optimiser.bid_pool
                if m.department.dept_id == dept.dept_id
            ]
            for dept in set(
                [m.department for m in self.optimiser.bid_pool]
            )
        }

        for dept_id in dept_ids:
            constraints.append({
                'type': 'ineq',
                'fun': (
                    lambda x: self.optimiser.adjusted_dept_unit_budget(
                        x,
                        dept_id, base_dept_unit_budgets,
                        dept_members
                    )
                ),
                'name': 'dept_budget_%d' % dept_id
            })

        constraints.append({
            'type': 'ineq',
            'fun': (
                lambda x:
                self.optimiser.max_team_size
                - self.optimiser.team_size(x)
            ),
            'name': 'team_size_ub'
        })
        constraints.append({
            'type': 'ineq',
            'fun': (
                lambda x:
                self.optimiser.team_size(x)
                - self.optimiser.min_team_size
            )
            , 'name': 'team_size_lb'
        })

        for i in range(5 * len(self.optimiser.bid_pool)):
            constraints.append({
                'type': 'ineq', 'fun': lambda x: x[i], 'name': 'lb_%d' % i
            })
            constraints.append({
                'type': 'ineq', 'fun': lambda x: 1 - x[i], 'name': 'ub_%d' % i
            })

        constraints.append({
            'type': 'ineq', 'fun': (
                lambda x:
                -1 + int(self.optimiser.get_team(x).within_budget())
            ),
            'name': 'budget_constraint'
        })

        for si, skill in enumerate(self.optimiser.skills):
            if skill not in self.optimiser.project.required_skills:

                for i in range(len(self.optimiser.worker_ids)):
                    constraints.append({
                        'type': 'ineq', 'fun': lambda x: 0 - x[i * 5 + si],
                        'name': 'non_required_skill_constraint'
                    })

        return constraints

    def test(self, x):
        """Test all constraints.

        Args:
            x: np.ndarray
                Solution vector to test.

        Returns:
            bool: True if all constraints met.
        """
        for cons in self.constraint_list:
            assert cons['type'] == 'ineq'
            if cons['fun'](x) < 0:
                if self.verbose:
                    print("Constraint violated: %s" % cons['name'])
                return False

        return True


class BHStep(object):
    """Smart stepping used by Basinhopping optimiser.

    Shuffles the team by randomly adding and removing workers to the
    solution vector based on the distance probabilities computed by
    `Basinhopping.assign_dist_probs_from_requirements`, while remaining
    within the min.max team size bounds.

    Note:
         The method will keep randomly suffling the team until it
         arrives at one that respects the constraints. If it does not
         achieve this within the time limit, it returns the old
         solution vector.

    Attributes:
        optimiser: Basinhopping
            Optimiser to which this object belongs.
        bid_pool: list
            List of workers to choose from.
        skills: list
            Indicates which skills are hard skills.
            By default: ['A', 'B', 'C', 'D', 'E']
        project: project.Project
            Project that a team is being selected for.
        worker_unit_budgets: dict
            Records how many units each worker has available.
        max_team_size: int
            Maximum team size.
        min_team_size: int
            Minimum team size
        time_limit: int
            Number of seconds after which the method times out and
            returns the old solution vector instead of a new one to
            step to.
    """

    def __init__(self, optimiser,
                 max_team_size=MAX_TEAM_SIZE,
                 min_team_size=MIN_TEAM_SIZE,
                 time_limit=1):

        self.optimiser = optimiser
        self.bid_pool = optimiser.bid_pool
        self.skills = ['A', 'B', 'C', 'D', 'E']
        self.project = optimiser.project
        self.worker_unit_budgets = optimiser.worker_unit_budgets

        self.max_team_size = max_team_size
        # team needs at least as many members as the maximum required
        # skill units:
        self.min_team_size = max(
            [self.project.get_skill_requirement(skill)['units']
             for skill in self.project.required_skills]
        )
        self.min_team_size = max(self.min_team_size, min_team_size)
        self.time_limit = time_limit

    def __call__(self, x):
        """Called by basinhopping on each hop.

        Args:
            x: np.ndarray
                Current solution vector

        Returns:
            np.ndarray: new solution vector to step to (if method
                        times out, this is equal to x).
        """
        constraints_met = False
        old_x = x
        timeout = time.time() + self.time_limit

        while not constraints_met:

            # determine how many workers to add and remove to team:
            number_to_add = min(
                Random.randint(0, self.max_team_size),
                len(self.bid_pool) - self.optimiser.team_size(x)
            )
            number_to_add = max(0, number_to_add)

            new_size = self.optimiser.team_size(x) + number_to_add
            min_remove = max(0, new_size - self.min_team_size)
            max_remove = new_size - self.min_team_size
            if max_remove < min_remove:
                number_to_remove = 0
            else:
                number_to_remove = Random.randint(
                    min_remove, max_remove
                )

            # choose members to add:
            assert (len(self.bid_pool)
                    >= number_to_add + self.optimiser.team_size(x))
            current_team = self.optimiser.get_team(x)

            new_team_members = list(current_team.members.values())
            choose_from = [bid for bid in self.bid_pool
                           if bid not in new_team_members]
            p = list(self.optimiser.assign_dist_probs_from_requirements(
                workers=choose_from
            ).values())
            to_add = Random.weighted_choice(choose_from, number_to_add, p=p)

            for a in to_add:
                new_team_members.append(a)
                start = self.bid_pool.index(a) * 5

                required_skill_count = len(self.project.required_skills)
                add_skills = Random.choices(
                    self.project.required_skills,
                    min(required_skill_count,
                        self.worker_unit_budgets[a.worker_id])
                )
                for skill in add_skills:
                    si = self.skills.index(skill)
                    x[start + si] = 1

            # now select and remove required number of workers:
            to_remove = Random.choices(new_team_members, number_to_remove)
            for r in to_remove:
                start = self.bid_pool.index(r) * 5
                for i in range(5):
                    x[start + i] = 0

            constraints_met = self.optimiser.constraints.test(x)
            if time.time() > timeout:
                x = old_x
                break
        return x
