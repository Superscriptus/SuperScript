from .organisation import Team
from .utilities import Random
from .config import MAX_TEAM_SIZE, MIN_TEAM_SIZE

from scipy.optimize import basinhopping
from scipy.spatial import minkowski_distance
import numpy as np
import pandas as pd
import time
import pickle

## TODO:
# - remove save functionality (and results_Dir) (and exp_number)
# - move imports of MIN and MAX_TEAM_SIZE to main.py (pass via factory)
# - clean up constraints
# - write unit tests
# - comment on use of Paths in docs (use of non-pickleable lambda functions and class methods)
# - it is possible for the new takestep to remove members that have just been added. Prevent this?


class OptimiserFactory:

    @staticmethod
    def get(optimiser_name, project, bid_pool, model):
        if optimiser_name == "ParallelBasinhopping":
            return Optimiser(project, bid_pool, model, 0)


class Optimiser:

    def __init__(self, project, bid_pool, model, exp_number,
                 verbose=False, save_flag=False,
                 results_dir='developing_model_spec/experiments/optimisation/',
                 min_team_size=MIN_TEAM_SIZE,
                 max_team_size=MAX_TEAM_SIZE):

        self.project = project
        self.bid_pool = bid_pool
        self.model = model
        self.exp_number = exp_number
        self.skills = ['A', 'B', 'C', 'D', 'E']
        self.worker_ids = [m.worker_id for m in bid_pool]
        self.worker_unit_budgets = self.get_worker_units_budgets()
        self.constraints = self.build_constraints()
        self.verbose = verbose
        self.save_flag = save_flag
        self.results_dir = results_dir
        self.min_team_size = min_team_size
        self.max_team_size = max_team_size

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

        contributions = dict()
        # for skill in project.required_skills:
        for skill in self.skills:
            contributions[skill] = []

        team_members = {}
        for wi, worker_id in enumerate(self.worker_ids):

            start = wi * 5
            in_team = False

            # for si, skill in enumerate(project.required_skills):
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
        test_team = self.get_team(x)

        self.project.team = test_team
        self.model.inventory.success_calculator.calculate_success_probability(
            self.project
        )

        return -self.project.success_probability

    def get_worker_units_budgets(self):
        return {
            worker.worker_id: worker.contributions.get_remaining_units(
                self.project.start_time, self.project.length
            )
            for worker in self.bid_pool
        }

    def build_constraints(self):
        constraints = []

        for wi, worker in enumerate(self.bid_pool):
            start = wi * 5

            constraints.append({
                'type': 'ineq',
                'fun': (lambda x:
                        self.worker_unit_budgets[worker.worker_id]
                        - sum(np.round(x[start:start + 5]))
                        ),
                'name': 'worker_unit_budget_%d' % worker.worker_id
            })

        base_dept_unit_budgets = {
            dept.dept_id: dept.get_remaining_unit_budget(
                self.project.start_time, self.project.length
            )
            for dept in set(
                [m.department for m in self.bid_pool]
            )
        }
        dept_ids = base_dept_unit_budgets.keys()
        dept_members = {
            dept.dept_id: [m for m in self.bid_pool if m.department.dept_id == dept.dept_id]
            for dept in set(
                [m.department for m in self.bid_pool]
            )
        }

        for dept_id in dept_ids:
            constraints.append({
                'type': 'ineq',
                'fun': lambda x: self.adjusted_dept_unit_budget(
                    x, dept_id, base_dept_unit_budgets, dept_members
                ),
                'name': 'dept_budget_%d' % dept_id
            })

        constraints.append({
            'type': 'ineq', 'fun': (lambda x:
                                    self.max_team_size - self.team_size(x)),
            'name': 'team_size_ub'
        })
        constraints.append({
            'type': 'ineq', 'fun': (lambda x:
                                    self.team_size(x) - self.min_team_size)
            , 'name': 'team_size_lb'
        })

        for i in range(5 * len(self.bid_pool)):
            constraints.append({
                'type': 'ineq', 'fun': lambda x: x[i], 'name': 'lb_%d' % i
            })
            constraints.append({
                'type': 'ineq', 'fun': lambda x: 1 - x[i], 'name': 'ub_%d' % i
            })

        constraints.append({
            'type': 'ineq', 'fun': (
                lambda x:
                -1 + int(self.get_team(x).within_budget())
            ),
            'name': 'budget_constraint'
        })

        for si, skill in enumerate(self.skills):
            if skill not in self.project.required_skills:

                for i in range(len(self.worker_ids)):
                    constraints.append({
                        'type': 'ineq', 'fun': lambda x: 0 - x[i * 5 + si],
                        'name': 'non_required_skill_constraint'
                    })

        return constraints

    def adjusted_dept_unit_budget(self, x, dept_id,
                                  base_dept_unit_budgets, dept_members):

        base = base_dept_unit_budgets[dept_id]
        for m in dept_members[dept_id]:
            start = self.bid_pool.index(m)
            base -= sum(np.round(x[start:start + 5]))

        return base

    def test_constraints(self, x, verbose=False):

        for cons in self.constraints:
            assert cons['type'] == 'ineq'
            if cons['fun'](x) < 0:
                if verbose:
                    print("Constraint violated: %s" % cons['name'])
                return False

        return True

    def team_size(self, x):

        size = 0
        for wi, worker_id in enumerate(self.worker_ids):
            start = wi * 5
            if sum(np.round(x[start:start + 5])) > 0:
                size += 1

        return size

    def solve(self, guess, niter, repeat,
              maxiter = 100, catol = 0.0, rhobeg = 0.6):

        minimizer_kwargs = {"method": 'COBYLA',
                            'constraints': self.constraints,
                            'options': {'maxiter': maxiter, 'disp': False,
                                        'catol': catol, 'rhobeg': rhobeg}}

        my_constraints = MyConstraints(self)
        my_takestep = MyTakeStep(self)

        start_time = time.time()
        ret = basinhopping(self.objective_func, guess,
                           minimizer_kwargs=minimizer_kwargs,
                           niter=niter, seed=70470,
                           accept_test=my_constraints,
                           take_step=my_takestep)
                           #callback=print_fun,)

        elapsed_time = time.time() - start_time
        if self.verbose:
            print("%d iterations took %.2f seconds" % (niter, elapsed_time))

        best_team = self.get_team(ret.x)
        self.project.team = best_team
        self.model.inventory.success_calculator.calculate_success_probability(
            self.project
        )
        if self.save_flag:
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

        return elapsed_time, ret

    def compute_distances_from_requirements(self, project=None,
                                            workers=None, p=2):

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
            minkowski_distance(row[project.required_skills],
                               required_levels, p)
            for ri, row in worker_table.iterrows()
        ]
        worker_table['prob'] = [(1 / d) if d > 0 else 0
                                for d in worker_table.distance]
        #1 / worker_table.distance
        if sum(worker_table['prob']) > 0:
            worker_table['prob'] /= sum(worker_table['prob'])

        worker_table.sort_values('prob', ascending=False, inplace=True)

        return dict(zip(worker_table.id, worker_table.prob))

    def smart_guess(self, p=2):

        x = np.zeros(5 * len(self.bid_pool))

        worker_table = pd.DataFrame()
        worker_dict = {m.worker_id: m
                       for m in self.bid_pool}
        worker_table['id'] = worker_dict.keys()

        for skill in self.project.required_skills:
            worker_table[skill] = [
                m.get_skill(skill) for m in self.bid_pool
            ]

        required_levels = [
            self.project.requirements.hard_skills[skill]['level']
            for skill in self.project.required_skills
        ]

        worker_table['distance'] = [
            minkowski_distance(row[self.project.required_skills],
                               required_levels, p)
            for ri, row in worker_table.iterrows()
        ]
        worker_table['prob'] = [(1 / d) if d > 0 else 0
                                for d in worker_table.distance]
        if sum(worker_table['prob']) > 0:
            worker_table['prob'] /= sum(worker_table['prob'])

        worker_table.sort_values('prob', ascending=False, inplace=True)

        size = np.random.randint(self.min_team_size, self.max_team_size + 1)
        members = np.random.choice(
            worker_table.id, size=size, replace=False, p=worker_table.prob
        )
        members = [worker_dict[wid] for wid in members]

        for m in members:
            start = self.bid_pool.index(m) * 5

            required_skill_count = len(self.project.required_skills)
            add_skills = Random.choices(
                self.project.required_skills,
                min(required_skill_count,
                    m.contributions.get_remaining_units(
                        self.project.start_time, self.project.length)
                    )
            )
            for skill in add_skills:
                si = self.skills.index(skill)
                x[start + si] = 1

        return x


class MyConstraints(object):

    def __init__(self, optimiser):
        self.test = optimiser.test_constraints

    def __call__(self, **kwargs):
        x = kwargs["x_new"]
        return self.test(x)


class MyTakeStep(object):

    def __init__(self, optimiser,
                 max_team_size=MAX_TEAM_SIZE,
                 min_team_size=MIN_TEAM_SIZE):

        self.optimiser = optimiser
        self.bid_pool = optimiser.bid_pool
        self.skills = ['A', 'B', 'C', 'D', 'E']
        self.project = optimiser.project
        self.worker_unit_budgets = optimiser.worker_unit_budgets

        self.max_team_size = max_team_size
        # team needs at least as many members as the maximum required skill units:
        self.min_team_size = max(
            [self.project.get_skill_requirement(skill)['units']
             for skill in self.project.required_skills]
        )
        self.min_team_size = max(self.min_team_size, min_team_size)

    def __oldcall__(self, x):

        # determine how many workers to add and remove to team:
        number_to_add = min(Random.randint(0, MAX_TEAM_SIZE),
                            len(self.bid_pool) - self.optimiser.team_size(x))

        new_size = self.optimiser.team_size(x) + number_to_add
        min_remove = max(0, new_size - MAX_TEAM_SIZE)
        max_remove = new_size - MIN_TEAM_SIZE
        if max_remove < min_remove:
            number_to_remove = 0
        else:
            number_to_remove = Random.randint(min_remove, max_remove)

        # choose members to add:
        assert len(self.bid_pool) >= number_to_add + number_to_remove
        current_team = self.optimiser.get_team(x)

        to_add = []
        new_team_members = list(current_team.members.values())
        inverted = self.bid_pool[::-1]
        for i in range(number_to_add):
            chosen = Random.choice(inverted)
            while chosen in new_team_members:
                chosen = Random.choice(inverted)
            new_team_members.append(chosen)
            to_add.append(chosen)

        for a in to_add:
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

        # now select and remove required numer of workers:
        to_remove = Random.choices(new_team_members, number_to_remove)
        for r in to_remove:
            start = self.bid_pool.index(r) * 5
            for i in range(5):
                x[start + i] = 0

        return x

    def __call__(self, x):

        # determine how many workers to add and remove to team:
        number_to_add = min(Random.randint(0, MAX_TEAM_SIZE),
                            len(self.bid_pool) - self.optimiser.team_size(x))

        new_size = self.optimiser.team_size(x) + number_to_add
        min_remove = max(0, new_size - MAX_TEAM_SIZE)
        max_remove = new_size - MIN_TEAM_SIZE
        if max_remove < min_remove:
            number_to_remove = 0
        else:
            number_to_remove = Random.randint(min_remove, max_remove)

        # choose members to add:
        #assert len(self.bid_pool) >= number_to_add + number_to_remove
        assert len(self.bid_pool) >= number_to_add + self.optimiser.team_size(x)
        current_team = self.optimiser.get_team(x)

        to_add = []
        new_team_members = list(current_team.members.values())
        choose_from = [bid for bid in self.bid_pool
                       if bid not in new_team_members]
        p = list(self.optimiser.compute_distances_from_requirements(
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

        return x
