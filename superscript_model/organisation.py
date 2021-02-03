from interface import Interface, implements
from itertools import combinations
from numpy import percentile
import json

from .project import Project
from .utilities import Random
from .config import (TEAM_OVR_MULTIPLIER,
                     MIN_TEAM_SIZE,
                     MAX_TEAM_SIZE,
                     PRINT_DECIMALS_TO,
                     MAX_SKILL_LEVEL,
                     MIN_SOFT_SKILL_LEVEL,
                     SOFT_SKILLS,
                     DEPARTMENTAL_WORKLOAD,
                     WORKLOAD_SATISFIED_TOLERANCE,
                     UNITS_PER_FTE,
                     TRAINING_LENGTH,
                     HARD_SKILLS)


class Team:

    def __init__(self, project, members,
                 lead, round_to=PRINT_DECIMALS_TO,
                 soft_skills=SOFT_SKILLS,
                 contributions=None):
        self.project = project
        self.members = members
        self.lead = lead
        #self.assign_lead(self.project)
        self.round_to = round_to
        self.soft_skills = soft_skills  # used by compute_creativity_match()

        if contributions is None:
            # currently this is automatic, but could be handled by TeamAllocator:
            self.contributions = self.determine_member_contributions()
        else:
            self.contributions = contributions

        self.team_ovr = self.compute_ovr()
        self.skill_balance = self.compute_skill_balance()
        self.creativity_match = self.compute_creativity_match()

    @property
    def size(self):
        return len(self.members.keys())

    def assign_lead(self, project):
        if self.lead is not None:
            self.lead.assign_as_lead(project)

    def remove_lead(self, project):
        if self.lead is not None:
            self.lead.remove_as_lead(project)
            self.lead = None

    def compute_ovr(self, multiplier=TEAM_OVR_MULTIPLIER):

        skill_count = 0
        ovr = 0
        for skill in self.project.required_skills:

            workers = self.contributions[skill]
            for worker_id in workers:
                ovr += self.members[worker_id].get_skill(skill)
                skill_count += 1

        if skill_count > 0:
            return multiplier * ovr / skill_count
        else:
            return 0.0

    def rank_members_by_skill(self, skill):

        ranked_members = {
            member[0]: member[1].get_skill(skill)
            for member in self.members.items()
        }
        return {
            k: v for k, v in sorted(ranked_members.items(),
                                    reverse=True,
                                    key=lambda item: item[1])
        }

    def determine_member_contributions(self):

        dept_unit_budgets = {
            dept.dept_id: dept.get_remaining_unit_budget(
                self.project.start_time, self.project.length
            )
            for dept in set(
                [m.department for m in self.members.values()]
            )
        }
        member_unit_budgets = {
            member.worker_id: member.contributions.get_remaining_units(
                self.project.start_time, self.project.length
            )
            for member in self.members.values()
        }

        contributions = dict()
        for skill in self.project.required_skills:

            ranked_members = self.rank_members_by_skill(skill)
            skill_requirement = self.project.get_skill_requirement(skill)

            contributions[skill] = []
            unit_count = 0
            for member_id in ranked_members.keys():
                if unit_count >= skill_requirement['units']:
                    break

                member = self.members[member_id]
                if (dept_unit_budgets[member.department.dept_id] > 0
                        and member_unit_budgets[member.worker_id] > 0):
                    contributions[skill].append(member_id)
                    dept_unit_budgets[member.department.dept_id] -= 1
                    member_unit_budgets[member.worker_id] -= 1
                    unit_count += 1

        #self.assign_contributions_to_members(contributions)  # this will be called elsewhere

        return contributions

    def assign_contributions_to_members(self):

        for member_id in self.members.keys():

            units_contributed_by_member = 0
            for skill in self.contributions.keys():

                if member_id in self.contributions[skill]:
                    self.members[member_id].contributions.add_contribution(
                        self.project, skill
                    )
                    units_contributed_by_member += 1

            (self.members[member_id]
                .department.update_supplied_units(
                units_contributed_by_member, self.project
            ))

    def compute_skill_balance(self):

        skill_balance = 0
        number_with_negative_differences = 0
        for skill in self.project.required_skills:

            required_units = (
                self.project.get_skill_requirement(skill)['units']
            )
            required_level = (
                self.project.get_skill_requirement(skill)['level']
            )
            worker_skills = [
                self.members[worker_id].get_skill(skill)
                for worker_id in self.contributions[skill]
            ]
            skill_mismatch = (
                    (sum(worker_skills) / required_units) - required_level
            ) if required_units > 0 else 0

            if skill_mismatch < 0:
                skill_balance += skill_mismatch ** 2
                number_with_negative_differences += 1

        if number_with_negative_differences > 0:
            return skill_balance / number_with_negative_differences
        else:
            return 0

    def compute_creativity_match(self,
                                 max_skill_level=MAX_SKILL_LEVEL,
                                 min_skill_level=MIN_SOFT_SKILL_LEVEL):

        creativity_level = 0
        number_of_existing_skills = 0
        max_distance = max_skill_level - min_skill_level
        if len(self.members.keys()) > 1:
            max_distance /= (len(self.members.keys()) - 1)

        for skill in self.soft_skills:

            worker_skills = [
                member.get_skill(
                    skill, hard_skill=False
                )
                for member in self.members.values()
            ]

            pairs = list(combinations(worker_skills, 2))
            if len(pairs) > 0:
                creativity_level += (
                        sum([((p[1] - p[0]) / max_distance) ** 2
                             for p in pairs])
                        / len(pairs)
                )
                number_of_existing_skills += 1

        if number_of_existing_skills > 0:
            creativity_level /= number_of_existing_skills
        else:
            creativity_level = 0

        creativity_level = (creativity_level * max_distance) + 1
        return (self.project.creativity - creativity_level) ** 2

    def skill_update(self, success, skill_update_func):

        if success:
            modifier = skill_update_func.get_values(self.project.risk)
        else:
            modifier = skill_update_func.get_values(0)

        for skill, workers in self.contributions.items():
            for worker_id in workers:
                self.members[worker_id].peer_assessment(
                    success, skill, modifier
                )

    def log_project_outcome(self, success):

        if success:
            self.lead.model.grid.add_team_edges(self)
        for member in self.members.values():
            member.history.record(success)

    def within_budget(self):
        if self.project.budget is None:
            return True
        elif self.team_ovr <= self.project.budget:
            return True
        else:
            return False

    def to_string(self):

        output = {
            'project': self.project.project_id,
            'members': list(self.members.keys()),
            'lead': (self.lead.worker_id
                     if self.lead is not None
                     else None),
            'success_probability': round(
                self.project.success_probability, self.round_to
            ),
            'team_ovr': round(self.team_ovr, self.round_to),
            'skill_balance': round(self.skill_balance, self.round_to),
            'creativity_match': round(
                self.creativity_match, self.round_to
            ),
            'skill_contributions': self.contributions
        }
        return json.dumps(output, indent=4)


class OrganisationStrategyInterface(Interface):

    def invite_bids(self, project: Project) -> list:
        pass

    def select_team(self, project: Project,
                    bid_pool=None) -> Team:
        pass


class RandomStrategy(implements(OrganisationStrategyInterface)):

    def __init__(self, model,
                 min_team_size=MIN_TEAM_SIZE,
                 max_team_size=MAX_TEAM_SIZE):
        self.model = model
        self.min_team_size = min_team_size
        self.max_team_size = max_team_size

    def invite_bids(self, project: Project) -> list:

        bid_pool = [
            worker for worker in self.model.schedule.agents
            if worker.bid(project)
        ]
        return bid_pool

    def select_team(self, project: Project,
                    bid_pool=None) -> Team:

        size = Random.randint(self.min_team_size,
                              self.max_team_size)
        bid_pool = (self.model.schedule.agents
                    if bid_pool is None else bid_pool)

        if size > len(bid_pool):
            # print("Cannot select %d workers from "
            #      "bid_pool of size %d for project %d"
            #      % (size, len(bid_pool), project.project_id))
            workers = {}
            lead = None

        else:
            workers = {worker.worker_id: worker
                       for worker in
                       Random.choices(bid_pool, size)}
            lead = Random.choice(list(workers.values()))

        return Team(project, workers, lead)


class BasicStrategy(implements(OrganisationStrategyInterface)):

    def __init__(self, model,
                 min_team_size=MIN_TEAM_SIZE,
                 max_team_size=MAX_TEAM_SIZE):
        self.model = model
        self.min_team_size = min_team_size
        self.max_team_size = max_team_size

    def invite_bids(self, project: Project) -> list:

        bid_pool = [
            worker for worker in self.model.schedule.agents
            if worker.bid(project)
        ]
        return bid_pool

    def select_team(self, project: Project,
                    bid_pool=None) -> Team:

        bid_pool = (self.model.schedule.agents
                    if bid_pool is None else bid_pool)

        if len(bid_pool) < self.min_team_size:
            return Team(project, {}, None)
        else:
            return self.select_top_n(bid_pool, project)

    def rank_bids(self, bid_pool, project):

        ranked_bids = {}
        for worker in bid_pool:
            ranked_bids[worker.worker_id] = sum([
                worker.get_skill(skill)
                for skill in project.required_skills
            ])

        ranked_bids = {
            k: v for k, v in sorted(
                ranked_bids.items(),
                reverse=True,
                key=lambda item: item[1]
            )
        }
        return ranked_bids

    def select_top_n(self, bid_pool, project):
        worker_dict = {worker.worker_id: worker
                       for worker in bid_pool}
        ranked_bids = self.rank_bids(bid_pool, project)

        team_size = 0
        workers = {}
        team = Team(project, {}, None)
        for worker_id in ranked_bids.keys():
            workers[worker_id] = worker_dict[worker_id]
            test_team = Team(project, workers, workers[worker_id])
            if test_team.within_budget():
                team = test_team
                team_size += 1
            else:
                del workers[worker_id]

            if team_size >= self.max_team_size:
                break

        if team_size <= self.min_team_size:
            team = Team(project, {}, None)

        return team


class TeamAllocator:

    def __init__(self, model):
        self.model = model
        self.strategy = (RandomStrategy(model)
                         if self.model.organisation_strategy == "Random"
                         else BasicStrategy(model))

    def allocate_team(self, project: Project):
        bid_pool = self.strategy.invite_bids(project)
        team = self.strategy.select_team(
            project, bid_pool=bid_pool
        )

        # if team is not None and project.budget is not None:
        #     if team.team_ovr > project.budget:
        #         team.remove_lead(project)
        #         team = None
        #     else:
        #         team.assign_contributions_to_members()
        if team is not None and team.lead is not None:
            if team.within_budget():
                team.assign_contributions_to_members()
                team.assign_lead(project)
            else:
                team = None

        project.team = team


class Trainer:

    def __init__(self, model,
                 training_length=TRAINING_LENGTH,
                 max_skill_level=MAX_SKILL_LEVEL,
                 hard_skills=HARD_SKILLS):

        self.model = model
        self.training_length = training_length
        self.max_skill_level = max_skill_level
        self.hard_skills = hard_skills
        self.skill_quartiles = dict(zip(hard_skills, []))
        self.trainees = dict()

    def top_two_demanded_skills(self):
        return self.model.inventory.top_two_skills

    def update_skill_quartiles(self):
        for skill in self.hard_skills:
            self.skill_quartiles[skill] = percentile(
                [worker.get_skill(skill)
                 for worker in self.model.schedule.agents],
                [25, 50, 75]
            )

    def train(self):

        if (self.model.training_on
                and self.model.schedule.steps
                >= self.model.training_commences):

            self.advance_training()

            if self.model.training_mode == 'all':
                self.add_all_for_training()
            elif self.model.training_mode == 'slots':
                self.add_slots_for_training()
            else:
                print("Training mode not recognised!")
                pass

    def advance_training(self):

        worker_ids = list(self.trainees.keys())
        for worker_id in worker_ids:

            if worker_id in self.trainees.keys():
                self.trainees[worker_id].training_remaining -= 1
                # trainee.training_remaining = max(trainee.training_remaining, 0)
                if self.trainees[worker_id].training_remaining == 0:
                    del self.trainees[worker_id]

    @staticmethod
    def worker_free_to_train(worker):
        return worker.is_free(worker.now, worker.training_horizon,
                              slack=worker.contributions.units_per_full_time)

    def add_slots_for_training(self):

        skillA = self.top_two_demanded_skills()[0]
        skillB = self.top_two_demanded_skills()[1]
        sorted_workers = dict()

        for worker in self.model.schedule.agents:
            skill = (
                skillA
                if worker.get_skill(skillA) < worker.get_skill(skillB)
                else skillB
            )
            skill_value = worker.get_skill(skill)
            if skill_value < self.skill_quartiles[skill][1]:
                sorted_workers[worker.worker_id] = (skill, skill_value)

        sorted_workers = {
            k: v for k, v in sorted(
                sorted_workers.items(),
                reverse=False,
                key=lambda item: item[1][1]
            )
        }

        slots = 0
        target_slots = (
                (self.model.target_training_load
                 * self.model.worker_count)
                / (self.training_length * 100)
        )
        for worker_id in sorted_workers.keys():
            worker = self.model.schedule._agents[worker_id]

            if self.worker_free_to_train(worker):
                self.add_worker(worker, sorted_workers[worker_id][0])
                slots += 1
                if slots >= target_slots:
                    break

    def add_all_for_training(self):

        for worker in self.model.schedule.agents:
            requires_training = False

            if self.worker_free_to_train(worker):

                for skill in self.top_two_demanded_skills():
                    if worker.get_skill(skill) < self.skill_quartiles[skill][1]:
                        requires_training = True
                        skill_to_train = skill

                if requires_training:
                    self.add_worker(worker, skill_to_train)

    def add_worker(self, worker, skill_to_train):

        self.update_worker_skill(worker, skill_to_train)

        self.trainees[worker.worker_id] = worker
        worker.department.add_training(worker, self.training_length)
        worker.training_remaining = self.training_length
        for t in range(self.training_length):
            worker.contributions.total_contribution[worker.now + t] = (
                worker.contributions.units_per_full_time
            )

    def update_worker_skill(self, worker, skill_to_train):

        new_skill = min(
            self.skill_quartiles[skill_to_train][2],
            self.max_skill_level
        )
        worker.skills.hard_skills[skill_to_train] = new_skill


class Department:

    def __init__(self, dept_id,
                 workload=DEPARTMENTAL_WORKLOAD,
                 units_per_full_time=UNITS_PER_FTE,
                 tolerance=WORKLOAD_SATISFIED_TOLERANCE):

        self.dept_id = dept_id
        self.number_of_workers = 0
        self.workload = workload
        self.units_per_full_time = units_per_full_time
        self.tolerance = tolerance
        self.units_supplied_to_projects = dict()
        self.maximum_project_units = 0

    def update_supplied_units(self, units_contributed, project):

        for time_offset in range(project.length):
            time = project.start_time + time_offset

            if time not in self.units_supplied_to_projects.keys():
                self.units_supplied_to_projects[time] = units_contributed
            else:
                self.units_supplied_to_projects[time] += units_contributed

    def add_worker(self):
        self.number_of_workers += 1
        self.evaluate_maximum_project_units()

    def add_training(self, worker, length):

        start = worker.now
        units = self.units_per_full_time
        for time_offset in range(length):
            time = start + time_offset

            if time not in self.units_supplied_to_projects.keys():
                self.units_supplied_to_projects[time] = units
            else:
                self.units_supplied_to_projects[time] += units

    def evaluate_maximum_project_units(self):
        # Note that this is only called by add_worker()
        # the assumption that none of the other values change during simulation
        total_units_dept_can_supply = (
                self.number_of_workers * self.units_per_full_time
        )
        departmental_workload_units = (
                total_units_dept_can_supply * self.workload
        )
        self.maximum_project_units = (
                total_units_dept_can_supply - departmental_workload_units
        )

    def units_supplied_to_projects_at_time(self, time):
        return (
            self.units_supplied_to_projects[time]
        ) if time in self.units_supplied_to_projects.keys() else 0

    def is_workload_satisfied(self, start, length, tolerance=None):

        tolerance = self.tolerance if tolerance is None else tolerance
        for t in range(length):

            time = start + t
            if (self.units_supplied_to_projects_at_time(time)
                    >= (self.maximum_project_units - self.tolerance)):
                return False

        return True

    def get_remaining_unit_budget(self, start, length):

        budget_over_time = []
        for t in range(length):
            budget_over_time.append(
                self.maximum_project_units
                - self.units_supplied_to_projects_at_time(start + t)
                - self.tolerance
            )
        return min(budget_over_time)

    def to_string(self):
        output = {
            'dept_id': self.dept_id,
            'number_of_workers': self.number_of_workers,
            'workload': self.workload,
            'units_per_full_time': self.units_per_full_time,
            'tolerance': self.tolerance
        }
        return json.dumps(output, indent=4)
