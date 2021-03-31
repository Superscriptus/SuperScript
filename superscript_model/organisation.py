"""
SuperScript organisation module
===========

Classes:
    Team
        Team of workers, allocated to a project
    OrganisationStrategyInterface
        Interface class that defines how organisation strategies should
        behave. Actual strategies implement this interface, and
        determine how teams are allocated.
    RandomStrategy
        Strategy for random team allocation.
    BasicStrategy
        Naive strategy that aims to imrpove on random allocation.
    ParallelBasinhopping
        Optimisation method for team allocation that uses basinhopping
        with COBYLA optimisation at each hop. Can leverage multiple
        cores to run parallel optimisations and take the best result.
    TeamAllocator
        Class that handles team allocation.
    Trainer
        Class handles training of low skilled workers, with various
        options for how workers are trained.
    Department
        Department class. Each worker is registered in a department
        instance, which handles departmental workload and determines
        how much capacity each of its workers has to contribute to
        projects.
"""
from interface import Interface, implements
from itertools import combinations
from numpy import percentile, argmax
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
    """Team class.

        Note:
            During team allocation, trail teams may be created while
            trying to find the best team for the project (depending
            on which organisation strategy is in use).
        ...

        Attributes:
            project: project.Project
                The project to which this team is assigned
            members: dict
                Dictionary of workers that are members of this team.
                Key is worker_id, value is worker.Worker
            lead: worker.Worker
                Team lead. Responsible for advancing the project on each
                timestep.
            round_to: int
                Number of decimal places to round to when printing.
            soft_skills: list
                List that defines which skills are soft skills
                By default it is ['F', 'G', 'H', 'I', 'J']
            contributions: dict
                Records the contributions, by hard skill, that each
                team member makes to the project.
                Either determined externally during team allocation and
                passed into constructor, or determined internally
                during construction.
            team_ovr: float
                Team OVR value, used to determine probability of
                success.
            team_budget: float
                Team budget, used to determine if team is viable (when
                budget functionality is switched on at the simulation
                level). i.e. Is team within project budget.
            skill_balance: float
                Aka 'degree of skill match' captures the deficiency in
                hard skill provision by this team in relation to the
                project skill requirements.
            creativity_level: float
                Creativity level of the team (Captures 'cognitive
                diversity')
            creativity_match: float
                Captures how close the creativity level of this team is
                to the required creativity level of the project.
    """
    def __init__(self, project, members,
                 lead, round_to=PRINT_DECIMALS_TO,
                 soft_skills=SOFT_SKILLS,
                 contributions=None):
        self.project = project
        self.members = members
        self.lead = lead
        self.round_to = round_to
        self.soft_skills = soft_skills

        if contributions is None:
            self.contributions = self.determine_member_contributions()
        else:
            self.contributions = contributions

        # check that team leader actually contributes to project:
        # (otherwise worker may be replaced while still leading project)
        if (
                self.lead is not None
                and self.count_units_contributed_by_member(
                  self.lead.worker_id) == 0
        ):

            contributing_members = [
                m for m in self.members.values()
                if self.count_units_contributed_by_member(m.worker_id) > 0
            ]
            self.lead = Random.choice(contributing_members)

        self.team_ovr = self.compute_ovr()
        self.team_budget = self.compute_team_budget()
        self.skill_balance = self.compute_skill_balance()
        self.creativity_level = None
        self.creativity_match = self.compute_creativity_match()

    @property
    def size(self):
        """Returns size of team (int)."""
        return len(self.members.keys())

    def assign_lead(self):
        """Registers with the chosen lead worker that they are the
        leader of the team (and therefore responsible for
        progressing the project).

        Note:
            This is not called during construction of the team, because
            it is not desirable to assign and then un-assign the lead
            when creating trial teams during optimisation. It is only
            called during team allocation once the chosen team has been
            finalised.
        """
        if self.lead is not None:
            self.lead.assign_as_lead(self.project)

    def remove_lead(self):
        """Un-assign the team lead from their role."""
        if self.lead is not None:
            self.lead.remove_as_lead(self.project)
            self.lead = None

    def compute_ovr(self, multiplier=TEAM_OVR_MULTIPLIER):
        """Compute the team OVR

        Args:
            multiplier: int
                Multiplier for OVR calc. Default is 20.

        Returns:
            float: team OVR value
        """
        ovr = self.compute_team_budget()

        total_required_units = sum(
            [self.project.get_skill_requirement(skill)['units']
             for skill in self.project.required_skills]
        )
        if total_required_units > 0:
            return multiplier * ovr / total_required_units
        else:
            return 0.0

    def compute_team_budget(self):
        """Computes team skill budget, which is equal to the sum of the
        worker skill levels that are actively used by the project.

        Note:
             This method has safety feature in case self.contributions
             exceeds what is required by the project e.g. the project
             only requires 2 units of skill 'A' at level 2, but the
             team is contributing 3 units of skill 'A'.

             In this case the method only uses the top 2 of those 3
             units by skill level when computing the budget.

             This should not happen if member contributions are
             determined correctly.

        Returns:
            float: team budget value
        """
        skill_sum = 0
        for skill in self.project.required_skills:
            required_units = (
                self.project.get_skill_requirement(skill)['units']
            )
            workers = self.contributions[skill]
            skill_levels = [
                self.members[worker_id].get_skill(skill)
                for worker_id in workers
            ]
            skill_levels.sort(reverse=True)
            skill_sum += sum(skill_levels[0:required_units])

        return skill_sum

    def rank_members_by_skill(self, skill):
        """Ranks the team members by their skill level for a specific
        skill, in descending order.

        Args:
            skill: str
                Hard skill to rank by.
                Takes value in ['A',..., 'E']

        Returns:
            dict: ranked members
        """
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
        """Determines the skill contributions of team members to the
        project. Only called if this is not determined externally
        during team allocation (e.g. is RandomStrategy is in use).

        For each required skill, the members are ranked by skill level,
        and this method takes the top N available workers, where N is
        the number if units required by the project for that skill.
        Availability is determined by departmental workload and the
        worker having spare capacity to contribute for the duration
        of the project.

        Note:
            This method does not respect budgetary constraint, so it
            often assigns contributions that exceed the project budget,
            making the team invalid.

        Returns:
            dict: member skill contributions
        """
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

        return contributions

    def count_units_contributed_by_member(self, member_id):
        """Counts the number of units contributed to the project by an
        individual member.

        Args:
            member_id: int
                worker_id of the member to count contributions

        Returns: int
            Number of units contributed by this member.
        """
        units_contributed_by_member = 0
        for skill in self.contributions.keys():
            if member_id in self.contributions[skill]:
                units_contributed_by_member += 1

        return units_contributed_by_member

    def assign_contributions_to_members(self):
        """Registers which each individual member the skills that they
        are to contribute to this project.

        Note:
            As with team.assign_lead(), this is note called on
            construction of the team, but only when team allocation
            has been finalised. This is because we do not want to
            have to assign and un-assign contributions to members when
            creating trial teams during optimisation.

        """
        for member_id in self.members.keys():

            units_contributed_by_member = 0
            for skill in self.contributions.keys():

                if member_id in self.contributions[skill]:
                    (
                        self.members[member_id]
                        .contributions
                        .add_contribution(self.project, skill)
                    )
                    units_contributed_by_member += 1

            (self.members[member_id]
                .department.update_supplied_units(
                units_contributed_by_member, self.project
            ))

    def compute_skill_balance(self):
        """Computes the skill balance aka 'degree of skill match'

        Returns:
             float: skill balance value
        """
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
            worker_skills.sort(reverse=True)
            skill_mismatch = (
                (sum(worker_skills[0:required_units])
                 / required_units) - required_level
            ) if required_units > 0 else 0

            if skill_mismatch < 0:
                skill_balance += skill_mismatch ** 2
                number_with_negative_differences += 1

        if number_with_negative_differences > 0:
            return skill_balance / number_with_negative_differences
        else:
            return 0

    def compute_creativity_match(
            self,
            max_skill_level=MAX_SKILL_LEVEL,
            min_skill_level=MIN_SOFT_SKILL_LEVEL
    ):
        """Compute match between the creativity level of the team and
        the creativity level required by the project.

        Returns:
            float: creativity match value
        """

        self.creativity_level = 0.0
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
                self.creativity_level += (
                        sum([((p[1] - p[0]) / max_distance) ** 2
                             for p in pairs])
                        / len(pairs)
                )
                number_of_existing_skills += 1

        if number_of_existing_skills > 0:
            self.creativity_level /= number_of_existing_skills
        else:
            self.creativity_level = 0

        self.creativity_level = 1 + (
                self.creativity_level * max_distance
        )
        return (self.project.creativity - self.creativity_level) ** 2

    def skill_update(self, success, skill_update_func):
        """Update the member skills  on termination of project
        depending on project success status and project risk.

        Args:
            success: bool
                Was the project as success
            skill_update_func: function
                Returns skill modifier according to project risk
        """
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
        """Logs the outcome of the project in the member history and
        in the social network.

        Args:
            success: bool
                Was the project a success?
        """
        if success:
            self.lead.model.grid.add_team_edges(self)
        for member in self.members.values():
            member.history.record(success)

    def within_budget(self):
        """Tests if the team is within budget  for the project.

        Note:
            project.budget is None when budgetary constraint
            functionality is switched off at the simulation level.

        Returns:
            bool: True if budget constraint met
        """
        if self.project.budget is None:
            return True
        elif self.team_budget <= self.project.budget:
            return True
        else:
            return False

    def to_string(self):
        """Returns json formatted string for print or saving the
        details of this team."""
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
    """Interface class for organisation strategy that is used in team
    allocation.

    Note:
        Use of interface pattern is not very pythonic, it could
        be improved by using ABC pattern instead. One advantage
        would be that code shared across classes that implement
        the interface could be moved to the ABC and not need to
        be duplicated.
    """
    def invite_bids(self, project: Project) -> list:
        """Invite bids from workers to be considered for this
        project.
        """
        pass

    def select_team(self, project: Project,
                    bid_pool=None) -> Team:
        """Select team for this project from a bid pool of workers."""
        pass


class RandomStrategy(implements(OrganisationStrategyInterface)):
    """Random strategy for team allocation, implements interface.

    ...

    Attributes:
        model: model.SuperScriptModel
            Reference to main model, used to access list of
            agents (workers) via scheduler.
        min_team_size: int
            Minimum number of workers in team.
        max_team_size: int
            Maximum number of workers in team.
    """
    def __init__(self, model,
                 min_team_size=MIN_TEAM_SIZE,
                 max_team_size=MAX_TEAM_SIZE):
        self.model = model
        self.min_team_size = min_team_size
        self.max_team_size = max_team_size

    def invite_bids(self, project: Project) -> list:
        """Invite bids from workers.

        Calls worker.bid() with behaviour determined by the worker
        strategy that is in use.

        Args:
            project: project.Project

        Returns:
            bid_pool: list
                List of workers that are bidding for this project.
        """
        bid_pool = [
            worker for worker in self.model.schedule.agents
            if worker.bid(project)
        ]
        return bid_pool

    def select_team(self, project: Project,
                    bid_pool=None) -> Team:
        """Selects team at random from the supplied bid_pool.

        If the bid_pool is None, all agents in the simulation are
        available to select from.

        If the bid_pool is shorter than the chosen team size, an empty
        team is returned.

        Args:
            project: project.Project

            bid_pool: list (optional)
                Workers to choose from.

        Returns:
            organisation.Team: selected team
        """
        size = Random.randint(self.min_team_size,
                              self.max_team_size)
        bid_pool = (self.model.schedule.agents
                    if bid_pool is None else bid_pool)

        if size > len(bid_pool):
            workers = {}
            lead = None

        else:
            workers = {worker.worker_id: worker
                       for worker in
                       Random.choices(bid_pool, size)}
            lead = Random.choice(list(workers.values()))

        return Team(project, workers, lead)


class BasicStrategy(implements(OrganisationStrategyInterface)):
    """Basic strategy for team allocation, implements interface.

    Aims to improve on random team allocation, by selecting the
    most highly skilled workers available.

    Note:
        This strategy does respect budgetary constraints.

    ...

    Attributes:
      model: model.SuperScriptModel
          Reference to main model, used to access list of
          agents (workers) via scheduler.
      min_team_size: int
          Minimum number of workers in team.
      max_team_size: int
          Maximum number of workers in team.
    """

    def __init__(self, model,
                 min_team_size=MIN_TEAM_SIZE,
                 max_team_size=MAX_TEAM_SIZE):
        self.model = model
        self.min_team_size = min_team_size
        self.max_team_size = max_team_size

    def invite_bids(self, project: Project) -> list:
        """Invite bids from workers.

                Calls worker.bid() with behaviour determined by the worker
                strategy that is in use.

                Args:
                    project: project.Project

                Returns:
                    bid_pool: list
                        List of workers that are bidding for this project.
                """
        bid_pool = [
            worker for worker in self.model.schedule.agents
            if worker.bid(project)
        ]
        return bid_pool

    def select_team(self, project: Project,
                    bid_pool=None) -> Team:
        """Selects team consisting of the top N workers.

        If the bid_pool is None, all agents in the simulation are
        available to select from.

        If the bid_pool is shorter than the minimum team size, an
        empty team is returned.

        Args:
            project: project.Project

            bid_pool: list (optional)
                Workers to choose from.

        Returns:
            organisation.Team: selected team
        """
        bid_pool = (self.model.schedule.agents
                    if bid_pool is None else bid_pool)

        if len(bid_pool) < self.min_team_size:
            return Team(project, {}, None)
        else:
            return self.select_top_n(bid_pool, project)

    def rank_bids(self, bid_pool, project):
        """Ranks the workers in the bid_pool by total skill level
        summed across the skills that are required by this project.

        Args:
            bid_pool: list (optional)
                Workers to rank.
            project: project.Project
                Provides skill requirements to sum over.

        Returns:
            dict: workers ranked by total skill
        """
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
        """Selects a valid team (i.e. meet budget constraint) by
        working through the ranked bid_pool, sequentially adding
        workers until either:
            - the team exceeds the project budget
            - the maximum team size is reached
            - the end of the bid_pool is reached

        If the resulting team size is less than the minimum, an empty
        team is returned.

        Args:
            bid_pool: list (optional)
                Workers to choose from.

            project: project.Project

        Returns:
            organisation.Team: selected team

        """
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


class ParallelBasinhopping(implements(OrganisationStrategyInterface)):
    """Optimisation strategy for team allocation, implements interface.

    Tries to optimise the probability of project success using a
    optimiser supplied by the OptimiserFactory.

    Note:
        This is not currently very efficient when using a large number
        of cores (num_proc). 8 seems to work well, but 48 is too many!

    ...

    Attributes:
        model: model.SuperScriptModel
            Reference to main model, used to access list of
            agents (workers) via scheduler.
        optimiser_factory: optimisation.OptimiserFactory
            Supplies optimiser class.
        min_team_size: int
            Minimum team size.
        max_team_size: int
            Maximum team size.
        num_proc: int
            Number of processors (cores/threads) to use in parallel.
        niter: int
            Number of iterations to run optimiser for.
    """

    def __init__(self, model, optimiser_factory,
                 min_team_size=MIN_TEAM_SIZE,
                 max_team_size=MAX_TEAM_SIZE):

        self.model = model
        self.optimiser_factory = optimiser_factory
        self.min_team_size = min_team_size
        self.max_team_size = max_team_size
        self.num_proc = self.model.number_of_processors
        self.niter = self.model.number_of_basin_hops

    def invite_bids(self, project: Project) -> list:
        """Invite bids from workers for each possible start time offset
        for this project - creates a bid_pool for each offset.

        Calls worker.bid() with behaviour determined by the worker
        strategy that is in use.

        Args:
            project: project.Project

        Returns:
            bid_pool: dict
                Dictionary of bid_pools for each offset value.
        """
        base_start_time = project.start_time
        bid_pool = {}
        for offset in range(project.max_start_time_offset + 1):
            project.start_time = base_start_time + offset
            bid_pool[offset] = [
                worker for worker in self.model.schedule.agents
                if worker.bid(project)
            ]
        return bid_pool

    def select_team(self, project: Project,
                    bid_pool=None) -> Team:
        """Selects team with best probability of success, using the
        optimiser to find the best time at each start time offset,
        and then selecting the offset that produced the best team.

        Args:
            project: project.Project

            bid_pool: dict (optional)
                Workers to choose from at each offset value.

        Returns:
            organisation.Team: selected team
        """
        bid_pool = (
            {
             offset: self.model.schedule.agents
             for offset in range(project.max_start_time_offset + 1)
            }
            if bid_pool is None else bid_pool
        )

        base_start_time = project.start_time
        probabilities = []
        teams = []

        for offset in range(project.max_start_time_offset + 1):

            project.start_time = base_start_time + offset

            optimiser = self.optimiser_factory.get_optimiser(
                optimiser_name="Basinhopping",
                project=project,
                bid_pool=bid_pool[offset],
                model=self.model,
                niter=self.niter
            )

            runner = self.optimiser_factory.get_runner(
                runner_name="Parallel",
                optimiser=optimiser,
                num_proc=self.num_proc
            )
            team, probability = runner.run()
            teams.append(team)
            probabilities.append(probability)

        offset = argmax(probabilities)
        best_team = teams[argmax(probabilities)]

        project.start_time = base_start_time + offset
        project.progress = 0 - offset
        project.realised_offset = offset

        return best_team


class TeamAllocator:
    """Allocates team for project using one of the organisation
    strategies.

    ...

    Attributes:
      model: model.SuperScriptModel
          Reference to main model, used to access organisation strategy
          setting, to determine which strategy to instantiate.
      strategy: organisation.OrganisationStrategyInterface
          Strategy with invite_bids and select_team methods
    """

    def __init__(self, model, optimiser_factory):
        self.model = model

        if self.model.organisation_strategy == "Random":
            self.strategy = RandomStrategy(model)
        elif self.model.organisation_strategy == "Basic":
            self.strategy = BasicStrategy(model)
        elif self.model.organisation_strategy == "Basin":
            self.strategy = ParallelBasinhopping(model, optimiser_factory)

    def allocate_team(self, project: Project):
        """Allocates team to project.

        Note:
            If either the team or team.lead is None, then the team
            is invalid.

         Args:
             project: project.Project

         Returns:
             organisation.Team: allocated team
        """

        bid_pool = self.strategy.invite_bids(project)
        team = self.strategy.select_team(
            project, bid_pool=bid_pool
        )

        if team is not None and team.lead is not None:
            if team.within_budget():
                team.assign_contributions_to_members()
                team.assign_lead()
            else:
                team = None

        project.team = team


class Trainer:
    """Single instance of trainer is responsible for all training of
    low skilled workers.

    Currently the two available training methods are 'all' and 'slots'.

    Note:
        Training commences after a pre-defined number of timesteps.
        This is to avoid half the workforce immediately being absorbed
        into training when using 'all' mode.

    TODO:
        Refactor this to use the strategy pattern for different modes
        of training, rather than the 'if' clause currently used in
        the train() method.
    ...

    Attributes:
        model: model.SuperScriptModel
            Reference to main model to access training mode setting.
        training_length: int
            Number of steps that a training course lasts.
        max_skill_level: int
            Maximum possible skill level.
        hard_skills: list
            Identifies which skills are 'hard skills'.
            By default it is ['A', B', 'C', 'D', 'E'].
        skill_quartiles: dict
            Quartiles for each hard skill. Used to determine who needs
            training. Updated on each timestep.
        trainees: dict
            Record of the workers currently in training.

    """

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
        """Returns top two in demand skills at current time."""
        return self.model.inventory.top_two_skills

    def update_skill_quartiles(self):
        """Re-calculates the quartile values for the hard skills."""
        for skill in self.hard_skills:
            self.skill_quartiles[skill] = percentile(
                [worker.get_skill(skill)
                 for worker in self.model.schedule.agents],
                [25, 50, 75]
            )

    def train(self):
        """Advances training and then adds more workers to training
        according to which training mode is in use."""
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
        """Advances the training_remaining counter for each worker who
        is currently in training. When the counter reaches zero, the
        worker is removed from training.

        Note:
            The list of worker_ids is set before the loop, because the
            dictionary keys may change during the loop as workers
            reach the end of their training.
        """
        worker_ids = list(self.trainees.keys())
        for worker_id in worker_ids:
            if worker_id in self.trainees.keys():
                self.trainees[worker_id].training_remaining -= 1

                if self.trainees[worker_id].training_remaining == 0:
                    del self.trainees[worker_id]

    @staticmethod
    def worker_free_to_train(worker):
        """Checks that the worker is free during the proposed training:
         - not assigned to any project work
         - departmental workload is met.

         Note:
             'slack' is set to the equivalent of full time, because
              the departmental workload must be met + there must be
              enough spare capacity for the worker to allocate all of
              their time to training.

        Args:
            worker: worker.Worker
                Worker to check if free.
         """
        return worker.is_free(
            worker.now, worker.training_horizon,
            slack=worker.contributions.units_per_full_time
        )

    def add_slots_for_training(self):
        """This method creates a number of training slots on each
        timestep to try and meet the model.target_training_load.

        For each worker, their lowest skill is selected out of the
        current top two in demand skills. If their skill level for
        this skill is below the median of the workforce, they are
        add to the list of candidates for training. This list is
        sorted in ascending skill order, and candidates are added
        for training until all the slots are filled.

        Only the worst skill is trained for each worker.
        """

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
                / self.training_length
        )
        for worker_id in sorted_workers.keys():
            worker = self.model.schedule._agents[worker_id]

            if self.worker_free_to_train(worker):
                self.add_worker(worker, sorted_workers[worker_id][0])
                slots += 1
                if slots >= target_slots:
                    break

    def add_all_for_training(self):
        """All workers whose skill level for one of the op two in
        demand is below median are added for training.

        Only one of their skills can be trained, with a preference
        for the most in-demand skill.
        """
        for worker in self.model.schedule.agents:
            requires_training = False

            if self.worker_free_to_train(worker):

                for skill in self.top_two_demanded_skills():
                    if worker.get_skill(skill) < self.skill_quartiles[skill][1]:
                        requires_training = True
                        skill_to_train = skill
                        break

                if requires_training:
                    self.add_worker(worker, skill_to_train)

    def add_worker(self, worker, skill_to_train):
        """Updates worker skill and adds worker for training by:
        - adding self.trainees
        - registering with the department that worker is on training
        - setting the workers training_remaining counter
        - blocking out the workers contributions for the duration of
        training so that they are not available for project work

        Note:
            Skill update happens immediately when worker enters
            training, which is not realistic.

        Args:
            worker: worker.Worker
                Worker to train.
            skill_to_train: str
                Which hard skill to train.
        """
        self.update_worker_skill(worker, skill_to_train)

        self.trainees[worker.worker_id] = worker
        worker.department.add_training(worker, self.training_length)
        worker.training_remaining = self.training_length
        for t in range(self.training_length):
            worker.contributions.total_contribution[worker.now + t] = (
                worker.contributions.units_per_full_time
            )

    def update_worker_skill(self, worker, skill_to_train):
        """Skill update mechanism.

        Skill level is boosted to the third quartile across
        the work force.

        The worker's training_tracker is also update, to log how much
        each skill changes due to training over the course of the
        simulation.

        Args:
            worker: worker.Worker
                Worker to train.
            skill_to_train: str
                Which hard skill to train.
        """
        old_skill = worker.skills.hard_skills[skill_to_train]
        new_skill = min(
            self.skill_quartiles[skill_to_train][2],
            self.max_skill_level
        )
        worker.skills.hard_skills[skill_to_train] = new_skill
        worker.skills.training_tracker[skill_to_train] = (
            worker.skills.hard_skills[skill_to_train] - old_skill
        )

    def training_boost(self):
        """Instantly boost the skill levels of all workers whose skill
        is below median, making them equal to the thrid quartile value.

        Note:
            This method can be called externally as an intervention
            during simulation to test the effects of training.
        """
        for worker in self.model.schedule.agents:
            for skill in self.hard_skills:
                if (
                        worker.get_skill(skill)
                        < self.skill_quartiles[skill][1]
                ):
                    worker.skills.hard_skills[skill] = (
                        self.skill_quartiles[skill][2]
                    )


class Department:
    """Department class.

    Keeps track of how many units its workers are contributing to
    projects (and training) at each timestep, and ensures that
    the departmental workload is always met.

    ...

    Attributes:
        dept_id: int
            Unique integer ID for this department.
        number_of_workers: int
            Number of worker in this department.
        workload: float
            Fraction of the total capacity that must be kept free from
            project work or training to ensure that departmental
            workload is met.
        units_per_full_time: int
            Number of units of work that are equivalent to full time
            for an individual.
        slack: int
            Number of units of spare capacity required when checking
            if departmental workload is met. Default value set in
            config.
        units_supplied_to_projects: dict
            Logs total number of units supplied by department workers
            to project (and training) at each timestep.
        maximum_project_units: int
            The total number of units that the department can supply to
            project at any time (if none of its workers are on
            training).
    """

    def __init__(self, dept_id,
                 workload=DEPARTMENTAL_WORKLOAD,
                 units_per_full_time=UNITS_PER_FTE,
                 tolerance=WORKLOAD_SATISFIED_TOLERANCE):

        self.dept_id = dept_id
        self.number_of_workers = 0
        self.workload = workload
        self.units_per_full_time = units_per_full_time
        self.slack = tolerance
        self.units_supplied_to_projects = dict()
        self.maximum_project_units = 0

    def update_supplied_units(self, units_contributed, project):
        """Registers that a specified number of units will be
        contributed for the duration of this project.

        Args:
            units_contributed: int
                Number of units to register
            project: project.Project
                Project to which these units are being contributed
                (defines the duration of the contribution).
        """
        for time_offset in range(project.length):
            time = project.start_time + time_offset

            if time not in self.units_supplied_to_projects.keys():
                self.units_supplied_to_projects[time] = units_contributed
            else:
                self.units_supplied_to_projects[time] += units_contributed

    def add_worker(self):
        """Adds a worker to the department. Updates the maximum
        capacity of the department accordingly."""
        self.number_of_workers += 1
        self.evaluate_maximum_project_units()

    def add_training(self, worker, length):
        """Registers that a worker is now entering training.

        Their full time equivalent number of units is added to the
        total being supplied by the department, over the duration
        of the training.

        Args:
            worker: worker.Worker
                Worker that is being trained.
            length: int
                Number of timesteps they will be trained for (starting
                on current timestep).
        """

        start = worker.now
        units = self.units_per_full_time
        for time_offset in range(length):
            time = start + time_offset

            if time not in self.units_supplied_to_projects.keys():
                self.units_supplied_to_projects[time] = units
            else:
                self.units_supplied_to_projects[time] += units

    def evaluate_maximum_project_units(self):
        """Determines maximums theoretical capacity while still meeting
        departmental workload.
        """

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
        """Get total units supplied by departmental workers at this
        time.

        Args:
            time: int
                Timestep to check.

        Returns:
            int: total units supplied at time.
        """

        return self.units_supplied_to_projects.get(time, 0)

    def is_workload_satisfied(self, start, length, slack=None):
        """Check if departmental workload is satisfied over specified
         time range.

        Note:
            Different amounts of 'slack' are required in different
            situations. If slack is zero, then departmental workload
            might be satisfied, but there is not spare capacity in the
            department for workers to enter training or join projects.

            To enter training there must be at least 10 units of slack
            (equivalent to full time) in the department.

        Args:
            start: int
                Timestep that range starts at.
            length: int
                Length of range in timesteps.
            slack: int
                Number of additional units above baseline workload
                to keep free.

        Returns: bool
            True if workload is satisfied (with requested slack).
        """
        slack = self.slack if slack is None else slack
        for t in range(length):

            time = start + t
            if (
                    self.units_supplied_to_projects_at_time(time)
                    >= (self.maximum_project_units - slack)
            ):
                return False

        return True

    def get_remaining_unit_budget(self, start, length):
        """Determine how many units are available across the department
        over the specified time range.

        Note:
            This uses the default 'slack' value as set in the config
            file.

        Args:
            start: int
                Timestep that range starts at.
            length: int
                Length of range in timesteps.

        Returns:
            int: number of units available.
        """
        budget_over_time = []
        for t in range(length):
            budget_over_time.append(
                self.maximum_project_units
                - self.units_supplied_to_projects_at_time(start + t)
                - self.slack
            )
        return min(budget_over_time)

    def to_string(self):
        """Returns json formatted string for printing or saving
        the details of the department.
        """

        output = {
            'dept_id': self.dept_id,
            'number_of_workers': self.number_of_workers,
            'workload': self.workload,
            'units_per_full_time': self.units_per_full_time,
            'tolerance': self.slack
        }
        return json.dumps(output, indent=4)
