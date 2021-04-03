"""
SuperScript project module
===========

Classes:
    ProjectInventory
        Inventory of all projects in the simulation. Creates, tracks
        and interacts with projects.
    Project
        Each project created is an instance of this class
    Project requirements
        Handles projects requirements (skill, risk etc)
    SuccessCalculator
        Inventory has a single instance of SuccessCalculator, for
        computing project success probabilities
"""

from copy import deepcopy
import numpy as np
import json
import pickle

from .function import FunctionFactory
from .utilities import Random
from .config import (MAXIMUM_TIMELINE_FLEXIBILITY,
                     PROJECT_LENGTH,
                     DEFAULT_START_OFFSET,
                     DEFAULT_START_TIME,
                     P_HARD_SKILL_PROJECT,
                     PER_SKILL_MAX_UNITS,
                     PER_SKILL_MIN_UNITS,
                     MIN_REQUIRED_UNITS,
                     MIN_SKILL_LEVEL,
                     MAX_SKILL_LEVEL,
                     MIN_PROJECT_CREATIVITY,
                     MAX_PROJECT_CREATIVITY,
                     RISK_LEVELS,
                     P_BUDGET_FLEXIBILITY,
                     MAX_BUDGET_INCREASE,
                     HARD_SKILLS)


class ProjectInventory:
    """Inventory of all projects in the simulation

    This class is responsible for creating projects, and keeping track
    of which projects are active, handling null_projects (those without
    a valid team allocated), and logging project/team data using the
    model DataCollector.

    During project creation, the team allocation also takes place and
    then the project success probability is determined.

    This class also has the capability to either save all the projects
    to disk (for re-use in later simulations), or to load pre-defined
    projects from disk.
    ...

    Attributes:
        projects: dict
            Stores project instances (key is project_id)
        null_projects: dict
            Stores projects which are 'null' (either team or lead is
            None due to failure to allocate a viable team). The null
            projects are removed on each timestep
        null_count: int
            Counts number of null projects
        index_total: int
            Count total number of projects added to the simulation, to
            ensure that new projects have a unique project_id
        team_allocator: organisation.TeamAllocator
            Class that uses predefined strategies to invite bids and
            select team for a project
        timeline_flexibility_func: function
            Function that produces the timeline flexibility value for a
            project. The actual start time offset is then chosen based
            on this. (How it is chosen depends on the organisation
            strategy.)
        max_timeline_flex: int
            Parameter for timeline_flexibility_func. Defines upper
            limit on timeline flexibility.
        success_calculator: SuccessCalculator
            Calculates project success probabilities (and determines
            is they succeed or fail).
        success_history: dict
            Records timeseries of project successes.
        fail_history: dict
            Records timeseries of project failures.
        total_skill_requirement: dict
            Stores current total requirements for each hard skill
            across all newly created projects (updated each timestep).
        social_network: network.SocialNetwork
            This is used in the computation to determine the team
            chemistry based on their historical collaborations, which
            are stored in the network.
        model: model.SuperScriptModel
            Reference to the main model, used for various purposes
            including accessing model level config variables (which
            need to be defined at the model level in order to work
            with the Mesa GUI controls), and for doing data collection
            using the DataCollector.
        skill_update_func: function
            Passed to team.skill_update method to update member skills
            on project termination, depending on project success.
            The function is obtained from function.FunctionFactory
        io_dir: string
            Directory for reading/writing projects (if save_flag or
            load_flag is True)
        save_flag: bool
            If True all projects will be saved for later use.
        load_flag: bool
            If True the inventory will try to load predefined projects
            from project_file.pickle in the specified io_dir, instead
            of creating new projects at random.
    """

    def __init__(self,
                 team_allocator,
                 timeline_flexibility='NoFlexibility',
                 max_timeline_flex=MAXIMUM_TIMELINE_FLEXIBILITY,
                 hard_skills=HARD_SKILLS,
                 social_network=None,
                 model=None,
                 save_flag=False,
                 load_flag=False,
                 io_dir='./'):
        """Create and configure inventory - called once in model.py on
        simulation setup.
        """
        self.projects = dict()
        self.null_projects = dict()
        self.null_count = 0
        self.index_total = 0
        self.team_allocator = team_allocator
        self.timeline_flexibility_func = (
            FunctionFactory.get(timeline_flexibility)
        )
        self.max_timeline_flex = max_timeline_flex

        self.success_calculator = SuccessCalculator()
        self.success_history = dict()
        self.fail_history = dict()

        self.total_skill_requirement = dict(zip(
            hard_skills, [0 for s in hard_skills]
        ))
        self.social_network = social_network
        self.model = model
        self.skill_update_func = (FunctionFactory.get(
            'SkillUpdateByRisk'
        ) if self.model.update_skill_by_risk_flag
          else FunctionFactory.get('IdentityFunction'))

        self.io_dir = io_dir
        self.save_flag = save_flag
        if self.save_flag:
            self.all_projects = {}

        self.load_flag = load_flag
        assert not self.save_flag & self.load_flag

        if self.load_flag:
            try:
                with open(io_dir + '/project_file.pickle', 'rb') as ifile:
                    self.all_projects = pickle.load(ifile)
            except FileNotFoundError:
                print(
                    'Cannot load predefined projects: '
                    'project_file.pickle not found.'
                )
                self.load_flag = False

    @property
    def active_count(self):
        """Count number of projects that are currently active (have
        started). Used by SSDataCollector.
        """
        return sum([1 for p
                    in self.projects.values()
                    if p.progress >= 0])

    @property
    def top_two_skills(self):
        """Returns top two in-demand skills at current time.
        """
        return list(self.total_skill_requirement.keys())[:2]

    def remove_null_projects(self):
        """Removes null projects (which have failed to have a viable
        team assigned to them. If this is not called the null projects
        will hang because there is no team lead to advance them.
        """
        nulls = list(self.null_projects.keys())
        self.null_count = len(nulls)

        for project_id in nulls:
            self.log_project_data_collector(
                self.null_projects[project_id], null=True, success=False
            )
            self.delete_project(project_id)
            del self.null_projects[project_id]

    def get_start_time_offset(self):
        """Chooses a start time offset uniformly at random from the
        probabilities produce by timeline_flexibility_func.get()

        Note:
            For 'Random' and 'Basic' team allocation strategies, this
            offset is used as the actual offset. For 'Basin' (and any
            future optimisation methods), the actual offset is chosen
            as the one which produces the highest probability of
            project success, with this offset as the upper limit.
        """
        p_vector = (
            self.timeline_flexibility_func
            .get_values(np.arange(self.max_timeline_flex + 1))
        )

        r = Random.uniform()
        lb = 0
        for i in range(self.max_timeline_flex + 1):
            if r < p_vector[i] + lb:
                return i
            lb += p_vector[i]

    def determine_total_skill_requirements(self, projects):
        """Sums all skill requirements across a list of projects
        and then ranks them (so that the top two can be selected).

        Args:
            projects: list
                List of projects over which to sum skill requirements
        """
        for skill in self.total_skill_requirement.keys():
            self.total_skill_requirement[skill] = sum([
                project.requirements.hard_skills[skill]['units']
                for project in projects
            ])
        self.rank_total_requirements()

    def rank_total_requirements(self):
        """Ranks the hard skills by their total required units in
        descending order.
        """
        self.total_skill_requirement = {
            k: v for k, v in sorted(
                self.total_skill_requirement.items(),
                reverse=True,
                key=lambda item: item[1]
            )
        }

    def get_loaded_projects_for_timestep(self, time,
                                         auto_offset=False):
        """Returns predefined projects for this timestep, from the set
         of projects that were loaded on construction of the inventory.

         Args:
             time: int
                Timestep to get projects for.
            auto_offset:
                If True, the realised_offset is equal to the
                max_start_time_offset, and is automatically applied to
                the start_time of the project.

        Note:
            The auto_offset feature is cumbersome. But it allows a
            different team allocation strategy (e.g. 'Basin') to be
            used on the loaded projects than the strategy that was
            originally used (e.g. 'Random') in the simulation that
            created them. This is important functionality for comparing
            different team allocation strategies on the same set of
            projects.
        """
        predefined_projects = self.all_projects.get(time, [])
        new_projects = [deepcopy(p) for p in predefined_projects]
        for project in new_projects:
            project.inventory = self
            project.team = None

            start_time_offset = self.get_start_time_offset()
            project.max_start_time_offset = start_time_offset
            project.realised_offset = (
                start_time_offset if auto_offset
                else None
            )
            project.progress = 0 - project.max_start_time_offset
            project.start_time = (
                time + project.max_start_time_offset
                if auto_offset else time
            )
            project.requirements.budget_functionality_flag = (
                self.model.budget_functionality_flag
            )
            project.requirements.budget = (
                project.requirements.calculate_budget()
            )

        return new_projects

    def create_projects(self, new_projects_count,
                        time, length):
        """Creates a batch of new projects. Called once per timestep.

        First the projects are created, then the total_skill_requirements
        are updated. Then a team is allocated for each project, the
        success probability is calculated and the project is added to
        the inventory.

        If project save to disk is enabled, the new projects are logged
        the self.all_projects to be saved at the end of the simulation.

        Args:
            new_projects_count: int
                Number of new projects to create (defined in
                config.py).
            time: int
                Start time for the projects
            length: int
                Length of each project

        Note:
            If 'auto_offset' is True, then the realised_offset is equal
            to the max_start_time_offset. (See docstring for
            get_loaded_projects_for_timestep).
        """
        auto_offset = (
            False if self.model.organisation_strategy == 'Basin'
            else True
        )

        if self.load_flag:
            new_projects = self.get_loaded_projects_for_timestep(
                time, auto_offset
            )
        else:
            new_projects = []
            for i in range(new_projects_count):
                p = Project(
                    self, self.index_total + i,
                    project_length=length,
                    start_time_offset=self.get_start_time_offset(),
                    start_time=time,
                    auto_offset=auto_offset
                )
                new_projects.append(p)

            self.determine_total_skill_requirements(new_projects)
            new_projects = self.rank_projects(new_projects)

        if self.save_flag:
            self.all_projects[time] = new_projects

        for p in new_projects:
            self.team_allocator.allocate_team(p)
            self.success_calculator.calculate_success_probability(p)
            self.add_project(p)

    def save_projects(self):
        """Saves self.all_projects to disk for re-use in later
        simulations.
        """
        if self.save_flag:
            with open(
                    self.io_dir + '/project_file.pickle',
                    'wb') as ofile:

                pickle.dump(self.all_projects, ofile)

    @staticmethod
    def rank_projects(project_list):
        """Ranks a list of projects by risk and then creativity in
        descending order, so that high-risk, high-creativity projects
        will be allocated a team first.

        Args:
            project_list: list
                List of projects to rank
        """
        project_list.sort(
            reverse=True, key=lambda x: (x.risk, x.creativity)
        )
        return project_list

    def add_project(self, project):
        """Adds project to the inventory, providing it has a valid team
        allocated. Otherwise, the project is added to the dictionary
        of null_projects which are removed from the simulation each
        timestep.

        Args:
            project: project.Project
        """
        if project.team is None or project.team.lead is None:
            self.null_projects[project.project_id] = project

        if project.project_id not in self.projects.keys():
            self.projects[project.project_id] = project
            self.index_total += 1
        else:
            raise KeyError('Project ID %d already exists in inventory.'
                           % project.project_id)

    def delete_project(self, project_id):
        """Safely removes project from inventory.

        Args:
            project: project.Project
        """
        try:
            del self.projects[project_id]
        except KeyError:
            print('Project ID %d not in inventory.' % project_id)
            raise

    def advance_projects(self):
        """Advances all projects in inventory.

        Deprecated because project are advanced individually by the
        project lead in the Worker.step() method.

        Note:
            Safe looping over dictionary which can change during the
            loop if projects terminate when advanced.
        """
        project_ids = list(self.projects.keys())

        for pid in project_ids:
            if pid in self.projects.keys():
                self.projects[pid].advance()

    def log_project_data_collector(self, project, null, success):
        """Method that creates and logs row of data relating to the
        project and its assigned team. This is called either when
        the project terminates, or if when it is removed from the
        simulation if it is a null_project (i.e. no team allocated
        or invalid team).

        Uses the model.datacollector instance of SSDataCollector
        which uses a Mesa table to collect this data.

        Args:
            project: project.Project
                The project to log
            null: bool
                Was the project a null_project? If True the team data
                elements are set to None
            success: bool
                Was the project successful?
        """
        success_calculator = project.inventory.success_calculator
        success_calculator.get_component_values(project)

        next_row = {
            "project_id": project.project_id,
            "prob": project.success_probability,
            "risk": project.risk,
            "budget": project.budget,
            "null": null,
            "success": success,
            "maximum_offset": project.max_start_time_offset,
            "realised_offset": project.realised_offset,
            "start_time": project.start_time,
            "ovr_prob_cpt": (
                success_calculator
                .probability_ovr
                .get_values(success_calculator.ovr)
            ),
            "skill_balance_prob_cpt": (
                success_calculator
                .probability_skill_balance
                .get_values(success_calculator.skill_balance)
            ),
            "creativity_match_prob_cpt": (
                success_calculator
                .probability_creativity_match
                .get_values(success_calculator.creativity_match)
            ),
            "risk_prob_cpt": (
                success_calculator
                .probability_risk
                .get_values(success_calculator.risk)
            ),
            "chemistry_prob_cpt": (
                success_calculator
                .probability_chemistry
                .get_values(success_calculator.chemistry)
            )
        }

        if not null and project.team is not None:
            next_row["team_budget"] = project.team.team_budget
            next_row["team_ovr"] = project.team.team_ovr
            next_row["team_creativity_level"] = (
                project.team.creativity_level
            )
            next_row["team_creativity_match"] = (
                project.team.creativity_match
            )
            next_row["team_size"] = len(project.team.members)
        else:
            next_row["team_budget"] = None
            next_row["team_ovr"] = None
            next_row["team_creativity_level"] = None
            next_row["team_creativity_match"] = None
            next_row["team_size"] = None

        self.model.datacollector.add_table_row(
            "Projects", next_row, ignore_missing=False
        )


class Project:
    """Project class.

    This class directly handles advancing and termination of project,
    and tracks project progress. Also instantiates ProjectRequirements
    class which handles all skill and budget requirements.

    Properties of this class are provided fo easy access to features of
    its requirements that are required to calculate probabilities of
    success (e.g. risk, creativity, etc).
    ...

    Attributes:
        inventory: inventory
            Reference to parent inventory
        project_id: int
            Unique integer identifier
        length: int
            Length of project in timesteps
        progress: int
            Tracks project progress.
            <0 mean project has not started (inactive)
        max_start_time_offset: int
            Maximum allowed start time offset for this project
            i.e. largest number of timesteps in future when this
            project could start
        realised_offset: int
            Actual start time offset, either set to equal
            max_start_time_offset or chosen by optimiser, depending on
            which team allocation strategy is in use
        start_time: int
            Timestep on which this project starts
        team: organisation.Team
            The team that is allocated to this project. If team is None
            or team.lead is None, project is null
        requirements: project.ProjectRequirements
            Class that handles all requirement for this project
        success_probability: float
            Project probability of success ( must be >=0.0)
        """
    def __init__(self,
                 inventory: ProjectInventory,
                 project_id=42,
                 project_length=PROJECT_LENGTH,
                 start_time_offset=DEFAULT_START_OFFSET,
                 start_time=DEFAULT_START_TIME,
                 auto_offset=True):

        self.inventory = inventory
        self.project_id = project_id
        self.length = project_length
        self.progress = 0 - start_time_offset
        self.max_start_time_offset = start_time_offset
        self.realised_offset = (start_time_offset
                                if auto_offset else None)
        self.start_time = (start_time + start_time_offset
                           if auto_offset else start_time)
        self.team = None
        self.requirements = ProjectRequirements(
            budget_functionality_flag
            =self.inventory.model.budget_functionality_flag,
            p_budget_flexibility
            =self.inventory.model.p_budget_flexibility,
            max_budget_increase
            =self.inventory.model.max_budget_increase
        )
        self.success_probability = 0.0

    def advance(self):
        """Advance project by one timestep, terminate if complete."""
        self.progress += 1
        if self.progress >= self.length:
            self.terminate()

    def terminate(self):
        """Terminate project.

        Determine if project was a success, update team member skills,
        log outcome. Clear the Team and then delete project from
        inventory.
        """
        success = (
            self.inventory.success_calculator.determine_success(self)
        )
        if self.team is not None:
            self.team.skill_update(
                success, self.inventory.skill_update_func
            )
            self.team.log_project_outcome(success)
            self.team.remove_lead()
            self.team = None

        self.inventory.delete_project(self.project_id)

    @property
    def required_skills(self):
        """Returns dictionary of required units and levels by hard
        skill."""
        return self.requirements.get_required_skills()

    @property
    def risk(self):
        """Returns project risk (int)."""
        return self.requirements.risk

    @property
    def creativity(self):
        """Returns creativity requirement (float)."""
        return self.requirements.creativity

    @property
    def chemistry(self):
        """Returns Team chemistry (float)."""
        chemistry = np.mean(
            [member.individual_chemistry(self)
             for member in self.team.members.values()]
        )
        if self.inventory.social_network is not None:
            chemistry += (
                self.inventory.social_network
                    .get_team_historical_success_flag(self.team)
            )
        return chemistry

    @property
    def budget(self):
        """Returns project budget (float)."""
        return self.requirements.budget

    def get_skill_requirement(self, skill):
        """Get unit and level requirements for a specific hard skill.

        Args:
            skill: string
                Takes value from ['A', 'B, ..., 'E']
        Returns:
            dict: requirements for skill
        """
        return self.requirements.hard_skills[skill]


class ProjectRequirements:
    """Class that handles skill and budget requirements for a specific
    project instance.

    Attributes:
        risk: int
            Choice uniformly at random from predefined list
        creativity: int
            Chosen uniformly at random between min and max
        flexible_budget: bool
            Does this project have a flexible budget?
            Determined with specified probability.
        p_hard_skill_required: float
            Probability that each had skill will be required.
        min_skill_required: int
            At least one skill must have at least this many required
            units.
        per_skill_max: int
            Maximum number of units that can be required for single
            skill.
        per_skill_min: int
            Minimum number of units that can be required (if skill is
            selected with probability p_hard_skill_required).
        min_skill_level: int
            Minimum skill level that can be required.
        max_skill_level = max_skill_level
            Maximum skill level that can be required.
        hard_skills: dict
            Dictionary that stores requirements for each hard skill.
            Element for each skill has an entry for 'level' and 'units'
        total_skill_units: int
            Total number of skill units required by this project
        budget_functionality_flag: bool
            Flag that allows budget functionality to be switched on/off
            If False, any team is valid for this project.
            If True, team must be within buget to be valid.
        max_budget_increase: float
            Multiplier giving maximum allowed increase in the budget,
            if this project has a flexible budget.
        budget: float
            Actual budget for this project.
    """
    def __init__(self,
                 p_hard_skill_required=P_HARD_SKILL_PROJECT,
                 per_skill_max=PER_SKILL_MAX_UNITS,
                 per_skill_min=PER_SKILL_MIN_UNITS,
                 min_skill_required=MIN_REQUIRED_UNITS,
                 hard_skills=HARD_SKILLS,
                 min_skill_level = MIN_SKILL_LEVEL,
                 max_skill_level = MAX_SKILL_LEVEL,
                 min_project_creativity=MIN_PROJECT_CREATIVITY,
                 max_project_creativity=MAX_PROJECT_CREATIVITY,
                 risk_levels=RISK_LEVELS,
                 p_budget_flexibility=P_BUDGET_FLEXIBILITY,
                 max_budget_increase=MAX_BUDGET_INCREASE,
                 budget_functionality_flag=True):

        self.risk = Random.choice(risk_levels)
        self.creativity = Random.randint(min_project_creativity,
                                         max_project_creativity)
        self.flexible_budget = (
            True if Random.uniform() <= p_budget_flexibility else False
        )

        self.p_hard_skill_required = p_hard_skill_required
        self.min_skill_required = min_skill_required
        self.per_skill_max = per_skill_max
        self.per_skill_min = per_skill_min
        self.min_skill_level = min_skill_level
        self.max_skill_level = max_skill_level

        self.hard_skills = dict(zip(hard_skills,
                                    [{
                                        'level': None,
                                        'units': 0}
                                        for s in hard_skills])
                                )
        self.total_skill_units = None

        max_assigned_units = 0
        while max_assigned_units < self.min_skill_required:

            self.assign_skill_requirements()
            max_assigned_units = max(
                [s['units'] for s in self.hard_skills.values()
                 if s['level'] is not None]
            )
        self.budget_functionality_flag = budget_functionality_flag
        self.max_budget_increase = max_budget_increase
        self.budget = self.calculate_budget()

    def select_non_zero_skills(self):
        """Selects which (hard) skills will be included in the
        requirements, each with probability self.p_hard_skill_required

        Also sets self.total_skill_units required by the project, based
        on how many hard skills are required.

        Note:
            At least one skill must be required.

        Returns:
            n_skills: int
                The number of non-zero (i.e. required) hard skills
            non_zero_skills: list
                Randomly ordered list of the required skills.
                This is then used to assign the required units and
                levels of each skill.
        """
        n_skills = 0
        while n_skills == 0:

            non_zero_skills = [s for s in self.hard_skills.keys()
                               if Random.uniform() <= self.p_hard_skill_required]
            Random.shuffle([non_zero_skills])
            n_skills = len(non_zero_skills)

        self.total_skill_units = Random.randint(n_skills * self.per_skill_min + 1,
                                                n_skills * self.per_skill_max)
        return n_skills, non_zero_skills

    def assign_skill_requirements(self):
        """Selects which skills are required by this project, then
        allocates the self.total_skill_units across these skills
        at random and assigns a randomly select required 'level'
        for each skill.
        """
        n_skills, non_zero_skills = self.select_non_zero_skills()
        remaining_skill_units = self.total_skill_units
        for i, skill in enumerate(non_zero_skills):

            a = (remaining_skill_units
                 - (n_skills - (i + 1)) * self.per_skill_max)
            a = max(a, self.per_skill_min)

            b = (remaining_skill_units
                 - (n_skills - (i + 1)) * self.per_skill_min)
            b = min(b, self.per_skill_max)

            units = Random.randint(a, b)
            self.hard_skills[skill]['level'] = Random.randint(
                self.min_skill_level, self.max_skill_level
            )
            self.hard_skills[skill]['units'] = units
            remaining_skill_units -= units

    def get_required_skills(self):
        """Returns a list of which skills are required by this project.
        """
        return [skill for skill
                in self.hard_skills.keys()
                if self.hard_skills[skill]['level'] is not None]

    def calculate_budget(self,
                         flexible_budget_flag=None,
                         max_budget_increase=None):
        """Determine the budget for this project, taking into account
        budget flexibility (if enabled and if this project has a
        flexible budget).

        Args:
             flexible_budget_flag: bool
                Allows user to override the projects existing flag.
                If None, then self.flexible_budget is used.
             max_budget_increase: bool
                Allows user to override the default
                If None, then self.max_budget_increase is used.

        Returns:
            float: budget value
        """
        if flexible_budget_flag is None:
            flexible_budget_flag = self.flexible_budget
        if max_budget_increase is None:
            max_budget_increase = self.max_budget_increase

        if not self.budget_functionality_flag:
            return None

        budget = 0
        for skill in self.get_required_skills():
            budget += (self.hard_skills[skill]['units']
                       * self.hard_skills[skill]['level'])

        if flexible_budget_flag:
            budget *= max_budget_increase

        return budget

    def to_string(self):
        """Returns json formatted string for printing or saving project
        requirements.
        """
        output = {
            'risk': self.risk,
            'creativity': self.creativity,
            'flexible_budget': self.flexible_budget,
            'budget': self.budget,
            'p_hard_skill_required': self.p_hard_skill_required,
            'min_skill_required': self.min_skill_required,
            'per_skill_cap': self.per_skill_max,
            'total_skill_units': self.total_skill_units,
            'hard_skills': self.hard_skills
        }
        return json.dumps(output, indent=4)


class SuccessCalculator:
    """Class for calculating project probability of success, and also
    for determining project success/fail on termination.

    Attributes:
        probability_ovr: function
            Function for calculating OVR contribution to probability
        probability_skill_balance: function
            Function for calculating skill balance contribution to
            probability
        probability_creativity_match: function
            Function for calculating creativity match contribution to
            probability
        probability_risk: function
            Function for calculating risk contribution to probability
        probability_chemistry: function
            Function for calculating chemistry contribution to
            probability
        ovr: float
            OVR value, argument for above function
        skill_balance: float
            skill_balance value, argument for above function
        creativity_match: float
            creativity_match value, argument for above function
        risk: float
            risk value, argument for above function
        chemistry: float
            chemistry value, argument for above function
    """
    def __init__(self):
        self.probability_ovr = (
            FunctionFactory.get('SuccessProbabilityOVR')
        )
        self.probability_skill_balance = (
            FunctionFactory.get('SuccessProbabilitySkillBalance')
        )
        self.probability_creativity_match = (
            FunctionFactory.get('SuccessProbabilityCreativityMatch')
        )
        self.probability_risk = (
            FunctionFactory.get('SuccessProbabilityRisk')
        )
        self.probability_chemistry = (
            FunctionFactory.get('SuccessProbabilityChemistry')
        )
        self.ovr = 0.0
        self.skill_balance = 0.0
        self.creativity_match = 0.0
        self.risk = 0.0
        self.chemistry = 0.0

    def get_component_values(self, project):
        """Set the component values from this project and its team.

        Args:
            project: project.Project
        """
        if project.team is not None:
            self.ovr = project.team.team_ovr
            self.skill_balance = project.team.skill_balance
            self.creativity_match = project.team.creativity_match
            self.risk = project.risk
            self.chemistry = project.chemistry
        else:
            self.ovr = 0.0
            self.skill_balance = 0.0
            self.creativity_match = 0.0
            self.risk = 0.0
            self.chemistry = 0.0

    def calculate_success_probability(self, project):
        """Calculate the success probability for this project,
        and ensure that it is non-negative.

        Args:
            project: project.Project
        """
        if project.team is None or project.team.lead is None:
            project.success_probability = 0
        else:
            self.get_component_values(project)
            probability = (
                self.probability_ovr.get_values(self.ovr)
                + self.probability_skill_balance.get_values(
                              self.skill_balance)
                + self.probability_creativity_match.get_values(
                              self.creativity_match)
                + self.probability_risk.get_values(self.risk)
                + self.probability_chemistry.get_values(self.chemistry)
            ) / 100
            project.success_probability = max(0, probability)

    def determine_success(self, project):
        """Determine if the project was successful.

        Args:
            project: project.Project
        """
        success = Random.uniform() <= project.success_probability
        time = project.inventory.model.schedule.steps

        if success:
            if time in project.inventory.success_history.keys():
                project.inventory.success_history[time] += 1
            else:
                project.inventory.success_history[time] = 1
        else:
            if time in project.inventory.fail_history.keys():
                project.inventory.fail_history[time] += 1
            else:
                project.inventory.fail_history[time] = 1

        project.inventory.log_project_data_collector(
            project, null=False, success=success
        )

        return success

    def to_string(self, project):
        """Returns json formatted string summary of project success
        probability and components, for printing or saving."""
        self.get_component_values(project)
        output = {
            'ovr (value, prob): ': (
                self.ovr,
                self.probability_ovr.get_values(self.ovr)
            ),
            'skill balance (value, prob): ': (
                self.skill_balance,
                self.probability_skill_balance.get_values(self.skill_balance)
            ),
            'creativity match (value, prob): ': (
                self.creativity_match,
                self.probability_creativity_match.get_values(
                    self.creativity_match)
            ),
            'risk (value, prob): ': (
                self.risk,
                self.probability_risk.get_values(self.risk)
            ),
            'chemistry (value, prob): ': (
                self.chemistry,
                self.probability_chemistry.get_values(self.chemistry)
            )
        }
        return json.dumps(output, indent=4)
