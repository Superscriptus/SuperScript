"""
SuperScript worker module
===========

Worker class and associated classes.

Worker is a subclass of mesa.Agent

"""

from mesa import Agent
from interface import Interface, implements
import json

from .project import Project
from .utilities import Random
from .organisation import Department
from .config import (HARD_SKILLS,
                     SOFT_SKILLS,
                     MAX_SKILL_LEVEL,
                     MIN_SOFT_SKILL_LEVEL,
                     P_HARD_SKILL,
                     WORKER_OVR_MULTIPLIER,
                     PRINT_DECIMALS_TO,
                     UNITS_PER_FTE,
                     WORKER_SUCCESS_HISTORY_LENGTH,
                     WORKER_SUCCESS_HISTORY_THRESHOLD,
                     SKILL_DECAY_FACTOR)


class Worker(Agent):
    """Subclass of mesa.Agent

    This class stores the skills of the worker, their contributions to
    projects, and their history. It also handles worker replacement due
    to inactivity.

    The step() method is called by the Mesa scheduler.

    Note:
        Various properties are defined that are only used by the
        SSDataCollector agent_reporters. They return copies of
        the dictionaries that are needed. Previously these were
        defined as tracking methods in tracking.py, and were
        returning references to the same dictionaries. But this
        produced unexpected behaviour e.g. The values in hard_skills
        were not updating in the DataCollector.

    ...

    Attributes:
        worker_id: int
            unique identifier for the worker
        skills: SkillMatrix
            handles worker skills
        department: organisation.Department
            department that this worker belongs to
        strategy: WorkerStrategyInterface
            Selected in config.py. Determines how worker bids for projects
        leads_on: dict
            records which projects this worker leads on. Project leads
            advance their projects on step().
        contributions: WorkerContributions
            tracks which projects the worker contributes to and when
        history: WorkerHistory
            tracks recent project success rate (for calculation of
            'momentum')
        training_remaining: int
            how many timesteps is this worker going to be on training.
            <= 0 if not on training.
        timesteps_inactive: int
            how many timesteps since worker last contributed to a project.
            Worker is replaced if this exceeds threshold
            (REPLACE_AFTER_INACTIVE_STEPS). Set to zero if worker is booked
            to work on future project, to ensure that worker is not
            replaced before this project starts.
    """

    def __init__(self, worker_id: int,
                 model, department=Department(0)):
        """
        Create new worker in specified department and call base class
        constructor to add worker(agent) to scheduler.
        """
        super().__init__(worker_id, model)
        self.worker_id = worker_id
        self.skills = SkillMatrix()
        self.department = department
        self.department.add_worker()

        self.strategy = (
            AllInStrategy("AllIn")
            if self.model.worker_strategy == "AllIn"
            else StakeStrategy("Stake")
        )
        self.leads_on = dict()
        self.contributions = WorkerContributions(self)
        self.history = WorkerHistory()
        self.training_remaining = 0
        self.timesteps_inactive = 0

    @property
    def contributes(self):
        """
        dict: returns dictionary of worker contributions by skill
            at each timestep.
        """
        return self.contributions.get_contributions()

    @property
    def contributes_now(self):
        """
        dict: returns dictionary of worker contributions by skill
              at current timestep (or None if no contributions).
        """
        return (
            self.contributions.get_contributions().get(self.now, None)
        )

    @property
    def now(self):
        """int: current simulation step (from scheduler)"""
        return self.model.schedule.steps

    @property
    def training_horizon(self):
        """int: length of training (defined in config)"""
        return self.model.trainer.training_length

    @property
    def hard_skills(self):
        """Returns a copy of hard_skills dictionary,
        used by DataCollector.
        """
        return self.skills.hard_skills.copy()

    @property
    def training_tracker(self):
        """Returns a copy of tracker dictionary,
        used by DataCollector.
        """
        return self.skills.training_tracker.copy()

    @property
    def skill_decay_tracker(self):
        """Returns a copy of tracker dictionary,
        used by DataCollector.
        """
        return self.skills.skill_decay_tracker.copy()

    @property
    def peer_assessment_tracker(self):
        """Returns a copy of tracker dictionary,
        used by DataCollector.
        """
        return self.skills.peer_assessment_tracker.copy()

    @property
    def worker_ovr(self):
        """Returns current worker OVR value,
        used by DataCollector.
        """
        return self.skills.ovr

    def assign_as_lead(self, project):
        """Assigns this worker as project lead.

        Args:
            project: instance of project.Project
        """
        self.leads_on[project.project_id] = project

    def remove_as_lead(self, project):
        """Removes this worker as project lead."""
        del self.leads_on[project.project_id]

    def step(self):
        """Worker step method, called by Mesa scheduler on each
        simulation timestep.
        First the worker advances all projects that they lead on,
        then any unused skills are subject to decay, then their
        recent activity is checked (for worker replacement).

        Note:
            Projects stored as list before loop, because leads_on
            dict can change size during loop (if project terminates).
        """
        projects = list(self.leads_on.values())

        for project in projects:
            if project in self.leads_on.values():
                project.advance()

        self.skills.decay(self)
        self.check_activity()

    def get_skill(self, skill, hard_skill=True):
        """Get worker level for a specified skill.

        Args:
            skill: str
                Takes value from ['A', 'B, ..., 'J']
            hard_skill: bool, optional
                Indicates if the requested skill is a hard of soft
                skill

        Returns:
            float: skill level

        """
        if hard_skill:
            return self.skills.hard_skills[skill]
        else:
            return self.skills.soft_skills[skill]

    def is_free(self, start, length, slack=None):
        """Checks if worker is free over specified time period (i.e. is
        currently committed at 0% FTE and departmental workload is
        satisfied).

        Args:
            start: int
                Beginning of period to check
            length: int
                Length of period to check
            slack: int, optional
                A buffer added to departmental workload baseline

        Returns:
            Ture if free, False otherwise
        """
        return self.contributions.is_free_over_period(
            start, length, slack
        )

    def check_activity(self):
        """Replaces worker if they have been inactive for more than
        self.model.replace_after_inactive_steps timesteps.

        Note:
            Added new clause to check that worker has not been
            assigned to a project starting within the planning
            horizon. If so, they survive replacement.

        """

        if (self.contributions.get_units_contributed(self.now) > 0
                and self.training_remaining == 0):
            self.timesteps_inactive = 0

        elif sum([self.contributions.get_units_contributed(self.now + t) > 0
                  for t in range(self.model.inventory.max_timeline_flex)]):
            self.timesteps_inactive = 0

        else:
            self.timesteps_inactive += 1

        if (self.timesteps_inactive
                >= self.model.replace_after_inactive_steps):
            self.replace()

    def replace(self):
        """Replaces worker with a new worker in the same department.

            Tracks number of new workers that have been added during
            simulation, to ensure that new worker has a unique
            worker_id.
        """
        if len(self.leads_on) > 0:
            print("warning: replacing worker %d, leads on projects: "
                  % self.worker_id,
                  self.leads_on.keys())

        self.department.number_of_workers -= 1
        self.model.new_workers += 1

        if self.now in self.model.worker_turnover.keys():
            self.model.worker_turnover[self.now] += 1
        else:
            self.model.worker_turnover[self.now] = 1

        w = Worker(
            self.model.worker_count + self.model.new_workers,
            self.model, self.department
        )
        self.model.schedule.add(w)
        self.model.grid.replace_worker(self, w)
        self.model.schedule.remove(self)

    def bid(self, project):
        """Worker decides whether to bid for project, using the
            allocated strategy.

        Args:
              project: project.Project
        """
        return self.strategy.bid(project, self)

    def individual_chemistry(self, project):
        """Computes chemistry at the individual worker level, which
        contributes to calculation of project success probability
        (i.e. chemistry booster).

        Args:
            project: project.Project

        """
        chemistry = 0

        if (len(set(self.skills.top_two)
                .intersection(project.required_skills)) > 0):
            chemistry += 1

        chemistry += self.history.momentum()
        chemistry += project.risk >= 0.1 * self.skills.ovr
        return chemistry

    def peer_assessment(self, success, skill, modifier):
        """Models peer_assessment to update a hard skill at end of a
        project (called by Team.skill_update)

        Args:
            success: bool
                Was project success or failure
            skill: str
                Hard skill to update, takes value in ['A',...,'E']
            modifier: float
                A multiplier that depends on project success and risk,
                provided by 'SkillUpdateByRisk' function.

        """
        if success:
            mean = self.model.peer_assessment_success_mean
            stdev = self.model.peer_assessment_success_stdev
        else:
            mean = self.model.peer_assessment_fail_mean
            stdev = self.model.peer_assessment_fail_stdev

        weight = self.model.peer_assessment_weight
        old_skill = self.get_skill(skill)

        self.skills.hard_skills[skill] = (
            ((1 - weight) + (weight * Random.normal(mean, stdev)))
            * old_skill
        )

        self.skills.hard_skills[skill] *= modifier
        self.skills.hard_skills[skill] = min(
            self.skills.hard_skills[skill],
            self.skills.max_skill
        )
        self.skills.peer_assessment_tracker[skill] = (
            self.skills.hard_skills[skill] - old_skill
        )


class WorkerContributions:
    """Class that logs current and future contributions to projects.

    Attributes:
        per_skill_contributions: dict
            logs contributions of each skill to specific project at
            each timestep
        total_contribution: dict
            logs total number of units contributed to project at each
            timestep
        worker: worker.Worker
            the worker these contributions relate to
            (used to access skills)
        units_per_full_time: int
            Number of units equivalent ot 100% FTE, defined in config
    """
    def __init__(self, worker, units_per_full_time=UNITS_PER_FTE):
        self.per_skill_contributions = dict()
        self.total_contribution = dict()
        self.worker = worker
        self.units_per_full_time = units_per_full_time

    def get_contributions(self, time=None):
        """Returns log of which skill is contributed to which project,
        either at all times or at specific time.

        Args:
            time: int, optional
            Specific time for which to return skill contributions,
            otherwise returns full dict (all times)

        Returns:
            dictionary of contributions
        """
        if time is None:
            return self.per_skill_contributions
        elif time in self.per_skill_contributions.keys():
            return self.per_skill_contributions[time]
        else:
            return {}

    def get_skill_units_contributed(self, time, skill):
        """Return number of units of a specific skill that worker
        contributes at a given time.

        Args:
            time: int
                Timestep at which to count contributions
            skill: str
                Hard skill to return contributions for, takes value in
                ['A',...,'E']

        Returns:
            integer count of number of units contributed
        """
        contributions = self.get_contributions(time)

        if skill in contributions.keys():
            return len(contributions[skill])
        else:
            return 0

    def add_contribution(self, project, skill):
        """Log that worker contributes one unit of this skill to the
         project.

        Args:
            project: project.Project
                Project to which worker contributes this skill
            skill: str
                Hard skill contributed, takes value in ['A',...,'E']
        """
        for time_offset in range(project.length):
            time = project.start_time + time_offset

            if time not in self.per_skill_contributions.keys():
                self.per_skill_contributions[time] = {
                    skill: []
                    for skill in self.worker.skills.hard_skills.keys()
                }
            (self.per_skill_contributions[time][skill]
             .append(project.project_id))

            if time not in self.total_contribution.keys():
                self.total_contribution[time] = 1
            else:
                self.total_contribution[time] += 1

    def get_units_contributed(self, time):
        """Return total units contributed at this timestep

        Args:
            time: int
                timestep for which to return contributions
        """
        if time not in self.total_contribution.keys():
            return 0
        else:
            return self.total_contribution[time]

    def contributes_less_than_full_time(self, start, length):
        """Determine if worker contributes less than full time over
        this time period.

        Args:
            start: int
                Start of time period
            length: int
                Length of time period
        """
        for t in range(length):

            time = start + t
            contributes_at_time = self.get_units_contributed(time)
            if contributes_at_time >= self.units_per_full_time:
                return False

        return True

    def get_remaining_units(self, start, length):
        """Count the number of remaining units that this worker is
        able to contribute over this time period.

        Args:
            start: int
                Start of time period
            length: int
                Length of time period

        Note:
            Number of remaining units may be different at each timestep
            in this period. This method returns the minimum, because is
            the constraint on how much project work they can do.
        """
        remaining_units = []
        for t in range(length):

            time = start + t
            contributes_at_time = self.get_units_contributed(time)
            remaining_units.append(
                self.units_per_full_time - contributes_at_time
            )
        return min(remaining_units)

    def is_free_over_period(self, start, length, slack):
        """Determines if worker is 'free' over this time period.

        'Free' is defined as not working on projects and departmental
        workload is satisfied.

        Note:
            'slack' can be used to add a buffer above the required
             departmental workload

        Args:
            start: int
                Start of time period
            length: int
                Length of time period
            slack: int
                buffer to add to departmental workload
        """
        if ((self.get_remaining_units(start, length)
                == self.units_per_full_time)
            and (self.worker.department
                     .is_workload_satisfied(
                        start, length, slack))):
            return True
        else:
            return False


class WorkerHistory:
    """Class to track recent worker's success rate.

    This is used to determine 'momentum' which is a component of
    individual chemistry.

    Attributes:
        success_history_length: int
            Number of timesteps to track (beyond this value entries
            are removed from the history)
        success_history_threshold: float
            Ratio of successful project in history list that are
            required to produce momentum
        success_history: list
            List of bools that record recent project successes and
            failures
    """
    def __init__(self, success_history_length=WORKER_SUCCESS_HISTORY_LENGTH,
                 success_history_threshold=WORKER_SUCCESS_HISTORY_THRESHOLD):

        self.success_history_length = success_history_length
        self.success_history_threshold = success_history_threshold
        self.success_history = []

    def record(self, success):
        """Log a success (or failure) and remove the first element in
        the history list if it is too long.

        Args:
            success: bool
                Was the project successful
            """
        self.success_history.append(success)
        if len(self.success_history) > self.success_history_length:
            self.success_history.pop(0)

    def get_success_rate(self):
        """Compute recent success rate"""
        if len(self.success_history) == 0:
            return 0
        else:
            return sum(self.success_history) / len(self.success_history)

    def momentum(self):
        """Determine is recent success rate is greater than required
        threshold"""
        return self.get_success_rate() >= self.success_history_threshold


class WorkerStrategyInterface(Interface):
    """Interface class for worker strategies.

    Worker strategies decide if workers will bid for a project.

    In a future version they may also decide if a worker accepts an
    invitation to work on a project. (Currently not in use)
    """
    def bid(self, project: Project, worker: Worker) -> bool:
        pass

    def accept(self, project: Project) -> bool:
        """Currently not in use."""
        pass


class AllInStrategy(implements(WorkerStrategyInterface)):
    """AllIn strategy means that worker will bid for any project.

    Note: this does not respect the staking requirement.
    """
    def __init__(self, name: str):
        self.name = name

    def bid(self, project: Project, worker: Worker) -> bool:

        if (worker.department.is_workload_satisfied(
                project.start_time, project.length)
            and
            worker.contributions.contributes_less_than_full_time(
                project.start_time, project.length)):
            return True
        else:
            return False

    def accept(self, project: Project) -> bool:
        return True


class StakeStrategy(implements(WorkerStrategyInterface)):
    """Stake strategy - a worker will only bid for a project if:
        1. their departmental workload is satified
        2. they are currently working at less than full time for the
        project duration
        3. the project risk is less than or equal to half their OVR
    """
    def __init__(self, name: str):
        self.name = name

    def bid(self, project: Project, worker: Worker) -> bool:

        if (worker.department.is_workload_satisfied(
                project.start_time, project.length)
            and
            worker.contributions.contributes_less_than_full_time(
                project.start_time, project.length)
            and
                project.risk <= 0.5 * worker.skills.ovr):
            return True
        else:
            return False

    def accept(self, project: Project) -> bool:
        return True


class SkillMatrix:
    """Class that handles the skills of a worker.

    Attributes:
        hard_skills: dict
            Records skill level for each hard skill  ['A',..,'E']
        soft_skills: dict
            Records skill level for each soft skill ['F',..,'J']
        max_skill: int
            Maximum possible skill level
        hard_skill_probability: int
            Probability that each hard skill is non-zero on creation
        ovr_multiplier: int
            Multiplier for calculating worker OVR
        skill_decay_factor: float
            Multiplier for decay of unused skills
        round_to: int
            number of decimal places for printing
        peer_assessment_tracker: dict
            Records how much each skill changes due to peer assessment
        skill_decay_tracker: dict
            Records how much each skill changes due to decay
        training_tracker: dict
            Records how much each skill changes due to training

        Note:
            The trackers are reset to zero at the beginning of each
            simulation timestep
    """
    def __init__(self,
                 hard_skills=HARD_SKILLS,
                 soft_skills=SOFT_SKILLS,
                 max_skill=MAX_SKILL_LEVEL,
                 min_soft_skill=MIN_SOFT_SKILL_LEVEL,
                 hard_skill_probability=P_HARD_SKILL,
                 round_to=PRINT_DECIMALS_TO,
                 ovr_multiplier=WORKER_OVR_MULTIPLIER,
                 skill_decay_factor=SKILL_DECAY_FACTOR):

        self.hard_skills = dict(zip(hard_skills,
                                    [0.0 for s in hard_skills]))
        self.soft_skills = dict(
            zip(soft_skills,
                [Random.uniform(min_soft_skill, max_skill)
                 for s in soft_skills]
                )
        )
        self.max_skill = max_skill
        self.hard_skill_probability = hard_skill_probability
        self.ovr_multiplier = ovr_multiplier
        self.skill_decay_factor = skill_decay_factor
        self.round_to = round_to

        while sum(self.hard_skills.values()) == 0.0:
            self.assign_hard_skills()

        self.peer_assessment_tracker = {
            skill: 0 for skill in self.hard_skills
        }
        self.skill_decay_tracker = {
            skill: 0 for skill in self.hard_skills
        }
        self.training_tracker = {
            skill: 0 for skill in self.hard_skills
        }

    @property
    def ovr(self):
        """Calculates worker OVR from hard skills"""
        return (sum([s for s in
                     self.hard_skills.values()
                     if s > 0.0])
                / sum([1 for s in
                       self.hard_skills.values()
                       if s > 0.0])
                ) * self.ovr_multiplier

    @property
    def top_two(self):
        """Returns this workers top two skills (by level)"""
        ranked_skills = {
            k: v for k, v in sorted(
                self.hard_skills.items(),
                reverse=True,
                key=lambda item: item[1]
            )
        }
        return list(ranked_skills.keys())[:2]

    def assign_hard_skills(self):
        """Randomly allocates hard skills to worker"""
        for key in self.hard_skills.keys():
            if Random.uniform() <= self.hard_skill_probability:
                self.hard_skills[key] = Random.uniform(
                    0.0, self.max_skill)

    def decay(self, worker):
        """Decays unused skills by predefined factor"""
        for skill in self.hard_skills.keys():

            units = (
                worker.contributions
                      .get_skill_units_contributed(worker.now, skill)
            )
            if units == 0:
                old_skill = self.hard_skills[skill]
                self.hard_skills[skill] *= self.skill_decay_factor
                self.skill_decay_tracker[skill] = (
                    self.hard_skills[skill] - old_skill
                )

    def reset_skill_change_trackers(self):
        """Resets trackers to zero. Called at beginning of each
        simulation timestep
        """
        for skill in self.hard_skills:
            self.peer_assessment_tracker[skill] = 0
            self.skill_decay_tracker[skill] = 0
            self.training_tracker[skill] = 0

    def to_string(self):
        """Returns worker skills in json formatted string for printing
         or saving
         """
        output = {
            "Worker OVR":
                round(self.ovr, self.round_to),
            "Hard skills":
                [round(s, self.round_to)
                 for s in self.hard_skills.values()],
            "Soft skills":
                [round(s, self.round_to)
                 for s in self.soft_skills.values()],
            "Hard skill probability":
                self.hard_skill_probability,
            "OVR multiplier": self.ovr_multiplier}

        return json.dumps(output, indent=4)
