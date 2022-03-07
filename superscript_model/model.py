"""
SuperScript main model module
===========

Classes:
    SuperScriptModel
        Subclass of mesa.Model
        Handles model setup. Implements run_model() and step() methods.
        Defines model level variables, which can be controlled via
        the mesa GUI.
"""
from mesa import Model
from mesa.time import RandomActivation
import networkx as nx

from .worker import Worker
from .project import ProjectInventory
from .network import SocialNetwork
from .optimisation import OptimiserFactory
from .organisation import (TeamAllocator,
                           Department,
                           Trainer)
from .tracking import (SSDataCollector,
                       on_projects,
                       no_projects,
                       on_training)
from .config import (PROJECT_LENGTH,
                     NEW_PROJECTS_PER_TIMESTEP,
                     WORKER_COUNT,
                     DEPARTMENT_COUNT,
                     TRAINING_ON,
                     TRAINING_MODE,
                     TARGET_TRAINING_LOAD,
                     TRAINING_COMMENCES,
                     BUDGET_FUNCTIONALITY_FLAG,
                     PEER_ASSESSMENT_SUCCESS_MEAN,
                     PEER_ASSESSMENT_SUCCESS_STDEV,
                     PEER_ASSESSMENT_FAIL_MEAN,
                     PEER_ASSESSMENT_FAIL_STDEV,
                     PEER_ASSESSMENT_WEIGHT,
                     UPDATE_SKILL_BY_RISK_FLAG,
                     REPLACE_AFTER_INACTIVE_STEPS,
                     ORGANISATION_STRATEGY,
                     WORKER_STRATEGY,
                     SAVE_PROJECTS,
                     LOAD_PROJECTS,
                     IO_DIR,
                     SAVE_NETWORK,
                     SAVE_NETWORK_FREQUENCY,
                     DEPARTMENTAL_WORKLOAD,
                     TIMELINE_FLEXIBILITY,
                     NUMBER_OF_BASIN_HOPS,
                     NUMBER_OF_PROCESSORS,
                     P_BUDGET_FLEXIBILITY,
                     MAX_BUDGET_INCREASE,
                     SKILL_DECAY_FACTOR)


class SuperScriptModel(Model):
    """Subclass of mesa.Model

    Note:
        Many config variables are defined as attributes in this class,
        and then passed in to the classes where they are actually
        used. (Or model is passed and the variable is accessed via
        model.<varname>). This is to allow these config parameters to
        be set by the Mesa GUI controls, which requires that model
        level variables are used.
    ...

    Attributes:
        worker_count: int
            Number of workers in the simulation.
        new_projects_per_timestep:
            How many new projects to create on each timestep.
        project_length: int
            Length of each project in timesteps.
        budget_functionality_flag: bool
            If True, budget contraint is switched on.
        p_budget_flexibility: float
            Probability that an individual project has a flexible
            budget.
        max_budget_increase: float
            Multiplier that defines how much flexibility there is for
            a project that has a flexible budget (1.0 = no flex).
        new_workers: int
            Counts number of new workers added during simulation.
            Used to ensure that a unique worker_id is assigned
            when new workers are created.
        number_of_processors: int
            Number of processors to use (in parallel optimisation).
        number_of_basin_hops: int
            Number of basinhopping steps (if using basinhopping
            optimiser).
        departments: dict
            Dictionary that stores departments.
        peer_assessment_success_mean: float
            Mean value of distribution used in peer assessment,
            when project is successful.
        peer_assessment_success_stdev: float
            Standard deviation value of distribution used in peer
            assessment, when project is successful.
        peer_assessment_fail_mean: float
            Mean value of distribution used in peer assessment,
            when project fails.
        peer_assessment_fail_stdev: float
            Standard deviation value of distribution used in peer
            assessment, when project fails.
        peer_assessment_weight: float
            Weight used when combining peer assessment score with
            current skill value.
        update_skill_by_risk_flag: bool
            Whether to include the stage in skill update that boosts
            worker skill by larger amounts for riskier (successful)
            projects.
        replace_after_inactive_steps: int
            Number of timesteps of inactivity after which a worker
            is replaced.
        organisation_strategy: str
            Strategy to use for team allocation.
            Takes one of: "Random", "Basic", "Basin"
        worker_strategy: str
            Strategy for worker bidding.
            Takes one of: "AllIn", "Stake"
        G: nx.Graph
            Base graph for social network/
        grid: network.SocialNetwork
            Stores number of successful collaboration between each pair
            of workers.
        save_network_flag: bool
            Whether to save the social network for later analysis.
        save_network_freq: int
            How often to save the network (in number of timesteps).
        schedule: mesa.time.RandomActivation
            Mesa scheduler (determines order in which workers are
            updated).
        io_dir: str
            Path to directory for reading/writing projects.
        save_projects: bool
            Whether to save projects for use in later simulations.
        load_projects: bool
            Whether to load predefined projects from a previous
            simulation.
        timeline_flexibility: str
            Indicates type of timeline flexibility to use.
            (currently just switches flexibility on or off). .
        inventory: project.ProjectInventory
            Creates and keeps track of projects.
        training_on: bool
            Whether training is activated for this simulation.
        training_mode: str
            Mode to use for training.
            Currently: 'all' or 'slots'
        target_training_load: float
            Fraction of workforce that should be engaged in training.
        training_commences: int
            Timestep at which training starts (if activated).
        trainer: organisation.Trainer
            Handles all training of workers.
        departmental_workload: float
            Fraction of department capacity that needs to be held back
            to do departmental work.
        worker_turnover: dict
            Records how many workers are replaced on each timstep.
        running: bool
            Required by mesa to indicate that simulation is active.
        datacollector: tracking.SSDataCollector
            Leverages mesa functionality to save simulation data for
            later analysis.
        skill_decay_factor: float
            Multiplicative skill decay for unused hard skills
    """

    def __init__(self, worker_count=WORKER_COUNT,
                 department_count=DEPARTMENT_COUNT,
                 new_projects_per_timestep=NEW_PROJECTS_PER_TIMESTEP,
                 project_length=PROJECT_LENGTH,
                 training_on=TRAINING_ON,
                 training_mode=TRAINING_MODE,
                 target_training_load=TARGET_TRAINING_LOAD,
                 training_commences=TRAINING_COMMENCES,
                 budget_functionality_flag=BUDGET_FUNCTIONALITY_FLAG,
                 peer_assessment_success_mean=PEER_ASSESSMENT_SUCCESS_MEAN,
                 peer_assessment_success_stdev=PEER_ASSESSMENT_SUCCESS_STDEV,
                 peer_assessment_fail_mean=PEER_ASSESSMENT_FAIL_MEAN,
                 peer_assessment_fail_stdev=PEER_ASSESSMENT_FAIL_STDEV,
                 peer_assessment_weight=PEER_ASSESSMENT_WEIGHT,
                 update_skill_by_risk_flag=UPDATE_SKILL_BY_RISK_FLAG,
                 replace_after_inactive_steps=REPLACE_AFTER_INACTIVE_STEPS,
                 organisation_strategy=ORGANISATION_STRATEGY,
                 worker_strategy=WORKER_STRATEGY,
                 io_dir=IO_DIR,
                 save_projects=SAVE_PROJECTS,
                 load_projects=LOAD_PROJECTS,
                 save_network=SAVE_NETWORK,
                 save_network_freq=SAVE_NETWORK_FREQUENCY,
                 departmental_workload=DEPARTMENTAL_WORKLOAD,
                 timeline_flexibility=TIMELINE_FLEXIBILITY,
                 number_of_processors=NUMBER_OF_PROCESSORS,
                 number_of_basin_hops=NUMBER_OF_BASIN_HOPS,
                 p_budget_flexibility=P_BUDGET_FLEXIBILITY,
                 max_budget_increase=MAX_BUDGET_INCREASE,
                 skill_decay_factor=SKILL_DECAY_FACTOR):

        self.worker_count = worker_count
        self.new_projects_per_timestep = new_projects_per_timestep
        self.project_length = project_length
        self.budget_functionality_flag = budget_functionality_flag
        self.p_budget_flexibility = p_budget_flexibility
        self.max_budget_increase = max_budget_increase
        self.new_workers = 0
        self.departments = dict()

        self.number_of_processors = number_of_processors
        self.number_of_basin_hops = number_of_basin_hops

        self.peer_assessment_success_mean = peer_assessment_success_mean
        self.peer_assessment_success_stdev = peer_assessment_success_stdev
        self.peer_assessment_fail_mean = peer_assessment_fail_mean
        self.peer_assessment_fail_stdev = peer_assessment_fail_stdev
        self.peer_assessment_weight = peer_assessment_weight
        self.update_skill_by_risk_flag = update_skill_by_risk_flag
        self.replace_after_inactive_steps = replace_after_inactive_steps
        self.organisation_strategy = organisation_strategy
        self.worker_strategy = worker_strategy

        self.G = nx.Graph()
        self.grid = SocialNetwork(self, self.G)
        self.save_network_flag = save_network
        self.save_network_freq = save_network_freq

        self.schedule = RandomActivation(self)
        self.io_dir = io_dir
        self.save_projects = save_projects
        self.load_projects = load_projects
        self.timeline_flexibility = timeline_flexibility
        self.inventory = ProjectInventory(
            TeamAllocator(self, OptimiserFactory()),
            timeline_flexibility=self.timeline_flexibility,
            social_network=self.grid,
            model=self,
            save_flag=self.save_projects,
            load_flag=self.load_projects,
            io_dir=self.io_dir
        )
        self.training_on = training_on
        self.training_mode = training_mode
        self.target_training_load = target_training_load
        self.training_commences = training_commences
        self.trainer = Trainer(self)

        self.departmental_workload = departmental_workload
        for di in range(department_count):
            self.departments[di] = Department(
                di, workload=self.departmental_workload
            )

        self.skill_decay_factor = skill_decay_factor
        workers_per_department = worker_count / department_count
        assert workers_per_department * department_count == worker_count

        di = 0
        assigned_to_di = 0
        for i in range(self.worker_count):
            w = Worker(
                i, self, self.departments[di],
                skill_decay_factor=self.skill_decay_factor
            )
            self.schedule.add(w)

            assigned_to_di += 1
            if assigned_to_di == workers_per_department:
                di += 1
                assigned_to_di = 0

        self.grid.initialise()
        self.worker_turnover = dict()
        self.running = True
        self.datacollector = SSDataCollector()

    @property
    def time(self):
        """Returns current scheduler step."""
        return self.schedule.steps

    def step(self):
        """Step method to advance simulation by one timetsep.
        """

        for agent in self.schedule.agents:
            agent.skills.reset_skill_change_trackers()

        self.trainer.update_skill_quartiles()
        self.inventory.create_projects(self.new_projects_per_timestep,
                                       self.time, self.project_length)

        for di, dept in self.departments.items():
            dept.assign_work()

        self.schedule.step()
        self.trainer.train()
        self.inventory.remove_null_projects()

        if (self.save_network_flag
                and self.time % self.save_network_freq == 0):
            self.grid.save()

        self.datacollector.collect(self)
        assert (on_projects(self)
                + no_projects(self)
                + on_training(self)
                == self.worker_count)

    def run_model(self, step_count: int):
        """Run model for a number of timesteps.

        Note:
            Saves projects at the end of the run, if enabled.

        Args:
            step_count: int
                Number of steps to take.
        """
        for i in range(step_count):
            self.step()

        self.inventory.save_projects()
