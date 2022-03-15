"""
SuperScript configuration parameters.
===========

Each parameter is explained in the comments.

Variables whose value can be overridden in the model SuperScriptModel()
constructor are indicated with a [*] at the beginning of the comment.
"""

# Team:
TEAM_OVR_MULTIPLIER = 20  # Used to compute team OVR metric
MIN_TEAM_SIZE = 3  # Minimum number of workers in a team
MAX_TEAM_SIZE = 7  # Maximum number of workers in a team
ORGANISATION_STRATEGY = "Random"  # [*]Selects strategy for Team allocation. Can take ["Random", "Basin" "Basic"]
WORKER_STRATEGY = "AllIn"  # [*]Selects strategy for used by workers to bid. Can take ["AllIn", "Stake"]

# Department:
DEPARTMENTAL_WORKLOAD = 0.1  # [*]Fraction of departmental capacity that must be held back to meet workload.
WORKLOAD_SATISFIED_TOLERANCE = 2  # Minimum number of units slack when a worker is checking if they can bid.
UNITS_PER_FTE = 10  # Number of units equal to full-time equivalent.

# Worker (and Project):
HARD_SKILLS = ['A', 'B', 'C', 'D', 'E']  # These are the so-called 'hard' skills.
SOFT_SKILLS = ['F', 'G', 'H', 'I', 'J']  # These are the so-called 'soft' skills.

# Worker:
MAX_SKILL_LEVEL = 5  # Maximum allowed level for hard and soft skills.
MIN_SOFT_SKILL_LEVEL = 1  # Minimum possible values for soft skills.
P_HARD_SKILL = 0.8  # Probability that each hard skill exists (i.e. is non-zero) when worker created.
WORKER_OVR_MULTIPLIER = 20  # Multiplier for calculating worker OVR.
WORKER_SUCCESS_HISTORY_LENGTH = 4  # Number of previous projects to track in social network.
WORKER_SUCCESS_HISTORY_THRESHOLD = 0.75  # Required successful fraction of tracked projects to produce 'momentum'
SKILL_DECAY_FACTOR = 0.99  # Decay multiplier for unused skills.
ROI_RETURN_DICT = {
    True: 50,
    False: 10,
    'active': 5,
    'train': 5,
    'dept': 10
}   # The return on investment that is given by workers in different states: project success, fail, active on project,
# training and conducting departmental workload.
# Distribution parameters for peer assessment (success and fail):
PEER_ASSESSMENT_SUCCESS_MEAN = 1.05
PEER_ASSESSMENT_SUCCESS_STDEV = 0.2
PEER_ASSESSMENT_FAIL_MEAN = 0.95
PEER_ASSESSMENT_FAIL_STDEV = 0.2
PEER_ASSESSMENT_WEIGHT = 0.25
UPDATE_SKILL_BY_RISK_FLAG = True  # Whether to include the stage in skill update which is dependent on project risk.
REPLACE_AFTER_INACTIVE_STEPS = 10  # Worker removed from simulation after this many timesteps (worker turnover).

# General parameters:
PRINT_DECIMALS_TO = 1  # Number of decimal points to use when printing.

# Project parameters:
TIMELINE_FLEXIBILITY = "TimelineFlexibility"  # [*]Flexibility allowed? Values: ["TimelineFlexibility", "NoFlexibility"]
MAXIMUM_TIMELINE_FLEXIBILITY = 4  # Maximum project start time offset/
PROJECT_LENGTH = 5  # [*]Project length in timesteps.
DEFAULT_START_OFFSET = 0  # Default offset measured in timesteps from project creation.
DEFAULT_START_TIME = 0  # Default project start time.
SAVE_PROJECTS = False  # [*]Whether to save the projects created during the simulation to disk for re-use later.
LOAD_PROJECTS = False  # [*]Whether to load pre-defined projects from disk.

# Project requirements
P_HARD_SKILL_PROJECT = 0.8  # Probability that each hard skill is required by a project.
PER_SKILL_MAX_UNITS = 7  # Maximum number of units that can be required per hard skill.
PER_SKILL_MIN_UNITS = 1  # Minimum number of units that can be required per hard skill.
MIN_REQUIRED_UNITS = 2  # Minimum number of total units that can be required by a project.
MIN_SKILL_LEVEL = 1  # Minimum skill level that can be required.
MAX_SKILL_LEVEL = 5  # Maximum skill level that can be required.
MIN_PROJECT_CREATIVITY = 1  # Minimum desired creativity level.
MAX_PROJECT_CREATIVITY = 5  # Maximum desired creativity level.
RISK_LEVELS = [5, 10, 25]  # Possible project risk values.
P_BUDGET_FLEXIBILITY = 0.25  # [*]Probability that each project has a flexible budget.
MAX_BUDGET_INCREASE = 1.25  # [*]Multiplier that used to calculate  max budget flex.

# Model
NEW_PROJECTS_PER_TIMESTEP = 2  # [*]Number of new projects created on each timestep.
DEPARTMENT_COUNT = 10  # [*]Number of departments in the simulation (must be >= number of workers).
WORKER_COUNT = 100  # [*]Total number of workers in the simulation (constant).
BUDGET_FUNCTIONALITY_FLAG = True  # [*]Can be used to switch off budget functionality. (If False, budget is infinite.)
IO_DIR = './simulation_io/'  # [*]Directory to save projects and network to (or to read project file from).
REP_ID = 0  # ID number for simulation replicate.
SAVE_NETWORK = True  # [*]Whether to save the social network.

# Functions (parameters for the functions produced by function.FunctionFactory):
SUCCESS_PROBABILITY_OVR_GRADIENT = 0.75
SUCCESS_PROBABILITY_SKILL_BALANCE_GRADIENT = -2.5
SUCCESS_PROBABILITY_CREATIVITY_MATCH_RATE = 40
SUCCESS_PROBABILITY_CREATIVITY_MATCH_INTERCEPT = 10
SUCCESS_PROBABILITY_RISK_GRADIENT = -0.75
SUCCESS_PROBABILITY_RISK_INTERCEPT = 15/4
SUCCESS_PROBABILITY_CHEMISTRY_GRADIENT = 2.5
SKILL_UPDATE_BY_RISK_GRADIENT = 0.01
SKILL_UPDATE_BY_RISK_INTERCEPT = 1.0

# Trainer:
TRAINING_ON = True  # [*]Training functionality on/off
TRAINING_MODE = 'slots'  # [*]Training mode to use. Values: ['slots', 'all'].
TRAINING_LENGTH = 5  # Length of a training course.
TRAINING_COMMENCES = 10  # [*]Training does not commence until this timestep.
TARGET_TRAINING_LOAD = 0.1  # [*]Fraction of workforce that should be in training at any time (used by 'slots').

# Network:
HISTORICAL_SUCCESS_RATIO_THRESHOLD = 0.5  # Fraction of team that have worked together on successful projects.

# Optimisation:
NUMBER_OF_PROCESSORS = 8  # [*]Number of processors (cores) to use in parallel basinhopping.
NUMBER_OF_BASIN_HOPS = 10  # [*]Number of basinhopping iterations to run.


