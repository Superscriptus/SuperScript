# team:
TEAM_OVR_MULTIPLIER = 20
MIN_TEAM_SIZE = 3
MAX_TEAM_SIZE = 7
ORGANISATION_STRATEGY = "Random" #"Basin" #"Random"
WORKER_STRATEGY = "AllIn" #"Stake" #"AllIn"

# department:
DEPARTMENTAL_WORKLOAD = 0.1
WORKLOAD_SATISFIED_TOLERANCE = 2  # minimum number of units slack when a worker is checking if they can bid
UNITS_PER_FTE = 10  # number of units equal to full-time equivalent

# worker (and project):
HARD_SKILLS = ['A', 'B', 'C', 'D', 'E']
SOFT_SKILLS = ['F', 'G', 'H', 'I', 'J']

"""Worker parameters:
"""
# Maximum level for hard and soft skills:
MAX_SKILL_LEVEL = 5
# Minimum possible values for soft skills:
MIN_SOFT_SKILL_LEVEL = 1
# Probability that each hard skill is non-zero when worker created:
P_HARD_SKILL = 0.8
# Multiplier for calculating worker OVR:
WORKER_OVR_MULTIPLIER = 20
# Parameters for worker success history ('momentum'):
WORKER_SUCCESS_HISTORY_LENGTH = 4
WORKER_SUCCESS_HISTORY_THRESHOLD = 0.75
# Decay multiplier for unused skills:
SKILL_DECAY_FACTOR = 0.99
# Distribution parameters for peer assessment (success and fail):
PEER_ASSESSMENT_SUCCESS_MEAN = 1.05
PEER_ASSESSMENT_SUCCESS_STDEV = 0.2
PEER_ASSESSMENT_FAIL_MEAN = 0.95
PEER_ASSESSMENT_FAIL_STDEV = 0.2
PEER_ASSESSMENT_WEIGHT = 0.25
# Whether to include the stage in skill update which is dependent on
# project risk:
UPDATE_SKILL_BY_RISK_FLAG = True
REPLACE_AFTER_INACTIVE_STEPS = 10

"""General parameters:
"""
# Number of decimal points to use when printing:
PRINT_DECIMALS_TO = 1

"""Project parameters:
"""

MAXIMUM_TIMELINE_FLEXIBILITY = 4
PROJECT_LENGTH = 5
DEFAULT_START_OFFSET = 0
DEFAULT_START_TIME = 0
SAVE_PROJECTS = False
LOAD_PROJECTS = False

# project requirements
P_HARD_SKILL_PROJECT = 0.8
PER_SKILL_MAX_UNITS = 7
PER_SKILL_MIN_UNITS = 1
MIN_REQUIRED_UNITS = 2
MIN_SKILL_LEVEL = 1
MAX_SKILL_LEVEL = 5
MIN_PROJECT_CREATIVITY = 1
MAX_PROJECT_CREATIVITY = 5
RISK_LEVELS = [5,10,25]
P_BUDGET_FLEXIBILITY = 0.25
MAX_BUDGET_INCREASE = 1.25

# model
NEW_PROJECTS_PER_TIMESTEP = 2 #1# 20
DEPARTMENT_COUNT = 10
WORKER_COUNT = 100 # 50 #100
BUDGET_FUNCTIONALITY_FLAG = True
IO_DIR = './simulation_io/' #./'
SAVE_NETWORK = False
SAVE_NETWORK_FREQUENCY = 50

# functions:
SUCCESS_PROBABILITY_OVR_GRADIENT = 0.75
SUCCESS_PROBABILITY_SKILL_BALANCE_GRADIENT = -2.5
SUCCESS_PROBABILITY_CREATIVITY_MATCH_RATE = 40
SUCCESS_PROBABILITY_CREATIVITY_MATCH_INTERCEPT = 10
SUCCESS_PROBABILITY_RISK_GRADIENT = -0.75
SUCCESS_PROBABILITY_RISK_INTERCEPT = 15/4
SUCCESS_PROBABILITY_CHEMISTRY_GRADIENT = 2.5
SKILL_UPDATE_BY_RISK_GRADIENT = 0.01
SKILL_UPDATE_BY_RISK_INTERCEPT = 1.0

# trainer:
TRAINING_ON = True
TRAINING_MODE = 'slots'
TRAINING_LENGTH = 5
TRAINING_COMMENCES = 10
TARGET_TRAINING_LOAD = 10

# network:
HISTORICAL_SUCCESS_RATIO_THRESHOLD = 0.5

# optimisation:
NUMBER_OF_PROCESSORS = 8 
NUMBER_OF_BASIN_HOPS = 10
