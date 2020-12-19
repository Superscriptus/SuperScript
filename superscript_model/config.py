# team:
TEAM_OVR_MULTIPLIER = 20
MIN_TEAM_SIZE = 3
MAX_TEAM_SIZE = 7

# department:
DEPARTMENTAL_WORKLOAD = 0.1
WORKLOAD_SATISFIED_TOLERANCE = 2  # minimum number of units slack when a worker is checking if they can bid
UNITS_PER_FTE = 10  # number of units equal to full-time equivalent

# worker (and project):
HARD_SKILLS = ['A', 'B', 'C', 'D', 'E']
SOFT_SKILLS = ['F', 'G', 'H', 'I', 'J']
# worker:
MAX_SKILL_LEVEL = 5
MIN_SOFT_SKILL_LEVEL = 1
P_HARD_SKILL = 0.8
WORKER_OVR_MULTIPLIER = 20
WORKER_SUCCESS_HISTORY_LENGTH = 4
WORKER_SUCCESS_HISTORY_THRESHOLD = 0.75
SKILL_DECAY_FACTOR = 0.99

# general:
PRINT_DECIMALS_TO = 1

# project:
MAXIMUM_TIMELINE_FLEXIBILITY = 4
PROJECT_LENGTH = 5
DEFAULT_START_OFFSET = 0
DEFAULT_START_TIME = 0

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
MAX_BUDGET_INCREASE = 0.25

# model
NEW_PROJECTS_PER_TIMESTEP = 20
DEPARTMENT_COUNT = 10
WORKER_COUNT = 1000

# functions:
SUCCESS_PROBABILITY_OVR_GRADIENT = 0.75
SUCCESS_PROBABILITY_SKILL_BALANCE_GRADIENT = -2.5
SUCCESS_PROBABILITY_CREATIVITY_MATCH_RATE = 40
SUCCESS_PROBABILITY_CREATIVITY_MATCH_INTERCEPT = 10
SUCCESS_PROBABILITY_RISK_GRADIENT = -0.75
SUCCESS_PROBABILITY_RISK_INTERCEPT = 15/4
SUCCESS_PROBABILITY_CHEMISTRY_GRADIENT = 2.5

# trainer:
TRAINING_LENGTH = 5
TRAINING_COMMENCES = 10

# network:
HISTORICAL_SUCCESS_RATIO_THRESHOLD = 0.5
