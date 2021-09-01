"""
SuperScript local simulation batch runner.
===========

This script runs a batch simulations for all the parameter combinations that will be displayed in the
Streamlit application. Their data are saved to disk at: simulation_io/<BATCH_NAME>

It is best to pipe the standard output to a log file, as shown below.

Usage:  python batch_run_simulation_all.py >> simulation_io/<BATCH_NAME>.log
"""


import os
import pickle
import shutil
import sys
import time
import itertools
import multiprocessing

#BATCH_NAME = 'new_skill_decay_project_per_step_2_dep_wl_03_110621_v1.0'
MODEL_DIR = './superscript_model'
sys.path.append(os.path.normpath(MODEL_DIR))
from superscript_model import model

# These global configuration values override config.py and will be
# used in all the simulations:
REPLICATE_COUNT = 1  # Number of replicate simulations to run
STEPS = 100  # Number of time steps for each simulation
WORKER_COUNT = 100  # Total number of workers in simulation
NUMBER_OF_PROCESSORS = 8  # Number of cores to use for parallel optimiser
P_BUDGET_FLEXIBILITY = 0.25
MAX_BUDGET_INCREASE = 1.25

# This dictionary defines the specific simulations and their distinct
# configuration parameters:
SIMULATIONS = {
    'Random': {
        'WORKER_STRATEGY': 'AllIN',
        'ORGANISATION_STRATEGY': 'Random',
        'TIMELINE_FLEXIBILITY': 'NoFlexibility',
        'SAVE_PROJECTS': False,
        'LOAD_PROJECTS': True
    },
    'Basic': {
        'WORKER_STRATEGY': 'Stake',
        'ORGANISATION_STRATEGY': 'Basic',
        'TIMELINE_FLEXIBILITY': 'NoFlexibility',
        'SAVE_PROJECTS': False,
        'LOAD_PROJECTS': True
    },
    'Niter0': {
        'WORKER_STRATEGY': 'Stake',
        'ORGANISATION_STRATEGY': 'Basin',
        'TIMELINE_FLEXIBILITY': 'NoFlexibility',
        'NUMBER_OF_BASIN_HOPS': 0,
        'SAVE_PROJECTS': False,
        'LOAD_PROJECTS': True
    },
    'Basin': {
        'WORKER_STRATEGY': 'Stake',
        'ORGANISATION_STRATEGY': 'Basin',
        'TIMELINE_FLEXIBILITY': 'NoFlexibility',
        'NUMBER_OF_BASIN_HOPS': 10,
        'SAVE_PROJECTS': False,
        'LOAD_PROJECTS': True
    },
    'Basin_w_flex': {
        'WORKER_STRATEGY': 'Stake',
        'ORGANISATION_STRATEGY': 'Basin',
        'TIMELINE_FLEXIBILITY': 'TimelineFlexibility',
        'NUMBER_OF_BASIN_HOPS': 10,
        'SAVE_PROJECTS': False,
        'LOAD_PROJECTS': True
    }
}
# Details are as follows:
#
#    Time flexibility:
#    Without: 'Random', 'Basic', 'Niter0', 'Basin'
#    With: 'Basin_w_flex'
#
#    Worker strategy:
#    Using 'AllIn': 'Random'
#    Using 'Stake': 'Basic', 'Niter0', 'Basin', 'Basin_w_flex'

PPS = [1, 2, 3, 5, 10]
SD = [0.95, 0.99, 0.995]
DW = [0.1, 0.3]
TL = [0.1, 0.3, 0.0, 2.0]
BF = [0, 1]

combinations = list(itertools.product(PPS, SD, DW, TL, BF))
#print(combinations)
#print(len(combinations))


def print_sim(params):

    print("PPS = ", params[0])
    print("SD = ", params[1])
    print("DW = ", params[2])
    print("TL = ", params[3])
    print("BF = ", params[4])


def run_sim(batch_name, batch_id, parameters):

    save_dir = './simulation_io/' + batch_name
    os.mkdir(save_dir)

    new_projects = parameters[0]
    skill_decay = parameters[1]
    departmental_workload = parameters[2]
    training_load = 0.1 if parameters[3]==2.0 else parameters[3]
    training_flag = False if training_load==0.0 else True
    budget_functionality = parameters[4]

    for sim_type in SIMULATIONS.keys():

        sim_io_dir = save_dir + '/' + sim_type
        os.mkdir(sim_io_dir)
        print("Simulation: ", sim_type)

        if sim_type != 'Random':
            shutil.copyfile(
                save_dir + '/Random/project_file.pickle',
                sim_io_dir + '/project_file.pickle'
            )

        for ri in range(REPLICATE_COUNT):
            print("\t replicate: ", ri)

            if sim_type == 'Random' and ri == 0:
                save_projects = True
                load_projects = False
            else:
                save_projects = (
                    SIMULATIONS[sim_type]['SAVE_PROJECTS']
                )
                load_projects = (
                    SIMULATIONS[sim_type]['LOAD_PROJECTS']
                )

            try:
                abm = model.SuperScriptModel(
                    worker_count=WORKER_COUNT,
                    new_projects_per_timestep=new_projects,
                    number_of_processors=NUMBER_OF_PROCESSORS,
                    departmental_workload=departmental_workload,
                    budget_functionality_flag=budget_functionality,
                    training_on=training_flag,
                    target_training_load=training_load,
                    p_budget_flexibility=P_BUDGET_FLEXIBILITY,
                    max_budget_increase=MAX_BUDGET_INCREASE,
                    worker_strategy=SIMULATIONS[sim_type]['WORKER_STRATEGY'],
                    organisation_strategy=(
                        SIMULATIONS[sim_type]['ORGANISATION_STRATEGY']
                    ),
                    timeline_flexibility=(
                        SIMULATIONS[sim_type]['TIMELINE_FLEXIBILITY']
                    ),
                    number_of_basin_hops=(
                        SIMULATIONS[sim_type].get(
                            'NUMBER_OF_BASIN_HOPS', 0
                        )
                    ),
                    save_projects=save_projects,
                    load_projects=load_projects,
                    io_dir=sim_io_dir
                )

                start_time = time.time()
                abm.run_model(STEPS)
                # abm.run_model(50)
                # abm.trainer.training_boost()
                # abm.run_model(50)
                elapsed_time = time.time() - start_time
                print(
                    "Took %.2f seconds to run %d steps."
                    % (elapsed_time, STEPS)
                )

                tracked = abm.datacollector.get_model_vars_dataframe()

                with open(sim_io_dir
                          + '/model_vars_rep_%d.pickle'
                          % ri, 'wb') as ofile:

                    pickle.dump(tracked, ofile)

                projects = abm.datacollector.get_table_dataframe("Projects")

                with open(sim_io_dir
                          + '/projects_table_rep_%d.pickle'
                          % ri, 'wb') as ofile:

                    pickle.dump(projects, ofile)

                agents = abm.datacollector.get_agent_vars_dataframe()

                with open(sim_io_dir
                          + '/agents_vars_rep_%d.pickle'
                          % ri, 'wb') as ofile:

                    pickle.dump(agents, ofile)

            except:
                print("Could not complete simulation.")


pool_obj = multiprocessing.Pool(4)
answer = pool_obj.map(print_sim, combinations)
#print(answer)


if __name__ == "__main__":

