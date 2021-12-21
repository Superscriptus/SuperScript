"""
SuperScript local simulation runner for running replicates of Presets.

Note: also runs three re-runs of crashed simulations (due to unsafe use of np.random.choice with NaNs).
Note: if running remotely remember to pipe stdout and stderr to dev null: &>/dev/null 
===========

Data are saved to disk at: simulation_io/streamlit/<BATCH_NAME> and log files are produced for each
simulation and for the batch (main.replicate.log).
"""


import os
import pickle
import shutil
import sys
import pathlib
import time
import itertools
import multiprocessing
import numpy as np

MODEL_DIR = './superscript_model'
sys.path.append(os.path.normpath(MODEL_DIR))
from superscript_model import model

# These global configuration values override config.py and will be
# used in all the simulations:
REPLICATE_REASON = 'crash' # 'replicate'
REPLICATE_OFFSET = 0 # How many simulation have previously been run for these parameter values?
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

crashed = [
    # These are crashed simulations: 
    [10, 0.99, 0.3, 0.0, 0],
    [10, 0.995, 0.3, 0.1, 1],
    [10, 0.995, 0.3, 0.3, 1]
] 
# The following is all of the presets A-D with variations on SD and TL:
combinations = [
    # These are the presets:
    [10, 0.95, 0.3, 0.3, 1],
    [10, 0.95, 0.3, 0.0, 1],
    [10, 0.95, 0.3, 0.1, 1],
    [10, 0.95, 0.3, 2.0, 1],
    [10, 0.995, 0.3, 0.3, 1],
    [10, 0.99, 0.3, 0.3, 1],
    [1, 0.95, 0.1, 0.0, 1],
    [1, 0.995, 0.1, 0.0, 1],
    [1, 0.99, 0.1, 0.0, 1],
    [1, 0.95, 0.1, 0.1, 1],
    [1, 0.95, 0.1, 0.3, 1],
    [1, 0.95, 0.1, 2.0, 1],
    [3, 0.995, 0.1, 0.1, 1],
    [3, 0.95, 0.1, 0.1, 1],
    [3, 0.99, 0.1, 0.1, 1],
    [3, 0.995, 0.1, 0.0, 1],
    [3, 0.995, 0.1, 0.3, 1],
    [3, 0.995, 0.1, 2.0, 1],
    [2, 0.95, 0.3, 0.0, 1],
    [2, 0.995, 0.3, 0.0, 1],
    [2, 0.99, 0.3, 0.0, 1],
    [2, 0.95, 0.3, 0.1, 1],
    [2, 0.95, 0.3, 0.3, 1],
    [2, 0.95, 0.3, 2.0, 1],
]


def print_log(text, batch_name, save_dir):

    with open(os.path.join(save_dir, batch_name + '.log'), 'a') as outfile:
        outfile.write(text)


def print_main_log(text):
    with open('./simulation_io/streamlit/main.%s.log' % REPLICATE_REASON, 'a') as outfile:
        outfile.write(text)


def run_sim(parameter):

    total_time = 0

    new_projects = parameters[0]
    skill_decay = parameters[1]
    departmental_workload = parameters[2]
    training_load = 0.1 if parameters[3] == 2.0 else parameters[3]
    training_boost = True if parameters[3] == 2.0 else False
    training_flag = False if training_load == 0.0 else True
    budget_functionality = parameters[4]

    batch_name = (
            'pps_%d_sd_%.3f_dw_%.1f_tl_%.1f_tf_%d_tb_%d_bf_%d_010921_v1.1'
            % (
                new_projects, skill_decay, departmental_workload,
                training_load, training_flag, training_boost, budget_functionality
            )
    )
    save_dir = './simulation_io/streamlit/' + batch_name
    path = pathlib.Path(save_dir) 
    os.mkdir(path, exist_ok=True)

    for sim_type in SIMULATIONS.keys():

        sim_io_dir = save_dir + '/' + sim_type
        path = pathlib.Path(sim_io_dir) 
        os.mkdir(path, exist_ok=True)
        print_log("\nSimulation: " + sim_type, batch_name, save_dir)

        if sim_type != 'Random':
            shutil.copyfile(
                save_dir + '/Random/project_file.pickle',
                sim_io_dir + '/project_file.pickle'
            )

        for rc in range(REPLICATE_COUNT):
            ri = rc + REPLICATE_OFFSET 
            print_log("\t replicate: " + str(ri), batch_name, save_dir)

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
                    io_dir=sim_io_dir,
                    skill_decay_factor=skill_decay
                )

                start_time = time.time()

                if training_boost:
                    abm.run_model(int(np.ceil(STEPS/2)))
                    abm.trainer.training_boost()
                    abm.run_model(int(np.ceil(STEPS/2)))
                else:
                    abm.run_model(STEPS)

                elapsed_time = time.time() - start_time
                print_log(
                    "Took %.2f seconds to run %d steps."
                    % (elapsed_time, STEPS),
                    batch_name, save_dir
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

                total_time += elapsed_time

            except Exception as e:
                print_main_log("\nCould not complete simulation: " + batch_name)
                print_main_log("\n" + str(e))

    return total_time


if __name__ == "__main__":

    begin = time.time()

    pool_obj = multiprocessing.Pool(6)
    run_times = pool_obj.map(run_sim, crashed)
    # run_times = pool_obj.map(run_sim, combinations)

    all_run_time = time.time() - begin

    with open('./simulation_io/streamlit/main.%s.log' % REPLICATE_REASON, 'a') as outfile:

        outfile.write("\n")
        for t in run_times:
            outfile.write(str(t))

        outfile.write("\nSum: " + str(np.nansum(run_times)))
        outfile.write("\nNaNCount: " + str(sum(np.isnan(run_times))))
        outfile.write("\nAll run time: " + str(all_run_time))



