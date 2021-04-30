# Local simulation batch runner...

import time
import pickle
import sys, os, shutil

# Name for this batch of simulations:
# (Note: should include date and code version number)

BATCH_NAME = 'temp_v0.1' #'test_run_260321_v0.0'

# These global configuration values override config.py and will be
# used in all the simulations:
REPLICATE_COUNT = 1 # Number of replicate simulations to run
STEPS = 2  # Number of time steps for each simulation
WORKER_COUNT = 100  # Total number of workers in simulation
NEW_PROJECTS = 2  # Number of new projects created on each time step
DEPARTMENTAL_WORKLOAD = 0.1  # Fraction of department capacity to keep
                             # free for dept work.
NUMBER_OF_PROCESSORS = 8  # Number of cores to use for parallel optimiser
BUDGET_CONSTRAINT_FLAG = True
TRAINING_FLAG = True 
TRAINING_LOAD = 0.1
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


if __name__ == "__main__":

    MODEL_DIR = './superscript_model'
    SAVE_DIR = './simulation_io/' + BATCH_NAME
    os.mkdir(SAVE_DIR)
    sys.path.append(os.path.normpath(MODEL_DIR))
    from superscript_model import model

    for sim_type in SIMULATIONS.keys():

        sim_io_dir = SAVE_DIR + '/' + sim_type
        os.mkdir(sim_io_dir)
        print("Simulation: ", sim_type)

        if sim_type != 'Random':
            shutil.copyfile(
                SAVE_DIR + '/Random/project_file.pickle',
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
                    new_projects_per_timestep=NEW_PROJECTS,
                    number_of_processors=NUMBER_OF_PROCESSORS,
                    departmental_workload=DEPARTMENTAL_WORKLOAD,
                    budget_functionality_flag=BUDGET_CONSTRAINT_FLAG,
                    training_on=TRAINING_FLAG,
                    target_training_load=TRAINING_LOAD,
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
                #abm.run_model(50)
                #abm.trainer.training_boost()
                #abm.run_model(50)
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
