# Local simulation batch runner...

# Needs to use try catch on each run in case of crash (and log output).

import time
import pickle
import sys, os

# Name for this batch of simulations:
# (Note: should include date and code version number)
BATCH_NAME = 'test_run_260321_v0.0'
# These global configuration values override config.py:
STEPS = 100
WORKER_COUNT = 100
NEW_PROJECTS = 2
WORKER_STRATEGY = 'AllIN'
ORGANISATION_STRATEGY = 'Random'

if __name__ == "__main__":

    MODEL_DIR = './superscript_model'
    SAVE_DIR = './simulation_io/' + BATCH_NAME
    os.mkdir(SAVE_DIR)
    sys.path.append(os.path.normpath(MODEL_DIR))
    from superscript_model import model

    abm = model.SuperScriptModel(
        worker_count=WORKER_COUNT,
        new_projects_per_timestep=NEW_PROJECTS,
        worker_strategy=WORKER_STRATEGY,
        organisation_strategy=ORGANISATION_STRATEGY,
        io_dir=SAVE_DIR
    )

    start_time = time.time()
    abm.run_model(STEPS)
    elapsed_time = time.time() - start_time
    print(
        "Took %.2f seconds to run %d steps."
        % (elapsed_time, STEPS)
    )

    tracked = abm.datacollector.get_model_vars_dataframe()

    with open(SAVE_DIR
              + '/tracked_model_vars_wc_%d_np_%d_ts_%d.pickle'
              % (WORKER_COUNT, NEW_PROJECTS, STEPS), 'wb') as ofile:

        pickle.dump(tracked, ofile)

    projects = abm.datacollector.get_table_dataframe("Projects")

    with open(SAVE_DIR
              + '/tracked_projects_wc_%d_np_%d_ts_%d.pickle'
              % (WORKER_COUNT, NEW_PROJECTS, STEPS), 'wb') as ofile:

        pickle.dump(projects, ofile)

    agents = abm.datacollector.get_agent_vars_dataframe()

    with open(SAVE_DIR
              + '/tracked_agents_wc_%d_np_%d_ts_%d.pickle'
              % (WORKER_COUNT, NEW_PROJECTS, STEPS), 'wb') as ofile:

        pickle.dump(agents, ofile)

