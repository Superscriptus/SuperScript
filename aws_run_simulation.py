import time
import pickle
import sys, os

## Usage: python aws_run_simulation.py ./simulation_io 100 100 2

if __name__ == "__main__":

    MODEL_DIR = os.path.realpath(os.path.dirname('superscript_model'))
    sys.path.append(os.path.normpath(MODEL_DIR))
    from superscript_model import model

    if len(sys.argv) == 5:
        SAVE_DIR = sys.argv[1]
        STEPS = int(sys.argv[2])
        WORKER_COUNT = int(sys.argv[3])
        NEW_PROJECTS = int(sys.argv[4])
    else:
        STEPS = 100
        SAVE_DIR = MODEL_DIR
        WORKER_COUNT = 100
        NEW_PROJECTS = 2

    abm = model.SuperScriptModel(
        worker_count=WORKER_COUNT,
        new_projects_per_timestep=NEW_PROJECTS,
        io_dir=SAVE_DIR
    )

    start_time = time.time()
    abm.run_model(STEPS)
    elapsed_time = time.time() - start_time
    print("Took %.2f seconds to run %d steps." % (elapsed_time, STEPS))

    tracked = abm.datacollector.get_model_vars_dataframe()
    print(tracked)

    with open(SAVE_DIR
              + '/tracked_model_vars_wc_%d_np_%d_ts_%d.pickle'
              % (WORKER_COUNT, NEW_PROJECTS, STEPS), 'wb') as ofile:

        pickle.dump(tracked, ofile)

    projects = abm.datacollector.get_table_dataframe("Projects")
    print(projects)

    with open(SAVE_DIR
              + '/tracked_projects_wc_%d_np_%d_ts_%d.pickle'
              % (WORKER_COUNT, NEW_PROJECTS, STEPS), 'wb') as ofile:

        pickle.dump(projects, ofile)

    agents = abm.datacollector.get_agent_vars_dataframe()
    print(agents)

    with open(SAVE_DIR
              + '/tracked_agents_wc_%d_np_%d_ts_%d.pickle'
              % (WORKER_COUNT, NEW_PROJECTS, STEPS), 'wb') as ofile:

        pickle.dump(agents, ofile)

