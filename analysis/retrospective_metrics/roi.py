"""
SuperScript retrospective metric: ROI.
===========

This script calculates the return on investment metric for a given simulation, using the output pickle files.

Note: this is necessary at v1.1 as ROI calculation is not included in the main code. In a future version ROI tracking
will be included as standard.
"""

import pandas as pd
import pickle
import numpy as np

HARD_SKILLS = ['A', 'B', 'C', 'D', 'E']


def load_data(filepath):
    try:
        with open(filepath, 'rb') as infile:
            data = pickle.load(infile)
    except:
        print("Could not find file: ", filepath)
        data = None

    return data


def get_projects_for_worker(worker_id, timestep, worker_data):

    if worker_data.loc[timestep, worker_id].contributes is not None:
        contributions = [worker_data.loc[timestep, worker_id].contributes[skill] for skill in HARD_SKILLS]
        worker_projects = list(set([item for sublist in contributions for item in sublist]))
        return worker_projects

    else:
        return []


def get_running_projects(timestep, worker_data):

    workers_present = worker_data.loc[timestep, :].index
    running_projects = [get_projects_for_worker(w, timestep, worker_data) for w in workers_present]
    running_projects = list(set([item for sublist in running_projects for item in sublist]))
    return running_projects


def completed_projects(timestep, worker_data):

    try:
        previous_running = get_running_projects(timestep-1, worker_data)
        running = get_running_projects(timestep, worker_data)
        return [p for p in previous_running if p not in running]
    except KeyError:
        return None


if __name__ == "__main__":

    replicate = 0
    agents_f = '../../simulation_io/project_per_step_2_230521_v1.0/Basin_w_flex/agents_vars_rep_%d.pickle' % replicate
    projects_f = '../../simulation_io/project_per_step_2_230521_v1.0/Basin_w_flex/projects_table_rep_%d.pickle' % replicate

    agents = load_data(agents_f)
    projects = load_data(projects_f)

    # print(agents.head())
    # print(projects.head())
    # print(projects.columns)
    # print(len(projects))
    # print(agents.columns)
    #
    # print(get_projects_for_worker(4, 1, agents))
    # print(get_projects_for_worker(4, 4, agents))
    # print(get_projects_for_worker(4, 5, agents))

    for t in range(1, 101):
        #print(get_running_projects(t, agents))
        print(completed_projects(t, agents))
        #break



