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
import matplotlib.pyplot as plt

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

    contributions = worker_data.loc[timestep, worker_id].contributes
    if contributions is not None:
        contributions = [contributions[skill] for skill in HARD_SKILLS]
        worker_projects = list(set([item for sublist in contributions for item in sublist]))
        return worker_projects

    else:
        return []


def get_workers_for_project(pid, timestep, worker_data):
    workers_present = worker_data.loc[timestep, :].index

    workers = []
    for worker_id in workers_present:
        worker_projects = get_projects_for_worker(worker_id, timestep, worker_data)
        if pid in worker_projects:
            workers.append(worker_id)

    return workers


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


def get_project_status(pid, project_data):
    return project_data.loc[pid].success


def get_return(outcome, return_dict={True: 25, False: 5}):
    if outcome is None:
        return None
    else:
        return return_dict[outcome]


def calculate_total_cummulative_roi(worker_data, project_data):

    roi_worker_dict = dict()
    roi_total_dict = dict()

    for t in range(1, 101):
        print(t)

        roi_worker_dict[t] = {}
        roi_total_dict[t] = 0
        if t > 1:
            roi_total_dict[t] += roi_total_dict[t - 1]

        completed = completed_projects(t, worker_data)
        if completed is not None:
            status = [get_return(get_project_status(p, project_data)) for p in completed_projects(t, worker_data)]
        else:
            status = None
        # print(completed_projects(t, agents), status)

        if completed is not None and len(completed) > 0:

            for p in completed:
                status = get_return(get_project_status(p, project_data))
                p_worker = get_workers_for_project(p, t - 1, worker_data)

                for w in p_worker:
                    roi_total_dict[t] += status
                # print(get_workers_for_project(p, t-1, agents))

    return roi_total_dict


if __name__ == "__main__":

    replicate = 0
    agents_f = '../../simulation_io/project_per_step_2_230521_v1.0/Random/agents_vars_rep_%d.pickle' % replicate
    projects_f = '../../simulation_io/project_per_step_2_230521_v1.0/Random/projects_table_rep_%d.pickle' % replicate

    agents = load_data(agents_f)
    projects = load_data(projects_f)

    #print(agents.head())
    # print(projects.head())
    #print(projects.columns)
    # print(len(projects))
    # print(agents.columns)
    #
    # print(get_projects_for_worker(4, 1, agents))
    # print(get_projects_for_worker(4, 4, agents))
    # print(get_projects_for_worker(4, 5, agents))

    roi_tot_r = calculate_total_cummulative_roi(agents, projects)

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

    roi_tot_bwf = calculate_total_cummulative_roi(agents, projects)

    plt.plot(list(roi_tot_bwf.values()), label='Basin_w_flex')
    plt.plot(list(roi_tot_r.values()), label='Random')
    plt.legend()
    plt.title('Cumulative Total Return on Investment')
    plt.xlabel('time')
    plt.ylabel('cumulative ROI')
    plt.show()
            # NEXT: save worker ROI in dict, and average (mean or median) ROI for present workers.
        #break



