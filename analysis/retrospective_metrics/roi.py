"""
SuperScript retrospective metric: ROI.
===========

This script calculates the return on investment metric for a given simulation, using the output pickle files.

Note: this is necessary at v1.1 as ROI calculation is not included in the main code. In a future version ROI tracking
will be included as standard.
"""

import pandas as pd
import pickle
import itertools
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

    roi_total_dict = dict()

    for t in range(1, 101):
        print(t)

        reserve = []
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

            completed.extend(reserve)
            reserve = []
            for p in completed:
                try:
                    status = get_return(get_project_status(p, project_data))
                    p_worker = get_workers_for_project(p, t - 1, worker_data)

                    for w in p_worker:
                        roi_total_dict[t] += status

                except:
                    reserve.append(p)

                # print(get_workers_for_project(p, t-1, agents))

    return roi_total_dict


def calculate_instantaneous_roi(worker_data, project_data):

    roi = []
    reserve = []  # for instances where it appears that project finishes at timestep t, but it is logged at t+1

    for t in range(1, 101):

        workers_present_at_t = worker_data.loc[t, :].index
        roi_worker_dict = {
            w: 0
            for w in workers_present_at_t
        }

        completed = completed_projects(t, worker_data)

        if len(reserve) > 0:
            for p in reserve:

                try:
                    status = get_return(get_project_status(p, project_data))
                    p_worker = get_workers_for_project(p, t - 1, worker_data)

                    for w in p_worker:
                        roi_worker_dict[w] += status

                except:
                    print("Can find project %d on second attempt." % p)

        if completed is not None and len(completed) > 0:

            completed.extend(reserve)
            reserve = []
            for p in completed:

                try:
                    status = get_return(get_project_status(p, project_data))
                    p_worker = get_workers_for_project(p, t - 1, worker_data)

                    for w in p_worker:
                        roi_worker_dict[w] += status

                except:
                    print("Can find project %d on first attempt." % p)
                    reserve.append(p)

        roi.append(np.mean(list(roi_worker_dict.values())))

    return roi


def run_roi_for_all_simulations(sim_path='../../simulation_io/streamlit/', replicate_count=1):

    PPS = [1, 2, 3, 5, 10]
    SD = [0.95, 0.99, 0.995]
    DW = [0.1, 0.3]
    TL = [0.1, 0.3, 0.0, 2.0]
    BF = [0, 1]

    combinations = list(itertools.product(PPS, SD, DW, TL, BF))

    for pi, parameters in enumerate(combinations):
        print(pi, parameters)

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

        this_path = sim_path + batch_name

        for optimiser in ['Basin', 'Basin_w_flex', 'Niter0', 'Random']:
            for r in range(replicate_count):
                agents_f = this_path + '/' + optimiser + '/agents_vars_rep_%d.pickle' % r
                projects_f = this_path + '/' + optimiser + '/projects_table_rep_%d.pickle' % r

                try:
                    agents = load_data(agents_f)
                    projects = load_data(projects_f)

                    roi_list = calculate_instantaneous_roi(agents, projects)

                    with open(this_path + '/' + optimiser + '/roi_rep_%d.pickle' % r, 'wb') as out_file:
                        pickle.dump(roi_list, out_file)

                except:
                    print("Could not produce ROI for rep %d of : " % r, this_path + optimiser)


if __name__ == "__main__":

    run_roi_for_all_simulations()
    # replicate = 0
    #
    # agents_f = '../../simulation_io/skill_decay_0995_project_per_step_5_240621_v1.0/Random/agents_vars_rep_%d.pickle' % replicate
    # projects_f = '../../simulation_io/skill_decay_0995_project_per_step_5_240621_v1.0/Random/projects_table_rep_%d.pickle' % replicate
    # # agents_f = '../../simulation_io/project_per_step_5_230521_v1.0/Random/agents_vars_rep_%d.pickle' % replicate
    # # projects_f = '../../simulation_io/project_per_step_5_230521_v1.0/Random/projects_table_rep_%d.pickle' % replicate
    #
    # agents = load_data(agents_f)
    # projects = load_data(projects_f)
    #
    # #print(agents.head())
    # # print(projects.head())
    # #print(projects.columns)
    # # print(len(projects))
    # # print(agents.columns)
    # #
    # # print(get_projects_for_worker(4, 1, agents))
    # # print(get_projects_for_worker(4, 4, agents))
    # # print(get_projects_for_worker(4, 5, agents))
    #
    # # roi_tot_r = calculate_total_cummulative_roi(agents, projects)
    # roi_r = calculate_instantaneous_roi(agents, projects)
    #
    # agents_f = '../../simulation_io/skill_decay_0995_project_per_step_5_240621_v1.0/Basin_w_flex/agents_vars_rep_%d.pickle' % replicate
    # projects_f = '../../simulation_io/skill_decay_0995_project_per_step_5_240621_v1.0/Basin_w_flex/projects_table_rep_%d.pickle' % replicate
    # # agents_f = '../../simulation_io/project_per_step_5_230521_v1.0/Basin_w_flex/agents_vars_rep_%d.pickle' % replicate
    # # projects_f = '../../simulation_io/project_per_step_5_230521_v1.0/Basin_w_flex/projects_table_rep_%d.pickle' % replicate
    #
    # agents = load_data(agents_f)
    # projects = load_data(projects_f)
    #
    # # print(agents.head())
    # # print(projects.head())
    # # print(projects.columns)
    # # print(len(projects))
    # # print(agents.columns)
    # #
    # # print(get_projects_for_worker(4, 1, agents))
    # # print(get_projects_for_worker(4, 4, agents))
    # # print(get_projects_for_worker(4, 5, agents))
    #
    # # roi_tot_bwf = calculate_total_cummulative_roi(agents, projects)
    # roi_bwf = calculate_instantaneous_roi(agents, projects)
    #
    #
    # def movingaverage(interval, window_size):
    #     window = np.ones(int(window_size)) / float(window_size)
    #     return np.convolve(interval, window, 'same')
    #
    # # plt.plot(list(roi_tot_bwf.values()), label='Basin_w_flex')
    # # plt.plot(list(roi_tot_r.values()), label='Random')
    # plt.plot(roi_bwf, 'bo--', label='Basin_w_flex', linewidth=1)
    # plt.plot(roi_r, 'go--', label='Random', linewidth=1)
    #
    # x_av = movingaverage(roi_r, window_size=7)
    # plt.plot(x_av, c='g', linewidth=2)
    #
    # x_av = movingaverage(roi_bwf, window_size=7)
    # plt.plot(x_av, c='b', linewidth=2)
    #
    # plt.legend()
    # plt.title('Cumulative Total Return on Investment')
    # plt.xlabel('time')
    # plt.ylabel('cumulative ROI')
    # plt.show()
    #
    #
    #
