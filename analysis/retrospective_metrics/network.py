"""
SuperScript retrospective metric: Network of successful collaborations.
===========

This script reconstructs the network of historical successful collaborations at each timestep,
 using the output pickle files. So each edge weight gives the number of times that pair of workers
 have successfully worked on a project together up to that timestep.

Note: this is necessary at v1.1 as the gpickle network output is too disk-space intensive to save the network at every
timestep for a large batch of simulations.
"""

import pandas as pd
import pickle
import itertools
import numpy as np
import networkx as nx
from itertools import combinations
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


def update_graph(worker_data, project_data, t, p, G):

    status = get_project_status(p, project_data)

    if status:
        p_worker = get_workers_for_project(p, t - 1, worker_data)

        pairs = list(combinations(p_worker, 2))
        for pair in pairs:

            if (pair[0], pair[1]) not in G.edges():
                G.add_edge(pair[0], pair[1], weight=1)
            else:
                G[pair[0]][pair[1]]['weight'] += 1

    return G


def calculate_network(worker_data, project_data, directory_path,
                      rep, save_net=True, plot_net=False):

    reserve = []  # for instances where it appears that project finishes at timestep t, but it is logged at t+1
    G = nx.Graph()

    for t in range(1, 101):

        workers_present_at_t = worker_data.loc[t, :].index

        G.add_nodes_from(workers_present_at_t)
        removed_workers = [n for n in list(G.nodes) if n not in workers_present_at_t]
        G.remove_nodes_from(removed_workers)

        roi_worker_dict = {
            w: 0
            for w in workers_present_at_t
        }
        project_count_dict = {
            w: 0
            for w in workers_present_at_t
        }
        completed = completed_projects(t, worker_data)

        if len(reserve) > 0:
            for p in reserve:

                try:
                    G = update_graph(worker_data, project_data, t, p, G)

                except:
                    print("Can find project %d on second attempt." % p)

        if completed is not None and len(completed) > 0:

            reserve = []
            for p in completed:

                try:
                    G = update_graph(worker_data, project_data, t, p, G)

                except:
                    print("Can find project %d on first attempt." % p)
                    reserve.append(p)

        if save_net:
            file_path = directory_path + '/network_rep_%d_timestep_%d.adjlist' % (rep, t)
            nx.write_multiline_adjlist(G, file_path)
        if plot_net:
            nx.draw(G)
            plt.show()

    return G


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

                    roi_list = calculate_network(agents, projects)

                    with open(this_path + '/' + optimiser + '/roi_rep_%d.pickle' % r, 'wb') as out_file:
                        pickle.dump(roi_list, out_file)

                except:
                    print("Could not produce ROI for rep %d of : " % r, this_path + '/' + optimiser)


if __name__ == "__main__":

    #run_roi_for_all_simulations()

    replicate = 0

    agents_f = '../../simulation_io/skill_decay_0995_project_per_step_5_240621_v1.0/Random/agents_vars_rep_%d.pickle' % replicate
    projects_f = '../../simulation_io/skill_decay_0995_project_per_step_5_240621_v1.0/Random/projects_table_rep_%d.pickle' % replicate
    # # agents_f = '../../simulation_io/project_per_step_5_230521_v1.0/Random/agents_vars_rep_%d.pickle' % replicate
    # # projects_f = '../../simulation_io/project_per_step_5_230521_v1.0/Random/projects_table_rep_%d.pickle' % replicate
    #
    agents = load_data(agents_f)
    projects = load_data(projects_f)

    # print(agents.tail())
    # print(projects.tail())
    # print(projects.columns)
    # print(len(projects))
    # print(agents.columns)
    # print(agents.loc[~agents.contributes.isna()].tail())
    #
    G = calculate_network(agents, projects, None, replicate, save_net=False, plot_net=False)
    nx.draw(G)
    plt.show()
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
