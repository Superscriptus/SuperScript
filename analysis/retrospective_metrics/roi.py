"""
SuperScript retrospective metric: ROI.
===========

This script calculates the return on investment metric for a given simulation, using the output pickle files.

On each timestep a worker's ROI is the weighted average of the return for their various activities, weighted by
the number of units that they spend on each activity. (Training is handled differently because training is blocking so
a worker on training commits all 10 of their units to training and the weighted mean is redundant).

The returns are:

    Project success (on completion): 50,
    Project fail (on completion): 10,
    Active project: 5,
    Training: 5,
    Departmental work: 10

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


def get_worker_contributions(worker_id, timestep, worker_data):

    contributions = worker_data.loc[timestep, worker_id].contributes
    if contributions is not None:
        contributions = [contributions[skill] for skill in HARD_SKILLS]
        worker_projects = [item for sublist in contributions for item in sublist]

        worker_contributions = {
            project: worker_projects.count(project)
            for project in list(set(worker_projects))
        }
    else:
        worker_contributions = {}

    return worker_contributions


def get_contributions_to_project(pid, timestep, worker_data):
    workers_present = worker_data.loc[timestep, :].index

    workers = {}
    for worker_id in workers_present:
        worker_contributions = get_worker_contributions(worker_id, timestep, worker_data)
        if pid in worker_contributions.keys():
            workers[worker_id] = worker_contributions[pid]

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


def training_workers(timestep, worker_data):

    workers = worker_data.loc[timestep, :]
    workers = workers[workers['training_remaining'] > 0].index

    return list(set(workers))


def get_project_status(pid, project_data):
    return project_data.loc[pid].success


def get_return(outcome, return_dict={True: 50, False: 10}):
    if outcome is None:
        return None
    else:
        return return_dict[outcome]


def add_projects_to_worker_roi_dict(
        project_list, project_data, worker_data, return_dict, t,
        roi_worker_dict, project_count_dict,
        get_status=get_project_status,
        add_to_reserve_list=False,
        error_message=""
):
    reserve_list = []
    if project_list is not None and len(project_list) > 0:

        for p in project_list:

            try:
                return_value = get_return(
                    get_status(p, project_data),
                    return_dict=return_dict
                )
                p_worker = get_contributions_to_project(p, t, worker_data)

                for w in p_worker:
                    roi_worker_dict[w] += return_value * p_worker[w]
                    project_count_dict[w] += p_worker[w]

            except:
                print(error_message % p)
                if add_to_reserve_list:
                    reserve_list.append(p)

    return roi_worker_dict, project_count_dict, reserve_list


def calculate_instantaneous_roi(worker_data, project_data, legacy=False, dept_wl=0.1, units_per_fte=10):

    if legacy:
        return_dict = {
            True: 25,
            False: 5
        }
    else:
        return_dict = {
            True: 50,
            False: 10,
            'active': 5,
            'train': 5,
            'dept': 10
        }

    roi = []
    reserve = []  # for instances where it appears that project finishes at timestep t, but it is logged at t+1

    for t in range(1, 101):

        workers_present_at_t = worker_data.loc[t, :].index
        roi_worker_dict = {
            w: 0
            for w in workers_present_at_t
        }
        unit_count_dict = {
            w: 0
            for w in workers_present_at_t
        }
        trainers = training_workers(t, worker_data)
        active_projects = get_running_projects(t, worker_data)
        completed = completed_projects(t, worker_data)

        roi_worker_dict, unit_count_dict, _ = add_projects_to_worker_roi_dict(
            reserve, project_data, worker_data, return_dict, t-1,
            roi_worker_dict, unit_count_dict,
            error_message="Can't find project %d on second attempt."
        )

        roi_worker_dict, unit_count_dict, _ = add_projects_to_worker_roi_dict(
            active_projects, project_data, worker_data, return_dict, t,
            roi_worker_dict, unit_count_dict,
            get_status=lambda x, y: 'active',
            error_message="Can't find active project %d."
        )

        roi_worker_dict, unit_count_dict, reserve = add_projects_to_worker_roi_dict(
            completed, project_data, worker_data, return_dict, t-1,
            roi_worker_dict, unit_count_dict,
            add_to_reserve_list=True,
            error_message="Can't find project %d on first attempt"
        )

        for tr in trainers:
            if roi_worker_dict[tr] != 0:
                print("Trainer already on project work!")  # This is due to timestep offset (see github issue #16).

            roi_worker_dict[tr] += return_dict['train']
            unit_count_dict[tr] += 10

        departmental_unit_count = int(len(workers_present_at_t) * dept_wl * units_per_fte)
        units_added = 0
        workers_checked = 0

        for worker in unit_count_dict.keys():
            workers_checked += 1

            while unit_count_dict[worker] < units_per_fte and units_added < departmental_unit_count:
                roi_worker_dict[worker] += return_dict['dept']
                unit_count_dict[worker] += 1
                units_added += 1

                if workers_checked == len(unit_count_dict):
                    print("Warning: Could not assign full departmental workload!")
                    break

        # departmental_workers = [w for w in workers_present_at_t if roi_worker_dict[w] == 0]
        # if len(departmental_workers) < departmental_worker_count:
        #     print("Not enough D workers!")
        #     departmental_workers.extend(list(set(workers_present_at_t) - set(departmental_workers)))
        # else:
        #     for d in range(departmental_worker_count):
        #         roi_worker_dict[departmental_workers[d]] += return_dict['dept']
        #         unit_count_dict[departmental_workers[d]] += 1
        # for d in range(departmental_worker_count):
        #     roi_worker_dict[departmental_workers[d]] += return_dict['dept']
        #     unit_count_dict[departmental_workers[d]] += 1

        for worker, units in unit_count_dict.items():
            if 0 < units < 10:
                unit_count_dict[worker] = 10

        roi_worker_dict = {
            w: roi_worker_dict[w] / unit_count_dict[w]
            if unit_count_dict[w] > 0
            else 0
            for w in roi_worker_dict.keys()
        }
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
                    print("Could not produce ROI for rep %d of : " % r, this_path + '/' + optimiser)

def run_roi_for_preset_e(sim_path='../../simulation_io/streamlit/', replicate_count=1):

    combinations = [
        [3, 0.95, 0.1, 0.1, 1],
        [3, 0.99, 0.1, 0.1, 1],
        [3, 0.995, 0.1, 0.1, 1],
        [3, 0.995, 0.1, 0.0, 1],
        [3, 0.995, 0.1, 0.3, 1],
        [3, 0.995, 0.1, 2.0, 1]
    ]

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
                'preset_E_sd_%.3f_tl_%.1f_tf_%d_tb_%d_251021_v1.1'
                % (skill_decay, training_load, training_flag, training_boost)
        )

        this_path = sim_path + batch_name

        for optimiser in ['Basin', 'Basin_w_flex', 'Random']:
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
                    print("Could not produce ROI for rep %d of : " % r, this_path + '/' + optimiser)


if __name__ == "__main__":

    # run_roi_for_all_simulations()
    # run_roi_for_preset_e()

    replicate = 0
    #
    agents_f = '../../simulation_io/skill_decay_0995_project_per_step_5_240621_v1.0/Random/agents_vars_rep_%d.pickle' % replicate
    projects_f = '../../simulation_io/skill_decay_0995_project_per_step_5_240621_v1.0/Random/projects_table_rep_%d.pickle' % replicate
    # agents_f = '../../simulation_io/project_per_step_5_230521_v1.0/Random/agents_vars_rep_%d.pickle' % replicate
    # projects_f = '../../simulation_io/project_per_step_5_230521_v1.0/Random/projects_table_rep_%d.pickle' % replicate

    agents = load_data(agents_f)
    projects = load_data(projects_f)

    print(calculate_instantaneous_roi(agents, projects))
    # print(agents.head())
    # print(projects.head())
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
    # print(roi_r)
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
