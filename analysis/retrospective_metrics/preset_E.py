"""
SuperScript retrospective construction of Preset E from pre-simulated data (agents_vars.pickle)
===========

Preset E is the same as preset C, but is the 'Adaptive Organisation' version such that idle workers
are removed down to a Slack of 10%.

Prset C has:
PPS: 3
SD: 0.995
DL: 0.1
TL: 0.1
TF: 1
TB: 0
BF: 1

"""

import pandas as pd
import pickle
import itertools
import numpy as np
import matplotlib.pyplot as plt
from random import choice

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


def get_projects_units_for_worker(worker_id, timestep, worker_data):

    contributions = worker_data.loc[timestep, worker_id].contributes
    if contributions is not None:
        contributions = [contributions[skill] for skill in HARD_SKILLS]
        flat_contributions = [item for sublist in contributions for item in sublist]
        worker_projects = list(set(flat_contributions))
        units_worked = len(flat_contributions)
        return worker_projects, units_worked

    else:
        return [], None


def training_workers(timestep, worker_data):

    workers = worker_data.loc[timestep, :]
    workers = workers[workers['training_remaining'] > 0].index

    return list(set(workers))


def get_loads(timestep, worker_data, units_per_fte=10):

    workers_present = worker_data.loc[timestep, :].index
    training = set(training_workers(timestep, worker_data))
    train_load = len(training) / len(workers_present)

    project_load = (
            sum([
                get_projects_units_for_worker(wid, timestep, worker_data)[1]
                for wid in workers_present
                if get_projects_units_for_worker(wid, timestep, worker_data)[1] is not None
            ])
            / (len(workers_present) * units_per_fte)
    )

    dept_load = 0.1
    slack = 1 - project_load - train_load - dept_load

    return project_load, train_load, dept_load, slack


def compute_new_model_vars(worker_data, model_data):

    new_model_vars = pd.DataFrame()
    for col in [
        'ActiveProjects', 'RecentSuccessRate', 'SuccessfulProjects', 'FailedProjects',
        'NullProjects', 'AverageSuccessProbability', 'AverageTeamOvr', 'AverageTeamSize',
       ]:
        new_model_vars[col] = model_data[col]

    new_model_vars.index = model_data.index

    new_data = {}
    for new_col in [
        'WorkersOnProjects', 'WorkersWithoutProjects', 'WorkersOnTraining', 'AverageTeamSize',
        'AverageWorkerOvr', 'WorkerTurnover', 'ProjectLoad', 'TrainingLoad', 'DeptLoad', 'Slack', 'ProjectsPerWorker'
    ]:

        new_data[new_col] = []

    # REMOVE BASED ON SLACK NOT NUMBER OF WORKERS!!
    # can only remove inactive workers
    # down to a slack of 0.1
    # In cases where there are not enough inactive workers, slack will not get down to 0.1
    #
    for t in new_model_vars.index:
        w_tstep = t + 1  # conversion because worker state is saved at beginning of the next timestep
        workers_present = set(list(worker_data.loc[w_tstep, :].index))

        project_load, train_load, dept_load, slack = get_loads(w_tstep, worker_data)

        training = set(training_workers(w_tstep, worker_data))
        project_workers = set([w for w in workers_present if len(get_projects_for_worker(w, w_tstep, worker_data)) > 0])
        inactive = workers_present - training - project_workers

        data_slice = worker_data.copy()
        turnover_correction = 0

        while slack > 0.1 and len(inactive) > 0:
            to_remove = choice(list(inactive))

            if data_slice.loc[(w_tstep, to_remove), 'timesteps_inactive'] >= 5:
                turnover_correction += 1

            inactive.remove(to_remove)
            workers_present.remove(to_remove)

            data_slice = data_slice.loc[(w_tstep, workers_present), :]
            project_load, train_load, dept_load, slack = get_loads(w_tstep, data_slice)


        print("correction: ", turnover_correction)

        # compute metrics:
        w_count = 0
        all_projects_worked = []
        all_ovr = []
        turnover_count = 0
        for wid in workers_present:
            projects_worked, units_worked = get_projects_units_for_worker(wid, w_tstep, data_slice)

            all_projects_worked.append(len(projects_worked))
            all_ovr.append(data_slice.loc[(w_tstep, wid), 'ovr'])

            if len(projects_worked) > 0:
                w_count += 1

            if data_slice.loc[(w_tstep, wid), 'timesteps_inactive'] >= 5:
                turnover_count += 1

            project_load, train_load, dept_load, slack = get_loads(w_tstep, data_slice)
            # if units_worked is not None:
            #     print("UNITS: %d" % units_worked)

        new_data['WorkersOnProjects'].append(w_count)
        new_data['WorkersWithoutProjects'].append(len(workers_present) - w_count)
        new_data['ProjectsPerWorker'].append(np.mean(all_projects_worked))
        new_data['WorkersOnTraining'] = len(training_workers(w_tstep, data_slice))
        new_data['AverageWorkerOvr'].append(np.mean(all_ovr))
        new_data['ProjectLoad'] = project_load
        new_data['TrainingLoad'] = train_load
        new_data['DeptLoad'] = dept_load
        new_data['Slack'] = slack
        new_data['WorkerTurnover'] = turnover_count - turnover_correction

    return new_data


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


if __name__ == "__main__":

    #run_roi_for_all_simulations()

    replicate = 0
    #
    agents_f = '../../simulation_io/skill_decay_0995_project_per_step_5_240621_v1.0/Random/agents_vars_rep_%d.pickle' % replicate
    model_f = '../../simulation_io/skill_decay_0995_project_per_step_5_240621_v1.0/Random/model_vars_rep_%d.pickle' % replicate
    # projects_f = '../../simulation_io/skill_decay_0995_project_per_step_5_240621_v1.0/Random/projects_table_rep_%d.pickle' % replicate
    # agents_f = '../../simulation_io/project_per_step_5_230521_v1.0/Random/agents_vars_rep_%d.pickle' % replicate
    # projects_f = '../../simulation_io/project_per_step_5_230521_v1.0/Random/projects_table_rep_%d.pickle' % replicate

    agents = load_data(agents_f)
    model = load_data(model_f)
    # projects = load_data(projects_f)
    compute_new_model_vars(agents, model)
    # print(model.columns)
    # print(agents.head())
    # #print(projects.columns)
    # # print(len(projects))
    # print(agents.columns)
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
