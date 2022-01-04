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

But we run this for all SD and all TL also.

TODO: need to save project file and filtered agents file so that can run network reconstruction and ROI.
"""

import pandas as pd
import pickle
import itertools
from random import choice
import numpy as np
import os

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
        worker_projects = list(set([item for sublist in contributions for item in sublist]))
        worker_units = len([item for sublist in contributions for item in sublist])

        return worker_projects, worker_units

    else:
        return [], None


def training_workers(timestep, worker_data):

    workers = worker_data.loc[timestep, :]
    workers = workers[workers['training_remaining'] > 0].index

    return list(set(workers))


def get_loads(timestep, worker_data, dw=0.1, units_per_fte=10):

    workers_present_at_t = worker_data.loc[timestep, :].index
    train_load = len(training_workers(timestep, worker_data)) / len(workers_present_at_t)

    project_load = sum([
        get_projects_units_for_worker(w, timestep, worker_data)[1] for w in workers_present_at_t
        if get_projects_units_for_worker(w, timestep, worker_data)[1] is not None
    ]) / (len(workers_present_at_t) * units_per_fte)

    dept_load = min(dw, 1 - project_load - train_load)
    slack = 1 - dept_load - project_load - train_load

    return project_load, train_load, dept_load, slack


def get_new_model_vars(worker_data, model_vars, projects_data):

    new_agents = None
    new_model_vars = pd.DataFrame()
    keep_cols = [
        'ActiveProjects', 'RecentSuccessRate', 'SuccessfulProjects',
        'FailedProjects', 'NullProjects', 'AverageTeamSize',
        'AverageSuccessProbability', 'AverageTeamOvr'
    ]
    new_cols = [
        'WorkersOnProjects', 'ProjectsPerWorker',
        'WorkersWithoutProjects', 'WorkersOnTraining', 'AverageWorkerOvr',
        'WorkerTurnover', 'ProjectLoad', 'TrainingLoad', 'DeptLoad', 'Slack'
    ]
    for col in keep_cols:
        new_model_vars[col] = model_vars[col]
    new_model_vars.index = model_vars.index

    new_data = {col: [] for col in new_cols}

    for t in new_model_vars.index:

        # First we remove inactive workers to try to reduce slack to 10%:
        w_step = t + 1 #  conversion need because worker data is saved at the beginning of following timestep
        workers_present_at_t = worker_data.loc[w_step, :].index

        trainers = training_workers(w_step, worker_data)
        project_workers = [
            w for w in workers_present_at_t
            if len(get_projects_for_worker(w, w_step, worker_data)) > 0
        ]
        inactive = set(workers_present_at_t) - set(trainers) - set(project_workers)
        project_load, train_load, dept_load, slack = get_loads(w_step, worker_data)

        data_slice = worker_data.loc[(w_step, workers_present_at_t), :].copy()
        workers_present_at_t = set(worker_data.loc[w_step, :].index)

        turnover_correction = 0
        while slack > 0.1 and len(inactive) > 1:
            to_remove = choice(list(inactive))
            if int(data_slice.loc[(w_step, to_remove), 'timesteps_inactive']) == 5:
                turnover_correction += 1

            inactive.remove(to_remove)
            workers_present_at_t.remove(to_remove)
            data_slice = worker_data.loc[(w_step, workers_present_at_t), :].copy()
            project_load, train_load, dept_load, slack = get_loads(w_step, data_slice)

        # add to new agents data
        if new_agents is None:
            new_agents = data_slice
        else:
            new_agents = new_agents.append(data_slice)

        # Now we compute the new metrics:
        project_load, train_load, dept_load, slack = get_loads(w_step, data_slice)

        project_counts = []
        for w in workers_present_at_t:

            this_projects, this_units = get_projects_units_for_worker(w, w_step, data_slice)
            project_counts.append(len(this_projects))

        new_data['WorkersOnTraining'].append(len(training_workers(w_step, data_slice)))
        new_data['ProjectLoad'].append(project_load)
        new_data['TrainingLoad'].append(train_load)
        new_data['DeptLoad'].append(dept_load)
        new_data['Slack'].append(slack)
        new_data['AverageWorkerOvr'].append(np.mean(data_slice.ovr))
        new_data['ProjectsPerWorker'].append(np.mean(project_counts))

        workers_on_projects = sum([1 for count in project_counts if count > 0])
        new_data['WorkersOnProjects'].append(workers_on_projects)
        new_data['WorkersWithoutProjects'].append(
            len(workers_present_at_t) - workers_on_projects - len(training_workers(w_step, data_slice))
        )
        new_data['WorkerTurnover'].append(model_vars.loc[w_step - 1].WorkerTurnover - turnover_correction)

    for col in new_cols:
        new_model_vars[col] = new_data[col]

    return new_model_vars, new_agents, projects_data


def run_preset_E(sim_path='../../simulation_io/streamlit/', replicate_count=1):

    PPS = [3]
    SD = [0.95, 0.99, 0.995]
    DW = [0.1]
    TL = [0.1, 0.3, 0.0, 2.0]
    BF = [1]

    combinations = [
        [3, 0.95, 0.1, 0.1, 1],
        [3, 0.99, 0.1, 0.1, 1],
        [3, 0.995, 0.1, 0.1, 1],
        [3, 0.995, 0.1, 0.0, 1],
        [3, 0.995, 0.1, 0.3, 1],
        [3, 0.995, 0.1, 2.0, 1]
    ]
    # combinations = list(itertools.product(PPS, SD, DW, TL, BF))

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
        save_path = (
                sim_path
                + 'preset_E_sd_%.3f_tl_%.1f_tf_%d_tb_%d_251021_v1.1'
                % (skill_decay, training_load, training_flag, training_boost)
        )
        os.mkdir(save_path)

        for optimiser in ['Basin', 'Basin_w_flex', 'Random']:

            os.mkdir(save_path + '/' + optimiser)

            for r in range(replicate_count):
                agents_f = this_path + '/' + optimiser + '/agents_vars_rep_%d.pickle' % r
                model_f = this_path + '/' + optimiser + '/model_vars_rep_%d.pickle' % r
                projects_f = this_path + '/' + optimiser + '/projects_table_rep_%d.pickle' % r

                try:
                    agents = load_data(agents_f)
                    model = load_data(model_f)
                    projects = load_data(projects_f)

                    new_model_vars, new_agents, new_projects = get_new_model_vars(agents, model, projects)

                    with open(save_path + '/' + optimiser + '/model_vars_rep_%d.pickle' % r, 'wb') as out_file:
                        pickle.dump(new_model_vars, out_file)

                    with open(save_path + '/' + optimiser + '/agents_vars_rep_%d.pickle' % r, 'wb') as out_file:
                        pickle.dump(new_agents, out_file)

                    with open(save_path + '/' + optimiser + '/projects_table_rep_%d.pickle' % r, 'wb') as out_file:
                        pickle.dump(new_projects, out_file)

                except:
                    print(
                        "Could produce new model vars (preset E) for rep %d of : "
                        % r, this_path + '/' + optimiser
                    )


if __name__ == "__main__":

    run_preset_E()
    #run_preset_E(sim_path='../../simulation_io/streamlit_presets/')

    # replicate = 0
    #
    # agents_f = '../../simulation_io/skill_decay_0995_project_per_step_5_240621_v1.0/Random/agents_vars_rep_%d.pickle' % replicate
    # model_f = '../../simulation_io/skill_decay_0995_project_per_step_5_240621_v1.0/Random/model_vars_rep_%d.pickle' % replicate
    # projects_f = '../../simulation_io/skill_decay_0995_project_per_step_5_240621_v1.0/Random/projects_table_rep_%d.pickle' % replicate
    # agents_f = '../../simulation_io/project_per_step_5_230521_v1.0/Random/agents_vars_rep_%d.pickle' % replicate
    # projects_f = '../../simulation_io/project_per_step_5_230521_v1.0/Random/projects_table_rep_%d.pickle' % replicate

    # agents = load_data(agents_f)
    # model = load_data(model_f)
    # projects = load_data(projects_f)

    # print(model.head())
    # print(projects.head())
    # #print(projects.columns)
    # # print(len(projects))
    # print(agents.columns)
    # #
    # get_new_model_vars(agents, model, projects)

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
