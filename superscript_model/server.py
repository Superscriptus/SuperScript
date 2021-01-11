from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.UserParam import UserSettableParameter
from mesa.visualization.modules import (ChartModule,
                                        NetworkModule,
                                        TextElement)

from .model import SuperScriptModel
from .config import DEPARTMENT_COUNT
from .utilities import Random

r = lambda: Random.randint(0, 255)
colours = {di: '#%02X%02X%02X' % (r(), r(), r())
           for di in range(DEPARTMENT_COUNT)}


def network_portrayal(G):
    # The model ensures there is always 1 agent per node

    def node_color(agent):
        return colours.get(
            agent.department.dept_id, "#808080"
        )

    def edge_color(agent1, agent2):
        # if State.RESISTANT in (agent1.state, agent2.state):
        #    return "#000000"
        return "#e8e8e8"

    def edge_width(agent1, agent2):
        # if State.RESISTANT in (agent1.state, agent2.state):
        #    return 3
        return G[agent1.worker_id][agent2.worker_id]['weight']

    def get_agents(source, target):
        return G.nodes[source]["agent"][0], G.nodes[target]["agent"][0]

    portrayal = dict()
    portrayal["nodes"] = [
        {
            "size": 6,
            "color": node_color(agents[0]),
            "tooltip": "id: {}<br>state: {}".format(
                agents[0].worker_id, agents[0].department.dept_id
            ),
        }
        for (_, agents) in G.nodes.data("agent")
    ]

    portrayal["edges"] = [
        {
            "source": source,
            "target": target,
            "color": edge_color(*get_agents(source, target)),
            "width": edge_width(*get_agents(source, target)),
        }
        for (source, target) in G.edges
    ]

    return portrayal


class ConfigElement(TextElement):

    def __init__(self, parameter_name,
                 display_name):
        self.parameter_name = parameter_name
        self.display_name = display_name
        pass

    def render(self, model):

        switch = {True: 'on',
                  False: 'off'}
        return (self.display_name + ": "
                + switch[getattr(model, self.parameter_name)])


model_params = {
    "worker_count": UserSettableParameter(
        "slider",
        "Number of workers",
        100,
        10,
        1000,
        10,
        description="Choose how many workers to include in the model",
    ),
    "project_length": UserSettableParameter(
        "slider",
        "Project length",
        5,
        1,
        10,
        1,
        description="Choose length of each project",
    ),
    "new_projects_per_timestep": UserSettableParameter(
        "slider",
        "Projects per step",
        20,
        1,
        50,
        1,
        description="Choose number of projects created on each step",
    ),
    "target_training_load": UserSettableParameter(
        "slider",
        "Target training load",
        10,
        1,
        100,
        5,
        description="Choose target training load",
    ),
    "training_commences": UserSettableParameter(
        "slider",
        "Training commences on step",
        10,
        0,
        10,
        1,
        description="Training commences on step",
    ),
    "budget_functionality_flag": UserSettableParameter(
        "checkbox",
        "Budget constraint on/off",
        value=True,
        description="Turn budget constraint on or off",
    ),
    "peer_assessment_success_mean": UserSettableParameter(
        "slider",
        "Peer assessment success mean",
        1.05,
        1,
        2,
        0.05
    ),
    "peer_assessment_success_stdev": UserSettableParameter(
        "slider",
        "Peer assessment success stdev",
        0.2,
        0,
        1,
        0.1
    ),
    "peer_assessment_fail_mean": UserSettableParameter(
        "slider",
        "Peer assessment fail mean",
        0.95,
        0,
        1,
        0.05
    ),
    "peer_assessment_fail_stdev": UserSettableParameter(
        "slider",
        "Peer assessment fail stdev",
        0.2,
        0,
        1,
        0.1
    ),
    "peer_assessment_weight": UserSettableParameter(
        "slider",
        "Peer assessment weight",
        0.25,
        0,
        1,
        0.05
    ),
    "update_skill_by_risk_flag": UserSettableParameter(
        "checkbox",
        "Update skill by risk on/off",
        value=True,
        description="Update skill by risk on/off",
    )
}

network = NetworkModule(network_portrayal, 500, 500, library="d3")

training_element = ConfigElement(
    'training_on', 'Training'
)
budget_element = ConfigElement(
    'budget_functionality_flag', 'Budget constraint'
)

chart1 = ChartModule([{"Label": "ActiveProjects",
                       "Color": "Black"}],
                     data_collector_name='datacollector')

chart2 = ChartModule([{"Label": "SuccessfulProjects",
                       "Color": "Green"},
                      {"Label": "FailedProjects",
                       "Color": "Red"},
                      {"Label": "NullProjects",
                       "Color": "Orange"}
                      ],
                     data_collector_name='datacollector')

chart3 = ChartModule([{"Label": "AverageSuccessProbability",
                       "Color": "Blue"}],
                     data_collector_name='datacollector')

chart4 = ChartModule([{"Label": "WorkersOnProjects",
                       "Color": "Green"},
                      {"Label": "WorkersWithoutProjects",
                       "Color": "Red"},
                      {"Label": "WorkersOnTraining",
                       "Color": "Orange"}
                      ],
                     data_collector_name='datacollector')

chart5 = ChartModule([{"Label": "ProjectLoad",
                       "Color": "Green"},
                      {"Label": "TrainingLoad",
                       "Color": "Orange"},
                      {"Label": "DeptLoad",
                       "Color": "Blue"},
                      {"Label": "Slack",
                       "Color": "Red"}
                      ],
                     data_collector_name='datacollector')

chart6 = ChartModule([{"Label": "AverageWorkerOvr",
                       "Color": "Blue"},
                      {"Label": "AverageTeamOvr",
                       "Color": "Green"},
                      ],
                     data_collector_name='datacollector')

chart7 = ChartModule([{"Label": "AverageTeamSize",
                       "Color": "Blue"}],
                     data_collector_name='datacollector')

chart8 = ChartModule([{"Label": "RecentSuccessRate",
                       "Color": "Blue"}],
                     data_collector_name='datacollector')

chart9 = ChartModule([{"Label": "WorkerTurnover",
                       "Color": "Blue"}],
                     data_collector_name='datacollector')

chart10 = ChartModule([{"Label": "ProjectsPerWorker",
                       "Color": "Blue"}],
                      data_collector_name='datacollector')


server = ModularServer(
    # SuperScriptModel, [network, chart1, chart2, chart3, chart4, chart5, chart6, chart7, chart8], "SuperScript Model", model_params
    SuperScriptModel, [training_element, budget_element, chart1, chart2, chart3, chart4, chart5, chart6, chart7, chart8, chart9, chart10],
    "SuperScript Model", model_params
)
server.port = 8521
