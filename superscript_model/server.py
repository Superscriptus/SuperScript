from mesa.visualization.modules import NetworkModule
from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.UserParam import UserSettableParameter
from mesa.visualization.modules import ChartModule

from .model import SuperScriptModel
from .config import DEPARTMENT_COUNT
from .utilities import Random


r = lambda: Random.randint(0,255)
colours = {di: '#%02X%02X%02X' % (r(),r(),r())
           for di in range(DEPARTMENT_COUNT)}


def network_portrayal(G):
    # The model ensures there is always 1 agent per node

    def node_color(agent):
        return colours.get(
            agent.department.dept_id, "#808080"
        )

    def edge_color(agent1, agent2):
        #if State.RESISTANT in (agent1.state, agent2.state):
        #    return "#000000"
        return "#e8e8e8"

    def edge_width(agent1, agent2):
        #if State.RESISTANT in (agent1.state, agent2.state):
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
    "training_on": UserSettableParameter(
        "checkbox",
        "Training on/off",
        value=True,
        description="Turn training on or off",
    ),
    "budget_functionality_flag": UserSettableParameter(
        "checkbox",
        "Budget constraint on/off",
        value=True,
        description="Turn budget constraint on or off",
    )
}


network = NetworkModule(network_portrayal, 500, 500, library="d3")

chart1 = ChartModule([{"Label": "ActiveProjects",
                      "Color": "Black"}],
                    data_collector_name='datacollector')

chart2 = ChartModule([{"Label": "RecentSuccessRate",
                      "Color": "Blue"}],
                     data_collector_name='datacollector')

chart3 = ChartModule([{"Label": "SuccessfulProjects",
                      "Color": "Green"},
                      {"Label": "FailedProjects",
                       "Color": "Red"}],
                     data_collector_name='datacollector')

server = ModularServer(
    #SuperScriptModel, [network, chart1, chart2, chart3], "SuperScript Model", model_params
SuperScriptModel, [chart1, chart2, chart3], "SuperScript Model", model_params
)
server.port = 8521
