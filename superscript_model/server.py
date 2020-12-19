from mesa.visualization.modules import NetworkModule
from mesa.visualization.ModularVisualization import ModularServer

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


network = NetworkModule(network_portrayal, 500, 500, library="d3")

server = ModularServer(
    SuperScriptModel, [network], "SuperScript Model"
)
server.port = 8521
