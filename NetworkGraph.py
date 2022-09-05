import queue
from typing import List

import matplotlib.pyplot as plt
import networkx as nx

import SimulationFunctionXTX_BTX
import SimulationSpecification
from Node import *


# from Node import Node
'''
Modified:
1) pass c in node creation

'''

class NetworkGraph:
    """This class handling all the required action for generating a network of nodes. """
    nodes: List[Node]

    def __init__(self, simulation_specification: SimulationSpecification):
        self.simulation_specification = simulation_specification
        self.all_nodes_message_buffers = []
        self.nodes = []
        self.graph_matrix = []
        self.generate_network_graph()
        self.generate_message_buffers()
        self.instantiate_all_nodes()

    def generate_network_graph(self):
        """Generating a random graph for network. This method by 60% chance will consider each two nodes as neighbour.
        If generated graph contains a stand alon node without any neighbour, a new graph will be generated. """
        is_graph_needed = True
        while is_graph_needed:
            self.network_graph = nx.gnp_random_graph(self.simulation_specification.number_of_nodes, 0.6)
            self.graph_matrix = nx.to_numpy_array(self.network_graph)
            is_graph_needed = False
            for i in range(len(self.graph_matrix)):
                sum = 0
                for j in range(len(self.graph_matrix[i])):
                    sum += self.graph_matrix[i][j]
                if sum == 0:
                    is_graph_needed = True
                    break

    def generate_message_buffers(self):
        """Generates a list of queues to be used as a message buffer for each node. """
        for i in range(self.simulation_specification.number_of_nodes):
            message_queue = queue.Queue(0)
            self.all_nodes_message_buffers.append(message_queue)

    def instantiate_all_nodes(self):
        """Create the required list of nodes for simulation."""

        # Generating random B for each node: self.b represents the B in our function (XTX + BX). Since each node can
        # have separate function, values of the b vector fpr each node will be generated randomly but those values will
        # not differ a lot between different nodes.
        self.b = (1e-5 * (np.random.rand(self.simulation_specification.number_of_nodes,
                                         self.simulation_specification.x0.size) - np.full(
            (self.simulation_specification.number_of_nodes, self.simulation_specification.x0.size), 0.5)))

        print(f"Generate b for all nodes:\n {self.b}")
        self.b_sum = np.sum(self.b, 0)

        # Instantiating the function class using the sum of all B vectors.
        simulationFunctionXTX_BTX = SimulationFunctionXTX_BTX.SimulationFunctionXTX_BTX(self.b_sum)
        self.optimum_point = simulationFunctionXTX_BTX.get_optimum_x(self.simulation_specification.number_of_nodes)
        print(f"opt point \n {self.optimum_point}")

        # Instantiating all nodes
        for i in range(self.simulation_specification.number_of_nodes):
            node = Node(i,
                        self.simulation_specification.x0,
                        self.simulation_specification.epsilon,
                        self.simulation_specification.c,
                        self.all_nodes_message_buffers,
                        self.simulation_specification.min_accepted_divergence,
                        self.graph_matrix[i],
                        self.b[i],
                        simulationFunctionXTX_BTX)
            self.nodes.append(node)

    # Drawing the topology of generated network
    def draw_graph(self):
        nx.draw(self.network_graph)
        plt.show()
