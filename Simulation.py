import SimulationSpecification
from NetworkGraph import NetworkGraph
from time import sleep
import sys
import matplotlib.pyplot as plt
import numpy as np

simulation_specification: SimulationSpecification
network_graph: NetworkGraph
sys.setrecursionlimit(2000)

class Simulation:
    """This class representing the simulation and actin as the main class for calling other classes and methods to
    create a simulation environment or calculate and draw required graphs. """

    def __init__(self):
        self.simulation_specification = SimulationSpecification.SimulationSpecification()

    def set_simulation_specification(self):
        """Sets all the required specifications of a simulation."""
        self.simulation_specification.set_number_of_nodes()
        self.simulation_specification.set_x0()
        self.simulation_specification.set_epsilon()
        self.simulation_specification.set_c()
        self.simulation_specification.set_min_accepted_divergence()
        return

    def generate_network_graph(self, simulation_specification: SimulationSpecification):
        """Generates an instance of network graph based on the simulation specification."""
        self.network_graph = NetworkGraph(simulation_specification)
        return

    def generate_convergence_plot(self):
        """Calculates the distance between generated xi of each iteration with expected outcome and finally plot this
        distance over simulation iteration. """
        distances = [[] for a in range(simulation.simulation_specification.number_of_nodes)]
        x0 = np.array(simulation.network_graph.optimum_point)
        fig, axs = plt.subplots(1, 2, figsize=(13, 8))
        for j in range(simulation.simulation_specification.number_of_nodes):
            for i in range(len(simulation.network_graph.nodes[j].all_calculated_xis)):
                xi = np.array(simulation.network_graph.nodes[j].all_calculated_xis[i])
                dst = np.sqrt(np.sum((x0 - xi) ** 2))
                distances[j].append(dst)
            axs[0].plot(distances[j], '-', label=f'node_{j}')
            axs[1].plot(simulation.network_graph.nodes[j].all_calculated_xis, label=f'node_{j}')
        axs[0].legend(loc='lower left', ncol=1)
        axs[0].set_title('Evolution of the error on xi')
        axs[0].set_xlabel('Iteration')
        axs[0].set_ylabel('Distance till optimum point')
        axs[1].legend(loc='upper right', ncol=1)
        axs[1].set_title('Evolution xi')
        axs[1].set_xlabel('Iteration')
        axs[1].set_ylabel('Point value evolution')
        axs[0].grid()
        axs[1].grid()
        plt.savefig('books_read.png')
        plt.show()

        for j in range(simulation.simulation_specification.number_of_nodes):
            print(simulation.network_graph.nodes[j].all_calculated_xis[-20:-10])


    def wait_until_result_founded(self) -> None:
        is_all_dif_accepted = False
        while True and not (is_all_dif_accepted):
            sleep(5)
            is_all_dif_accepted = True
            for i in range(len(simulation.network_graph.nodes)):
                if not (simulation.network_graph.nodes[i].has_result_founded()):
                    is_all_dif_accepted = False
                    break
        print("Result founded: ")


# Creating and executing the simulation based on simulation specification:
simulation: Simulation = Simulation()
simulation.set_simulation_specification()

# Generating the network graph with specified number of nodes based on the simulation specification:
simulation.generate_network_graph(simulation.simulation_specification)
simulation.network_graph.draw_graph()

# Starting the all nodes' threads in simulation graph:
for i in range(len(simulation.network_graph.nodes)):
    simulation.network_graph.nodes[i].daemon = True
    simulation.network_graph.nodes[i].start()


# Continue the simulation until all nodes reach a consensus estimation about the target xi:
simulation.wait_until_result_founded()

# Plot the distance between selected xi and target xi in different iterations:
simulation.generate_convergence_plot()

# Finish the simulation!
sys.exit()
