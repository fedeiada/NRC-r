import random
import threading
import time
import numpy as np
from Message import *
import sys
sys.setrecursionlimit(2000)
'''
Modification:
1) pass c
2) added flag for loss communication
3) added sigma and rho
4) modified the update
5) inversion on z
'''

class Node(threading.Thread):
    """Representing the node and its actions and attributes. This class inherits the Thread class, so each instance
    of this class would be executed in a separate thread. This is important in order of improving the simulation
    speed and also simulating asynchronous communications """

    def __init__(self, node_id: int, x0: np.array, epsilon: float, c: float ,all_nodes_message_buffers: [],
                 minimum_accepted_divergence: float,
                 adjacency_vector: np.array,
                 function_constants: np.array, simulation_function_xtx_btx):

        super(Node, self).__init__()
        self.lock = threading.Lock()   
        self.node_id = node_id
        self.epsilon = epsilon
        #self.c = c
        self.xi = np.array(x0)  # + np.random.uniform(-1, 1, x0.size)

        '''
        ###### MODIFY INITIAL CONDITION #####
        self.xi[0] = self.xi[0] + np.random.randint(-1, 2)
        self.xi[1] = self.xi[1] + np.random.randint(-25, 25)
        self.xi[2] = self.xi[2] + np.random.randint(-1, 3)
        '''

        if self.node_id==0:
            self.sea_condition = np.array([16, 30, 20, 300])
        else:
            self.sea_condition = np.array([11, 40, 40, 300])


        self.all_calculated_xis = []
        self.evolution_costfun = []     # update value of cost_function
        self.function_constants = function_constants
        self.simulation_function_xtx_btx = simulation_function_xtx_btx
        self.ff = simulation_function_xtx_btx.get_fn(self.xi, sea_cond=self.sea_condition)

        self.all_nodes_message_buffers = all_nodes_message_buffers
        self.number_of_neighbors = np.sum(adjacency_vector)
        self.minimum_accepted_divergence = minimum_accepted_divergence
        self.adjacency_vector = adjacency_vector

        self.yi = np.zeros(x0.size)
        self.gi = np.zeros(x0.size)
        self.gi_old = np.zeros(x0.size)
        self.zi = np.eye(x0.size)
        self.cI = c * self.zi   # variable to be check in order to ensure robustness and large basin of attraction
        self.hi = np.eye(x0.size)
        self.hi_old = np.eye(x0.size)
        self.sigma_yi = np.zeros(x0.size)  # counter of the total mass-y sent
        self.sigma_zi = np.zeros((x0.size, x0.size))    # counter of the total mass-z sent, matrix x0.sizeXx0.size

        self.rho_yj = np.zeros((adjacency_vector.size, x0.size))   # counter of the total mass-y received from j, matrix dimXxo.size
        self.rho_yj_old = np.zeros((adjacency_vector.size, x0.size))
        self.rho_zj = np.zeros((adjacency_vector.size, x0.size, x0.size))  # counter of the total mass-z received from j
        self.rho_zj_old = np.zeros((adjacency_vector.size, x0.size, x0.size))

        self.msg_rel = True  # flag to simulate a packet loss

        self.is_ready_to_receive = False
        self.is_ready_to_update = False
        self.is_ready_to_transmit = True

    def run(self) -> None:
        """ This method is the start point for the executing the thread for each node. By calling the start() method
        on a node, this method will be called. """
        self.transmit_data()

    def transmit_data(self):
        """This method update the yi and zi in each iteration and create a message including the new updated yi and
        zi. Finally the new message will be broadcast to all neighbors of this node. """
        print(f"Node ID: {self.node_id} -  transmitting data started!\n")
        if self.is_ready_to_transmit:
            self.is_ready_to_transmit = False
            self.msg_rel = True  # always reset the simulation flag, starting point is reliable

            self.yi = (1 / (self.number_of_neighbors + 1)) * self.yi
            self.zi = self.zi / (self.number_of_neighbors + 1)

            self.sigma_yi = self.sigma_yi + self.yi
            self.sigma_zi = self.sigma_zi + self.zi


            if random.randrange(0, 9, 1) == 11:  # 1/10 chance to loss message
                self.msg_rel = False
                print("message LOST")

            '''
            if random.randrange(0, 9, 1)%2 == 0:  # 1/2 chance to loss message
                self.msg_rel = False
                print("message LOST")
            '''

            message = Message(self.node_id, self.sigma_yi, self.sigma_zi, self.msg_rel)
            self.broadcast(message)
            self.is_ready_to_receive = True
            print(f"Node ID: {self.node_id} -  transmitting data ended!\n")
            self.receive_data()
        return

    def broadcast(self, message):
        """This method will broadcast the passed message to all neighbors of this node."""
        print(f"Node ID: {self.node_id} -  broadcasting started!\n")
        with self.lock:
            i = 0
            while i < len(self.all_nodes_message_buffers):
                if self.adjacency_vector[i] == 1:
                    self.all_nodes_message_buffers[i].put(message)
                i += 1
        time.sleep(0.05)
        print(f"Node ID: {self.node_id} -  broadcasting ended!\n")
        return

    def receive_data(self):
        """This method will handle the reception of data from neighbors. Using the yi and zi from the messages,
        the yi and zi would be updated by this method."""
        if self.is_ready_to_receive:
            print(f"Node ID: {self.node_id} -  Receiving data started!\n")
            with self.lock:
                message = self.all_nodes_message_buffers[self.node_id].get()
                time.sleep(0.05)
            self.is_ready_to_receive = False
            self.j = message.node_id

            # update old virtual mass received
            self.rho_zj_old[self.j] = self.rho_zj[self.j]
            self.rho_yj_old[self.j] = self.rho_yj[self.j]

            #self.msg_rel_j = message.msg_rel         # check if received msg is reliable or not
            if message.msg_rel:       # update virtual mass of neighbour
                self.rho_yj[self.j] = message.sigma_yi
                self.rho_zj[self.j] = message.sigma_zi

            # update estimate
            self.yi = self.yi + self.rho_yj[self.j] - self.rho_yj_old[self.j]
            self.zi = self.zi + self.rho_zj[self.j] - self.rho_zj_old[self.j]

            self.is_ready_to_update = True
            print(f"Node ID: {self.node_id} -  Receiving data ended!\n")
            self.update_estimation()
        return

    def update_estimation(self):
        """This method will calculate the next xi and also update the hi and gi by using the new xi. """
        print(f"Node ID: {self.node_id} -  Updating data started!\n")
        if self.is_ready_to_update:
            self.is_ready_to_update = False
            self.all_calculated_xis.append(self.xi)
            self.evolution_costfun.append(self.ff)

            # check condition on z !!! modified !!!
            if np.less_equal(self.zi, self.cI).all():
                self.zi = self.cI

            self.xi = (1 - self.epsilon) * self.xi + np.matmul((self.epsilon * np.linalg.inv(self.zi)),
                                                               np.transpose(self.yi))
            self.ff = self.simulation_function_xtx_btx.get_fn(self.xi, sea_cond=self.sea_condition)    # new value of cost function
            '''####### BOUNDARY PROJECTION #########
             2<=m<=20, tau*B/pc<=N<=B/(k*v), 4<=M<=64
            if not self.xi[0] >= 2 and self.xi[0] <= 20:    # check m
                if self.xi[0] <= 2:
                    self.xi[0] = 2
                else:
                    self.xi[0] = 20
            if not self.xi[1] >= 400 and self.xi[1] <= 2000:    # check N
                if self.xi[1] <= 400:
                    self.xi[1] = 400
                else:
                    self.xi[1] = 2000
            if not self.xi[2] >= 2 and self.xi[2] <= 32:    # check M
                if self.xi[2] <= 2:
                    self.xi[2] = 2
                else:
                    self.xi[2] = 32
             '''

            self.gi_old = self.gi
            self.hi_old = self.hi

            self.hi = self.simulation_function_xtx_btx.get_hessian_fn(self.xi, sea_cond=self.sea_condition)
            self.gi = np.subtract(np.matmul(self.hi, self.xi), self.simulation_function_xtx_btx.get_gradient_fn(self.xi, sea_cond=self.sea_condition))

            self.yi = self.yi + self.gi - self.gi_old
            self.zi = self.zi + self.hi - self.hi_old

            self.is_ready_to_transmit = True
            print(f"Node ID: {self.node_id} -  Updating data ended!\n")
            self.transmit_data()
        return

    def has_result_founded(self):
        """This method will check and verify if the calculated xi in this node has sufficiently converged. If for the
        last calculated xi, the difference between xi and x(i-1) is less than minimum accepted divergence that has
        been provided by the user, then it would be considered that calculated xi has enough convergence to its
        target value. """
        self.is_convergence_sufficient = False
        if len(self.all_calculated_xis) > 10:
            self.is_convergence_sufficient = True
            for i in range(10):
                for j in range(self.all_calculated_xis[-(i + 1)].size):
                    if abs(abs(self.all_calculated_xis[-(i + 1)][j]) - abs(
                            self.all_calculated_xis[-(i + 1) - 1][j])) > self.minimum_accepted_divergence:
                        self.is_convergence_sufficient = False
        return self.is_convergence_sufficient
