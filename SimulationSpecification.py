import numpy as np
import Node

'''
Modification
1) specify c 
'''

class SimulationSpecification:
    """Data-holder for simulation specification. This class handles all the required user interactions in order to
    collect the simulation specification. """
    def __init__(self):
        self.number_of_nodes: list[Node] = 2
        self.x0 = np.array([0])
        self.epsilon = 0.0
        self.min_accepted_divergence = 0.0
        self.c = 0.0


    def set_number_of_nodes(self):
        """Number of nodes for simulation should be specified."""

        while self.number_of_nodes == 0:
            try:
                self.number_of_nodes = int(input("Please insert number of nodes:"))
            except:
                print("Your input should not be \"0\" or \"non integer\"! Please try again! ")
        return

    def set_x0(self):
        """Start point (X0) should be specified as a list. Ex. [1,1,2]"""

        self.x0 = np.array([12, 512, 4])

        # Please comment the hard coded value and uncomment the following code if inserting the start point from
        # console is preferred
        ##########################################################################################################
        #has_x0_set = False
        # while True:
        #     try:
        #         self.x0 = np.array(input("Please insert the start point (X0) as a list (Ex. [1,1,2]):"))
        #         break
        #     except:
        #         print("Your input should be a list\"! Please try again! ")
        #       has_x0_set = True
        return

    def set_epsilon(self):
        """Proper step-size should be specified. Ex. 0.01 """

        self.epsilon = 0.5

        # Please comment the hard coded value and uncomment the following code if inserting the start point from
        # console is preferred
        ##########################################################################################################
        # while self.epsilon == 0.0 or self.epsilon < 0 or self.epsilon > 1:
        #     try:
        #         self.epsilon = float(
        #             input("Please insert the epsilon (step size). Epsilon should be positive value but smaller than 1:"))
        #     except:
        #         print("Your input should be a positive float but smaller than 1. (Ex. 0.01): ")
        # return

    def set_c(self):
        """select c to guarantee a big enough basin of attracion """

        self.c = 0.4

        # Please comment the hard coded value and uncomment the following code if inserting the start point from
        # console is preferred
        ##########################################################################################################
        # while self.c == 0.0 or self.c < 0 or self.c > 1:
        #     try:
        #         self.c = float(
        #             input("Please insert the c. Epsilon should be positive value but smaller than 1:"))
        #     except:
        #         print("Your input should be a positive float but smaller than 1. (Ex. 0.01): ")
        # return

    def set_min_accepted_divergence(self):
        """Minimum accepted divergence should be specified. This value will be used to evaluate if calculated xi has
        converged sufficiently or not. """

        self.min_accepted_divergence = 0.001

        # Please comment the hard coded value and uncomment the following code if inserting the start point from
        # console is preferred
        ##########################################################################################################
        #  while True:
        #      try:
        #          self.min_accepted_divergence = float(input("Please insert minimum accepted divergence (Ex. 0.01):"))
        #          break
        #      except:
        #          print("Your input not correct!\"! Please try again! ")
        return
