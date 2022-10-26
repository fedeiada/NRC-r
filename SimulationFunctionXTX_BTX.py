import numpy as np

if False:        # select the simulation function
    class SimulationFunctionXTX_BTX:
        """This class representing the simulation function which has been considered as \"Transpose(X) * A * X +Bi * X
          (Assumed A=2*In in)\". This class can be used to calculate the gradient and hessian of the mentioned function. """

        def __init__(self, b_sum):
            self.b_sum = b_sum

        @staticmethod
        def get_fn(x: numpy.array, b: numpy.array) -> numpy.array:
            """This method can be used to calculate the outcome of the function for each given Xi and Bi"""
            #XTAX +BiX
            f = numpy.matmul(numpy.matmul(numpy.transpose(x)), x) + numpy.matmul(numpy.transpose(b), x)
            return f

        @staticmethod
        def get_gradient_fn(x: numpy.array, b: numpy.array) -> numpy.array:
            """This method can be used to calculate the gradient for any given Xi."""
            # 2Ax+Bi
            A = 2 *numpy.eye(x.size)
            return numpy.matmul(A, x) + b

        @staticmethod
        def get_hessian_fn(x: numpy.array) -> numpy.array:
            """This method can be used to calculate the hessian for any given Xi."""
            #2A
            return 2 * numpy.eye(x.size)

        def get_optimum_x(self,number_of_nodes):
            return ((1/( number_of_nodes)) * self.b_sum)
else:
    class SimulationFunctionXTX_BTX:
        """This class representing the simulation function which has been considered as \J_0". This class can be used to calculate the gradient and hessian of the mentioned function. """

        def __init__(self, b_sum):
            self.b_sum = b_sum

        @staticmethod
        def get_fn(x: np.array, pc=0.25, sea_cond=np.array([12, 40, 20, 300]),B=4000, toh=10e-3, Rc=0.5, eta=1) -> np.array:
            """This method can be used to calculate the outcome of the function for each given Xi and Bi. X = [n, N, M, pc]"""
            m = x[0]
            N = x[1]
            M = x[2]
            td = sea_cond[3] / (1449.2 + 4.6 * sea_cond[0] - 0.055 * (sea_cond[0] ** 2) + 0.00029 * (sea_cond[0] ** 3)
                                + (1.34 - 0.01 * sea_cond[0]) * (sea_cond[1] - 35) + 0.16 * sea_cond[2])
            bterm = (1 / np.exp((0.1 - (m*(N/eta)*np.log2(M)*0.2*np.exp(-((3*100)/(2*(M-1)*Rc))))) * 100))
            f = (bterm + np.log(m * (1 + pc) * N + B * (toh + td))
                    - np.log(m) - np.log(Rc) - np.log(B) - np.log(N / eta)
                    - np.log(np.log2(M)))
            return f

        @staticmethod
        def get_gradient_fn(x: np.array, pc=0.25, sea_cond=np.array([12, 40, 20, 300]), toh=1e-3, B=4000, eta=1, Rc=0.5, pb=0.1) -> np.array:
            """This method can be used to calculate the gradient for any given Xi."""
            m = x[0]; N = x[1]; M = x[2];
            p = 1 + pc
            td = sea_cond[3] / (1449.2 + 4.6 * sea_cond[0] - 0.055 * (sea_cond[0] ** 2) + 0.00029 * (sea_cond[0] ** 3)
                                +(1.34 - 0.01 * sea_cond[0]) * (sea_cond[1] - 35) + 0.16 * sea_cond[2])
            L = (toh + td) * B
            dJdm = -1 / m + ((N * p) / (L + N * m * p)) + \
                   (20 * N * np.exp((20 * N * m * np.exp(-300 / (Rc * (2 * M - 2))) * np.log(M)) / (
                               eta * np.log(2)) - 100 * pb) * np.exp(-300 / (Rc * (2 * M - 2))) * np.log(M)) / (
                               eta * np.log(2))
            dJdN = -1 / N + ((m * p) / (L + N * m * p)) + \
                   (20 * m * np.exp((20 * N * m * np.exp(-300 / (Rc * (2 * M - 2))) * np.log(M)) / (
                               eta * np.log(2)) - 100 * pb) * np.exp(-300 / (Rc * (2 * M - 2))) * np.log(M)) / (
                               eta * np.log(2))
            dJdM = -1 / (M * np.log(M)) + \
                   np.exp(
                       (20 * N * m * np.exp(-300 / (Rc * (2 * M - 2))) * np.log(M)) / (eta * np.log(2)) - 100 * pb) * (
                           (20 * N * m * np.exp(-300 / (Rc * (2 * M - 2)))) / (M * eta * np.log(2)) + (
                           12000 * N * m * np.exp(-300 / (Rc * (2 * M - 2))) * np.log(M)) / (
                                   Rc * eta * np.log(2) * (2 * M - 2) ** 2))

            return np.array([dJdm, dJdN, dJdM])

        @staticmethod
        def get_hessian_fn(x: np.array, pc=0.25,*, sea_cond=np.array([12, 40, 20, 300]), toh=1e-3, B=4000, Rc=0.5, eta=1, pb=1) -> np.array:
            """This method can be used to calculate the hessian for any given Xi."""
            m = x[0]
            N = x[1]
            M = x[2]
            td = sea_cond[3] / (1449.2 + 4.6 * sea_cond[0] - 0.055 * (sea_cond[0] ** 2) + 0.00029 * (sea_cond[0] ** 3)
                                +(1.34 - 0.01 * sea_cond[0]) * (sea_cond[1] - 35) + 0.16 * sea_cond[2])
            L = (toh + td) * B
            p = 1 + pc
            d2Jdm2 = -((N ** 2) * (p ** 2)) / ((L + N * m * p) ** 2) + 1 / (m ** 2) + \
                     (400 * (N ** 2) * np.exp((20 * N * m * np.exp(-300 / (Rc * 2 * (M - 1))) * np.log(M)) / (
                                 eta * np.log(2)) - 100 * pb) * np.exp(-600 / (Rc * 2 * (M - 1))) * (
                                  np.log(M) ** 2)) / ((eta ** 2) * (np.log(2) ** 2))
            d2JdmdN = -((N * m) * (p ** 2)) / (L + N * m * p) ** 2 + p / (L + N * m * p) + \
                      (20 * np.exp((20 * N * m * np.exp(-300 / (Rc * (2 * M - 2))) * np.log(M)) / (
                                  eta * np.log(2)) - 100 * pb) * np.exp(-300 / (Rc * (2 * M - 2))) * np.log(M)) / (
                                  eta * np.log(2)) + \
                      (400 * N * m * np.exp((20 * N * m * np.exp(-300 / (Rc * 2 * (M - 1))) * np.log(M)) / (
                                  eta * np.log(2)) - 100 * pb) * np.exp(-600 / (Rc * 2 * (M - 1))) * (
                                   np.log(M) ** 2)) / ((eta ** 2) * (np.log(2) ** 2))
            d2JdmdM = (20 * N * np.exp(
                (20 * N * m * np.exp(-300 / (Rc * (2 * M - 2))) * np.log(M)) / (eta * np.log(2)) - 100 * pb) * np.exp(
                -300 / (Rc * (2 * M - 2)))) / (M * eta * np.log(2)) + \
                      (20 * N * np.exp((20 * N * m * np.exp(-300 / (Rc * (2 * M - 2))) * np.log(M)) / (
                                  eta * np.log(2)) - 100 * pb) * np.exp(-300 / (Rc * (2 * M - 2))) * np.log(M) * (
                                   (20 * N * m * np.exp(-300 / (Rc * (2 * M - 2)))) / (M * eta * np.log(2)) +
                                   (12000 * N * m * np.exp(-300 / (Rc * (2 * M - 2))) * np.log(M)) / (
                                               Rc * eta * np.log(2) * (2 * M - 2) ** 2))) / (eta * np.log(2)) + (
                                  12000 * N * np.exp((20 * N * m * np.exp(-300 / (Rc * (2 * M - 2))) * np.log(M)) / (
                                      eta * np.log(2)) - 100 * pb) * np.exp(-300 / (Rc * (2 * M - 2))) * np.log(M)) / (
                                  Rc * eta * np.log(2) * (2 * M - 2) ** 2)
            d2JdN2 = -((m ** 2) * (p ** 2)) / ((L + N * m * p) ** 2) + (1 / (N ** 2)) + \
                     (400 * (m ** 2) * np.exp((20 * N * m * np.exp(-300 / (Rc * 2 * (M - 1))) * np.log(M)) / (
                                 eta * np.log(2)) - 100 * pb) * np.exp(-600 / (Rc * 2 * (M - 1))) * (
                                  np.log(M) ** 2)) / ((eta ** 2) * (np.log(2) ** 2))
            d2JdNdM = (20 * m * np.exp(
                (20 * N * m * np.exp(-300 / (Rc * (2 * M - 2))) * np.log(M)) / (eta * np.log(2)) - 100 * pb) * np.exp(
                -300 / (Rc * (2 * M - 2)))) / (M * eta * np.log(2)) + \
                      (20 * N * np.exp((20 * N * m * np.exp(-300 / (Rc * (2 * M - 2))) * np.log(M)) / (
                                  eta * np.log(2)) - 100 * pb) * np.exp(-300 / (Rc * (2 * M - 2))) * np.log(M) * (
                                   (20 * N * m * np.exp(-300 / (Rc * (2 * M - 2)))) / (M * eta * np.log(2)) +
                                   (12000 * N * m * np.exp(-300 / (Rc * (2 * M - 2))) * np.log(M)) / (
                                               Rc * eta * np.log(2) * (2 * M - 2) ** 2))) / (eta * np.log(2)) + (
                                  12000 * N * np.exp((20 * N * m * np.exp(-300 / (Rc * (2 * M - 2))) * np.log(M)) / (
                                      eta * np.log(2)) - 100 * pb) * np.exp(-300 / (Rc * (2 * M - 2))) * np.log(M)) / (
                                  Rc * eta * np.log(2) * (2 * M - 2) ** 2)
            d2JdM2 = 1 / ((M ** 2) * np.log(M)) + 1 / ((M) ** 2 * np.log(M) ** 2) + \
                     np.exp((20 * N * m * np.exp(-300 / (Rc * (2 * M - 2))) * np.log(M)) / (
                                 eta * np.log(2)) - 100 * pb) * (
                                 (20 * N * m * np.exp(-300 / (Rc * (2 * M - 2)))) / (M * eta * np.log(2)) + (
                                     12000 * N * m * np.exp(-300 / (Rc * (2 * M - 2))) * np.log(M)) / (
                                             Rc * eta * np.log(2) * (2 * M - 2) ** 2)) ** 2 \
                     - np.exp(
                (20 * N * m * np.exp(-300 / (Rc * (2 * M - 2))) * np.log(M)) / (eta * np.log(2)) - 100 * pb) * (
                                 (20 * N * m * np.exp(-300 / (Rc * (2 * M - 2)))) / (M ** 2 * eta * np.log(2)) - (
                                     24000 * N * m * np.exp(-300 / (Rc * (2 * M - 2)))) / (
                                             M * Rc * eta * np.log(2) * (2 * M - 2) ** 2) + (
                                             48000 * N * m * np.exp(-300 / (Rc * (2 * M - 2))) * np.log(M)) / (
                                             Rc * eta * np.log(2) * (2 * M - 2) ** 3) - (
                                             7200000 * N * m * np.exp(-300 / (Rc * (2 * M - 2))) * np.log(M)) / (
                                             Rc ** 2 * eta * np.log(2) * (2 * M - 2) ** 4))
            return np.array([[d2Jdm2, d2JdmdN, 0],
                               [d2JdmdN, d2JdN2,  0],
                               [0,         0, d2JdM2]])

        def get_optimum_x(self, number_of_nodes):
            return ((1 / (number_of_nodes)) * self.b_sum)
    '''
    class SimulationFunctionXTX_BTX:
        """This class representing the simulation function which has been considered as \J_0". This class can be used to calculate the gradient and hessian of the mentioned function. """

        def __init__(self, b_sum):
            self.b_sum = b_sum

        @staticmethod
        def get_fn(x: numpy.array, pc=0.25, sea_cond=numpy.array([12, 40, 20, 300]),B=4000, toh=10e-3, Rc=0.5, eta=1) -> numpy.array:
            """This method can be used to calculate the outcome of the function for each given Xi and Bi. X = [n, N, M, pc]"""
            m = x[0]
            N = x[1]
            M = x[2]
            td = sea_cond[3] / (1449.2 + 4.6 * sea_cond[0] - 0.055 * (sea_cond[0] ** 2) + 0.00029 * (sea_cond[0] ** 3)
                                + (1.34 - 0.01 * sea_cond[0]) * (sea_cond[1] - 35) + 0.16 * sea_cond[2])
            f = numpy.log(m*(1+pc)*N + B*toh+td) - numpy.log(m) - numpy.log(Rc) - numpy.log(B) - numpy.log(N/eta)\
                - numpy.log(numpy.log2(M))
            return f

        @staticmethod
        def get_gradient_fn(x: numpy.array, pc=0.25, sea_cond=numpy.array([12, 40, 20, 300]), toh=1e-3, B=4000, eta=1, mu=0.2) -> numpy.array:
            """This method can be used to calculate the gradient for any given Xi."""
            m = x[0]; N = x[1]; M = x[2];
            td = sea_cond[3] / (1449.2 + 4.6 * sea_cond[0] - 0.055 * (sea_cond[0] ** 2) + 0.00029 * (sea_cond[0] ** 3)
                                +(1.34 - 0.01 * sea_cond[0]) * (sea_cond[1] - 35) + 0.16 * sea_cond[2])
            djdm = -1/m + (N*(1 + pc)) / (m*( 1 + pc)*N + B * (toh+td))
                   #+(N*numpy.exp(-300/(M - 1))*numpy.log(M))/(5*eta*mu*numpy.log(2)*((N*m*numpy.exp(-300/(M - 1))*numpy.log(M))/(5*eta*numpy.log(2)) - 1/10))
            djdN = -1 / N + m*(1 + pc) / (m*( 1 + pc)*N + B * (toh+td))
                   #+(m*numpy.exp(-300/(M - 1))*numpy.log(M))/(5*eta*mu*numpy.log(2)*((N*m*numpy.exp(-300/(M - 1))*numpy.log(M))/(5*eta*numpy.log(2)) - 1/10))
            djdM = -1 / (M * numpy.log(M))
                   #+ ((N*m*numpy.exp(-300/(M - 1)))/(5*M*eta*numpy.log(2)) + (60*N*m*numpy.exp(-300/(M - 1))*numpy.log(M))/(eta*numpy.log(2)*(M - 1)**2))/(mu*((N*m*numpy.exp(-300/(M - 1))*numpy.log(M))/(5*eta*numpy.log(2)) - 1/10))
            return numpy.array([djdm, djdN, djdM])

        @staticmethod
        def get_hessian_fn(x: numpy.array, pc=0.25,*, sea_cond=numpy.array([12, 40, 20, 300]), toh=1e-3, B=4000) -> numpy.array:
            """This method can be used to calculate the hessian for any given Xi."""
            m = x[0]
            N = x[1]
            M = x[2]
            td = sea_cond[3] / (1449.2 + 4.6 * sea_cond[0] - 0.055 * (sea_cond[0] ** 2) + 0.00029 * (sea_cond[0] ** 3)
                                +(1.34 - 0.01 * sea_cond[0]) * (sea_cond[1] - 35) + 0.16 * sea_cond[2])
            L = (toh + td) * B
            p = 1 + pc
            d2Jdm2 = -((N ** 2) * (p ** 2)) / (L + N * m * p) ** 2 + 1 / m ** 2
            d2JdN2 = -((m ** 2) * (p ** 2)) / (L + N * m * p) ** 2 + 1 / N ** 2
            d2JdmdN = -((N * m) * (p ** 2)) / (L + N * m * p) ** 2 + p / (L + N * m * p)
            d2JdM2 = 1 / ((M ** 2) * numpy.log(M)) + 1 / ((M) ** 2 * numpy.log(M) ** 2)
            return numpy.array([[d2Jdm2, d2JdmdN, 0],
                               [d2JdmdN, d2JdN2,  0],
                               [0,         0, d2JdM2]])

        def get_optimum_x(self, number_of_nodes):
             return ((1 / (number_of_nodes)) * self.b_sum)
        '''
