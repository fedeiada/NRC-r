import numpy

if True:        # select the simulation function
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
    class SimulationFunction:
        """This class representing the simulation function which has been considered as \J_0". This class can be used to calculate the gradient and hessian of the mentioned function. """

        def __init__(self, b_sum):
            self.b_sum = b_sum

        @staticmethod
        def get_fn(x: numpy.array, B, toh, Rc, Nx) -> numpy.array:
            """This method can be used to calculate the outcome of the function for each given Xi and Bi. X = [n, N, M, pc]"""
            m = x[0]; N = x[1]; M = x[2]; pc = x[3]
            f = numpy.log(m*(1+pc)*N + B*toh)- numpy.log(m) - numpy.log(Rc) - numpy.log(B) - numpy.log(N-Nx)- numpy.log(numpy.log2(M))
            return f

        @staticmethod
        def get_gradient_fn(x: numpy.array, B, toh, Nx) -> numpy.array:
            """This method can be used to calculate the gradient for any given Xi."""
            m = x[0];N = x[1];M = x[2];pc = x[3]
            djdm = 1/m - N*(1 + pc) / (m*( 1 + pc)*N + B * toh)
            djdN = 1 / (N - Nx) + N*(1 + pc) / (m*( 1 + pc) + B * toh)
            djdM = 1 / (M * numpy.log(M))
            djdpc = - (m * N) / ( m*(1+pc)*N + B*toh)
            return numpy.array([djdm, djdN, djdM, djdpc]).T

        @staticmethod
        def get_hessian_fn(x: numpy.array, B, toh, Nx) -> numpy.array:
            """This method can be used to calculate the hessian for any given Xi."""
            m = x[0];N = x[1];M = x[2];pc = x[3]
            d2Jdm2 = -1/m^2 + (m^2*(1+pc^2)^2)/((m*(1+pc)*N) + B*toh)^2
            d2JdmdN = -((1+pc)*(m*(1+pc)+B*toh)-(m*(1+pc)^2))/(m*(1+pc) + B*toh)^2
            d2Jdmdpc = -(N*(m*(1+pc)*N + B*toh)-m*(N^2)*(1+pc))/(m*(1+pc)*N + B*toh)^2
            d2JdN2 = - 1/(N - Nx)^2
            d2JdNdp = -(m*(m*(1+pc)*N + B*toh)-N*(m^2)*(1+pc))/(m*(1+pc)*N + B*toh)^2
            d2JdM2 = - (numpy.log(M)+1)/((M*numpy.log(M))^2)
            d2Jdp2 = (m^2*N^2)/((m*(1+pc)*N + B*toh)^2)
            return numpy.array([[d2Jdm2,  d2JdmdN, 0,   d2Jdmdpc],
                                [d2JdmdN, d2JdN2,  0,    d2JdNdp]
                                [0      ,     0, d2JdM2,    0   ],
                                [d2Jdmdpc, d2JdNdp, 0 ,   d2Jdp2]])

        def get_optimum_x(self, number_of_nodes):
            return ((1 / (number_of_nodes)) * self.b_sum)