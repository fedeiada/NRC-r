import numpy


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


