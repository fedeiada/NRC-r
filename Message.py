'''
Modified:
1) changed the parameters passed
2) added msg_rel
'''

class Message:
    """Representing the message that should be passed between nodes. This class is a simple data-holder which
    contains the required data to be transferred between nodes. """

    def __init__(self, node_id, sigma_yi, sigma_zi, msg_rel):
        self.node_id = node_id
        self.sigma_yi = sigma_yi
        self.sigma_zi = sigma_zi
        self.msg_rel = msg_rel
