from copy import deepcopy

class WriterComposite:

    def __init__(self,writers):
        self.writers=writers

    def write(log_dict):
        for writer in self.writers:
            writer.write(deepcopy(log_dict))
