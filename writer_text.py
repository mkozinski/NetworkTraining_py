from os import path
import wandb
import numpy as np
import torch as th

class WriterText:

    def __init__(self,log_dir,name,params):

        self.params=params 
        self.log_dir=log_dir
        self.log_file_name=path.join(self.log_dir,name)

        text_file = open(self.log_file_name, "w")
        for p in params:
            text_file.write('\t')
            text_file.write(p)
        text_file.write('\n')
        text_file.close()

    def write(self,log_dict):

        keys=set(log_dict.keys())
        params=set(self.params)
        assert keys<=params, "the log_dict keys differ from self.params"

        text_file = open(self.log_file_name, "a")
        for p in params:
            text_file.write(str(log_dict[p]))
            text_file.write('\t')
        text_file.write('\n')
        text_file.close()
