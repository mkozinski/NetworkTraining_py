import wandb
import numpy as np
import torch as th

class WriterWandB:

    def __init__(self,entity,project,config):
        self.wandb_run=wandb.init(
            entity=entity,
            project=project,
            config=config
        )

    def write(self,log_dict):
        self.wandb_run.log(log_dict)
