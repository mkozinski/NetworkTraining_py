import wandb
import numpy as np
import torch as th

class WriterWANDB:

    def __init__(self,entity,project,config):
        self.wandb_run=wandb.init(
            entity=entity,
            project=project,
            config=config
        )

    def write(log_dict):
        self.run.log(log_dict)
