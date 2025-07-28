import os
import torch
import numpy as np
from .writer_text import WriterText
from .bettiMetricsTopo import BettiErrorMetric


class LoggerBettiTM:

  param_list=["BettiError0", "BettiError1", ]

  def __init__(self,logdir,name,preproc=lambda o,t: (o,t),
               saveBest=False,epochsSkipSaving=0,
               writer=None):
    self.log_dir=logdir
    self.name=name
    self.writer=writer

    self.preproc=preproc

    self.bestBE0=float("inf")
    self.bestBE1=float("inf")
    self.saveBest=saveBest
    self.epoch=0
    self.epochsSkipSaving=epochsSkipSaving

    if self.writer is None:
      self.writer=WriterText(self.log_dir,"log_"+name+"BettiTM.txt",self.param_list)

    self.bm=BettiErrorMetric()

  def add(self,img,output,target,l,net=None,optim=None):
    o,t=self.preproc(output,target)
    self.bm(o,t)

  def logEpoch(self,net=None,optim=None,scheduler=None):
    self.epoch+=1
    
    bettierrors=torch.mean(self.bm.aggregate().float(),dim=0)
    berr0=bettierrors[0].item()
    berr1=bettierrors[1].item()

    self.bm=BettiErrorMetric()

    self.writer.write({"BettiError0":berr0,
                       "BettiError1":berr1,})

    if self.epoch>self.epochsSkipSaving and self.saveBest:
      if berr0<self.bestBE0: 
        self.bestBE0=berr0
        if net:
          torch.save({'state_dict': net.state_dict()},
                      os.path.join(self.log_dir, 
                      'net_'+self.name+'_bestBE0.pth'))
        if optim:
          torch.save({'state_dict': optim.state_dict()},
                      os.path.join(self.log_dir, 
                      'optim_'+self.name+'_bestBE0.pth'))

      if berr1<self.bestBE1: 
        self.bestBE1=berr1
        if net:
          torch.save({'state_dict': net.state_dict()},
                      os.path.join(self.log_dir, 
                      'net_'+self.name+'_bestBE1.pth'))
        if optim:
          torch.save({'state_dict': optim.state_dict()},
                      os.path.join(self.log_dir, 
                      'optim_'+self.name+'_bestBE1.pth'))

