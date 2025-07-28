import os
import torch
import numpy as np
from .writer_text import WriterText
from medpy.metric import hd95


class LoggerHD95:

  param_list=["HD95",  ]

  def __init__(self,logdir,name,preproc=lambda o,t: (o,t),
               saveBest=False,epochsSkipSaving=0,
               writer=None):
    self.log_dir=logdir
    self.name=name
    self.writer=writer

    self.preproc=preproc

    self.bestHD95=float("inf")
    self.saveBest=saveBest
    self.epoch=0
    self.epochsSkipSaving=epochsSkipSaving

    if self.writer is None:
      self.writer=WriterText(self.log_dir,"log_"+name+"HD95.txt",self.param_list)

    self.hd95_vals=[]

  def add(self,img,output,target,l,net=None,optim=None):
    o,t=self.preproc(output,target)
    # process each image in the batch individually
    for oo,tt in zip(o,t):
        self.hd95_vals.append(hd95(oo,tt) if np.max(oo)>0 else float("inf"))

  def logEpoch(self,net=None,optim=None,scheduler=None):
    self.epoch+=1
    
    mean_hd95=torch.mean(torch.tensor(self.hd95_vals,dtype=float)).item()

    self.hd95_vals=[]

    self.writer.write({"HD95":mean_hd95,})

    if self.epoch>self.epochsSkipSaving and self.saveBest:
      if mean_hd95<self.bestHD95: 
        self.bestHD95=mean_hd95
        if net:
          torch.save({'state_dict': net.state_dict()},
                      os.path.join(self.log_dir, 
                      'net_'+self.name+'_bestHD95.pth'))
        if optim:
          torch.save({'state_dict': optim.state_dict()},
                      os.path.join(self.log_dir, 
                      'optim_'+self.name+'_bestHD95.pth'))

