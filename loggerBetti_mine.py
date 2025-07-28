import os
import torch
import numpy as np
from .writer_text import WriterText
from .myBettiMetric import MyBettiMetric

class LoggerBettiMine:

  param_list=["Betti0Mean", "Betti0Min","Betti1Mean", "Betti1Min", "B0MinThr", "B1MinThr" ]

  def __init__(self,logdir,name,
               start, stop,
               preproc=lambda o,t: (o,t),
               saveBest=False,epochsSkipSaving=0,
               writer=None):
    self.log_dir=logdir
    self.name=name
    self.writer=writer

    self.preproc=preproc

    self.bestBMean0=float("inf")
    self.bestBMin0 =float("inf")
    self.bestBMean1=float("inf")
    self.bestBMin1 =float("inf")
    self.saveBest=saveBest
    self.epoch=0
    self.epochsSkipSaving=epochsSkipSaving

    if self.writer is None:
      self.writer=WriterText(self.log_dir,"log_"+name+"BettiMine.txt",self.param_list)

    self.mbm=MyBettiMetric(start,stop)

  def add(self,img,output,target,l,net=None,optim=None):
    o,t=self.preproc(output,target)
    for oo,tt in zip(o,t):
      self.mbm.add(oo,tt)

  def logEpoch(self,net=None,optim=None,scheduler=None):
    self.epoch+=1
    
    be0mean,be1mean,be0min,be1min,be0minThr,be1minThr=\
      self.mbm.computeBettiErrors()
    self.mbm.reset()

    self.writer.write({"Betti0Mean":be0mean,
                       "Betti0Min" :be0min,
                       "Betti1Mean":be1mean,
                       "Betti1Min" :be1min,
                       "B0MinThr"  :be0minThr,
                       "B1MinThr"  :be1minThr})

    def saveBestParam(param,currentParam,paramname,net,optim):
      if self.epoch>self.epochsSkipSaving and self.saveBest:
        if param<currentParam: 
          if net:
            torch.save({'state_dict': net.state_dict()},
                        os.path.join(self.log_dir, 
                        'net_'  +self.name+'_best' + paramname+ '.pth'))
          if optim:
            torch.save({'state_dict': optim.state_dict()},
                        os.path.join(self.log_dir, 
                        'optim_'+self.name+'_best' + paramname+ '.pth'))
          return param
        else:
          return currentParam

    self.bestBMean0=saveBestParam(be0mean,self.bestBMean0,"BEMean0",net,optim)
    self.bestBMean1=saveBestParam(be1mean,self.bestBMean1,"BEMean1",net,optim)
    self.bestBMin0 =saveBestParam(be0min, self.bestBMin0, "BEMin0" ,net,optim)
    self.bestBMin1 =saveBestParam(be1min, self.bestBMin1, "BEMin1" ,net,optim)


