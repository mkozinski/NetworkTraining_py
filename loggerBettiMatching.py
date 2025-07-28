import os
import torch
import numpy as np
from .writer_text import WriterText
from medpy.metric import hd95

import sys
sys.path.append("../Betti-Matching-3D/build")
from betti_matching import BettiMatching

def bettiMatchingError(pr,gt,n=0):

    bne=BettiMatching(pr,gt)
    bmatch=bne.compute_matching()
    result=bne.get_matching()
    error_matched = 2*sum((pr[*result.input1_matched_birth_coordinates[n].T]-gt[*result.input2_matched_birth_coordinates[n].T])**2+\
                          (pr[*result.input1_matched_death_coordinates[n].T]-gt[*result.input2_matched_death_coordinates[n].T])**2)
    error_unmatched_pr = sum((pr[*result.input1_unmatched_birth_coordinates[n].T]-pr[*result.input1_unmatched_death_coordinates[n].T])**2)
    error_unmatched_gt = sum((gt[*result.input2_unmatched_birth_coordinates[n].T]-gt[*result.input2_unmatched_death_coordinates[n].T])**2)
    error = error_matched + error_unmatched_pr + error_unmatched_gt

    return error

class LoggerBettiMatching:

  param_list=["BettiMatchingError0", "BettiMatchingError1" ]

  def __init__(self,logdir,name,preproc=lambda o,t: (o,t),
               saveBest=False,epochsSkipSaving=0,
               writer=None):
    self.log_dir=logdir
    self.name=name
    self.writer=writer

    self.preproc=preproc

    self.bestE0=float("inf")
    self.bestE1=float("inf")
    self.saveBest=saveBest
    self.epoch=0
    self.epochsSkipSaving=epochsSkipSaving

    if self.writer is None:
      self.writer=WriterText(self.log_dir,"log_"+name+"BettiMatching.txt",self.param_list)

    self.error0=0.0
    self.error1=0.0
    self.count=0

  def add(self,img,output,target,l,net=None,optim=None):
    o,t=self.preproc(output,target)
    # process each image in the batch individually
    for oo,tt in zip(o,t):
        self.error0+=bettiMatchingError(oo,tt,n=0)
        self.error1+=bettiMatchingError(oo,tt,n=1)
    self.count+=len(o)

  def logEpoch(self,net=None,optim=None,scheduler=None):
    self.epoch+=1
    
    e0=self.error0/self.count
    e1=self.error1/self.count

    self.error0=0.0
    self.error1=0.0
    self.count=0

    self.writer.write({"BettiMatchingError0":e0,
                       "BettiMatchingError1":e1})

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

    self.bestE0=saveBestParam(e0,self.bestE0,"BettiMatching0",net,optim)
    self.bestE1=saveBestParam(e1,self.bestE1,"BettiMatching1",net,optim)
