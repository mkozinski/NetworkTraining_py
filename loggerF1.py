import os
import torch
import numpy as np
from .f1 import PRFromHistograms, F1FromPR
from .writer_text import WriterText

def reverse(t):
  idx = [i for i in range(t.size(0)-1, -1, -1)]
  idx = torch.LongTensor(idx)
  it  = t.index_select(0, idx)
  return it

class LoggerF1:

  param_list=["F1",]

  def __init__(self,logdir,name,preproc=lambda o,t: (o,t),
               nBins=10000,saveBest=False,epochsSkipSaving=0,
               writer=None):
    self.log_dir=logdir
    self.name=name
    self.writer=writer

    self.preproc=preproc
    self.nBins=nBins
    self.hPos=torch.zeros(self.nBins)
    self.hNeg=torch.zeros(self.nBins)
    self.bestF1=0
    self.saveBest=saveBest
    self.epoch=0
    self.epochsSkipSaving=epochsSkipSaving

    if self.writer is None:
      self.writer=WriterText(self.log_dir,"log_"+name+"F1.txt",self.param_list)

  def add(self,img,output,target,l,net=None,optim=None):
    o,t=self.preproc(output,target)
    pos=o[t==1]
    neg=o[t==0]
    self.hPos=self.hPos.to(pos)
    self.hNeg=self.hNeg.to(neg)
    self.hPos+=pos.histc(self.nBins,0,1)
    self.hNeg+=neg.histc(self.nBins,0,1)

  def logEpoch(self,net=None,optim=None,scheduler=None):
    self.epoch+=1
    precision,recall=PRFromHistograms(self.hPos,self.hNeg)
    f1s=F1FromPR(precision,recall)
    f=f1s.max()

    self.writer.write({"F1":f})

    if self.epoch>self.epochsSkipSaving and self.saveBest and f > self.bestF1:
      self.bestF1=f
      if net:
        torch.save({'state_dict': net.state_dict()},
                    os.path.join(self.log_dir, 
                    'net_'+self.name+'_bestF1.pth'))
      if optim:
        torch.save({'state_dict': optim.state_dict()},
                    os.path.join(self.log_dir, 
                    'optim_'+self.name+'_bestF1.pth'))

    self.hPos.zero_()
    self.hNeg.zero_()
