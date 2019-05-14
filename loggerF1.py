import os
import torch
import numpy as np
from .f1 import PRFromHistograms, F1FromPR

def reverse(t):
  idx = [i for i in range(t.size(0)-1, -1, -1)]
  idx = torch.LongTensor(idx)
  it  = t.index_select(0, idx)
  return it

class LoggerF1:

  def __init__(self,logdir,fname,transform,nBins=10000,saveBest=False):
    self.log_dir=logdir
    self.name=fname
    self.log_file=os.path.join(self.log_dir,"logF1"+self.name+".txt")
    text_file = open(self.log_file, "w")
    text_file.close()
    self.preproc=transform
    self.nBins=nBins
    self.hPos=torch.zeros(self.nBins)
    self.hNeg=torch.zeros(self.nBins)
    self.bestF1=0
    self.saveBest=saveBest

  def add(self,l,output,target):
    o,t=self.preproc(output.cpu().data,target.cpu().data)
    pos=o[t==1]
    neg=o[t==0]
    self.hPos+=pos.histc(self.nBins,0,1)
    self.hNeg+=neg.histc(self.nBins,0,1)

  def logEpoch(self,net):
    precision,recall=PRFromHistograms(self.hPos,self.hNeg)
    f1s=F1FromPR(precision,recall)
    f=f1s.max()
    text_file=open(self.log_file, "a")
    text_file.write('{}\n'.format(f))
    text_file.close()
    if self.saveBest and f > self.bestF1:
      self.bestF1=f
      torch.save({'state_dict': net.state_dict()},
                  os.path.join(self.log_dir, 
                  'net_'+self.name+'_bestF1.pth'))


    self.hPos.zero_()
    self.hNeg.zero_()
