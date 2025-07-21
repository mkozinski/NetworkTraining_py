import os
import torch
import numpy as np
from .ccq import ccqHistograms,ccqFromHistograms

from .writer_text import WriterText

class LoggerCCQ:

  param_list=["completeness","correctness","quality"]

  def __init__(self,logdir,name,
               val_range, radius, pr_is_dm=True, gt_is_dm=True,
               nBins=10000,preproc=lambda o,t: (o,t),saveBest=False,
               writer=None):
    self.log_dir=logdir
    self.name=name
    self.writer=writer

    self.preproc=preproc
    self.nBins=nBins
    self.val_range=val_range
    self.radius=radius
    self.pr_is_dm=pr_is_dm
    self.gt_is_dm=gt_is_dm
    self.hist_pr_gt         =torch.zeros(self.nBins,dtype=torch.long)
    self.hist_pr_close_to_gt=torch.zeros(self.nBins,dtype=torch.long)
    self.hist_pr_far_from_gt=torch.zeros(self.nBins,dtype=torch.long)
    self.bestQ=0
    self.saveBest=saveBest

    if self.writer is None:
      self.writer=WriterText(self.log_dir,"log_"+name+"CCQ.txt",self.param_list)

  def add(self,img,output,target,l,net=None,optim=None):
    pr,gt=self.preproc(output,target)
    hg,hc,hf=ccqHistograms(pr, gt,
                           self.val_range,
                           self.radius,
                           pr_is_dm=self.pr_is_dm,
                           gt_is_dm=self.gt_is_dm,
                           nbins=self.nBins)
    self.hist_pr_gt=self.hist_pr_gt.to(hg)+hg
    self.hist_pr_close_to_gt=self.hist_pr_close_to_gt.to(hc)+hc
    self.hist_pr_far_from_gt=self.hist_pr_far_from_gt.to(hf)+hf

  def logEpoch(self,net=None,optim=None,scheduler=None):
    cpl,cor,qal=ccqFromHistograms(self.hist_pr_gt,
                                  self.hist_pr_close_to_gt,
                                  self.hist_pr_far_from_gt)
    quality,qalind=qal.max(dim=0)
    completeness,correctness=cpl[qalind],cor[qalind]

    self.writer.write(
        {"completeness":completeness,
         "correctbess" :correctness,
         "quality"     :quality})

    if self.saveBest and quality > self.bestQ:
      self.bestQ=quality
      if net:
        torch.save({'state_dict': net.state_dict()},
                    os.path.join(self.log_dir, 
                    'net_'+self.name+'_bestQ.pth'))
      if optim:
        torch.save({'state_dict': optim.state_dict()},
                    os.path.join(self.log_dir, 
                    'optim_'+self.name+'_bestQ.pth'))

    self.hist_pr_gt.zero_()
    self.hist_pr_close_to_gt.zero_()
    self.hist_pr_far_from_gt.zero_()
