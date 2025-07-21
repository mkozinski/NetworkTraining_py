import os
from time import time
import torch
import numpy as np
from NetworkTraining_py.metrics import get_metrics
from NetworkTraining_py.writer_text import WriterText


def log_dict_from_metrics(metrics,param_list):
    log_dict={param:metrics[i] for i,param in enumerate(param_list)}
    assert set(param_list)==set(log_dict.keys())
    return log_dict

class LoggerMetrics:
  param_list=["accuracy","recall","dice","cldice","betti","betti_topo","l","epoch_time"]

  def __init__(self,save_dir,name,preproc=lambda o,t: (o,t),
               saveBest=False,epochsSkipSaving=0,writer=None):
    self.writer=writer
    self.log_dir=save_dir
    self.name   =name
    self.preproc=preproc
    self.bestBetti=1000000
    self.saveBest=saveBest
    self.epoch=0
    self.epochsSkipSaving=epochsSkipSaving

    if self.writer is None:
        self.writer=WriterText(self.log_dir,"log_"+name+"Metrics.txt",self.param_list)

    self.metrics = []

    self.previous_time=time()

  def add(self,img,output,target,l,net=None,optim=None):
    o=output.unsqueeze(0).cuda()
    e=torch.exp(o)
    p=e/(1+e)

    lbimg=(target[0][0]*255).cpu()
    primg=(p[0][0][0]*255).cpu() 
    maskimg=(target[1][0]*255).cpu()

    pr = (255*(primg<150)).detach().numpy().astype(np.uint16)
    lb = lbimg.detach().numpy().astype(np.uint16)

    img_pred = (pr > 250)
    img_true = (lb != 0)
    img_mask = maskimg.detach().numpy() 
    img_pred = np.logical_and(img_pred, img_mask)

    accuracy, recall, dice, betti, betti_topo, cldice = get_metrics(img_pred, img_true, img_mask)    

    self.metrics.append([accuracy, recall, dice, cldice, betti, betti_topo, l])

  def logEpoch(self,net=None,optim=None,scheduler=None):
    self.epoch+=1

    current_time=time()
    epoch_time=current_time-self.previous_time

    mean_metrics = np.nanmean(self.metrics, 0).tolist()
    mean_metrics.append(epoch_time)
    
    self.metrics.clear()
    
    log_dict=log_dict_from_metrics(mean_metrics,self.param_list)

    self.writer.write(log_dict)

    b = mean_metrics[4]

    if self.epoch>self.epochsSkipSaving and self.saveBest and b < self.bestBetti:
      self.bestBetti = b
      if net:
        torch.save({'state_dict': net.state_dict()},
                    os.path.join(self.log_dir, 
                    'net_'+self.name+'_bestBetti.pth'))
      if optim:
        torch.save({'state_dict': optim.state_dict()},
                    os.path.join(self.log_dir, 
                    'optim_'+self.name+'_bestBetti.pth'))

    self.previous_time=time()

    return mean_metrics

