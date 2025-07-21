import os
import torch
import numpy as np
from NetworkTraining_py.metrics import get_metrics

class LoggerMetrics:

  def __init__(self,logdir,fname,preproc=lambda o,t: (o,t),
               nBins=10000,saveBest=False,epochsSkipSaving=0):
    self.log_dir=logdir
    self.name=fname
    self.log_file=os.path.join(self.log_dir,"logMetrics_"+self.name+".txt")
    text_file = open(self.log_file, "w")
    text_file.write("accuracy,recall,dice,cldice,betti,betti_topo,loss\n")
    text_file.close()
    self.preproc=preproc
    self.nBins=nBins
    self.hPos=torch.zeros(self.nBins)
    self.hNeg=torch.zeros(self.nBins)
    self.bestBetti=10
    self.saveBest=saveBest
    self.epoch=0
    self.epochsSkipSaving=epochsSkipSaving

    self.metrics = []

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

    mean_metrics = np.nanmean(self.metrics, 0).tolist()
    
    self.metrics.clear()
    
    buffer = ','.join(str(x)[:6] for x in mean_metrics)
    text_file=open(self.log_file, "a")
    text_file.write(buffer)
    text_file.write("\n")
    text_file.close()

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

    return mean_metrics
