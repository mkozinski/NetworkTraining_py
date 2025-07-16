import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.optim as optim
import numpy as np
import sys
import time
import wandb
import gudhi as gd

from sys import path
path.append("../")
path.append("./")

import cv2
import matplotlib.pyplot as plt

from NetworkTraining_py.metrics import get_metrics

class trainer:

  def __init__(self, net, train_loader, optimizer, loss_function, logger, 
               tester, test_every,lr_scheduler=None, lrStepPer='batch', 
               preprocImgLbl=None, apex_opt_level=None, update_every=1,
               learningRate=1e-3, batchSize=5, alpha=0.5,sgm=5,a=0.8):
    # net
    # train_loader
    # optimizer
    # loss_function
    # logger
    # tester
    # test_every - a test is run every "test_every" epochs
    # lr_scheduler - 
    # lrStepPer - 'batch' or 'epoch' - make a learning rate update
    #             according to number of batches or epochs
    # preprocImgLbl - a function for batch preprocessing
    #                 I use it to copy the batches to gpu
    self.net=net
    self.dataLoader=train_loader
    self.optimizer=optimizer
    self.crit=loss_function
    self.logger=logger
    self.di=iter(self.dataLoader)
    self.epoch=0
    self.tot_iter=0
    self.prev_iter=self.tot_iter
    self.test_every=test_every
    self.tester=tester
    self.lr_scheduler=lr_scheduler
    self.lrStepPer=lrStepPer # 'batch' or 'epoch'
    self.update_every=update_every
    if preprocImgLbl==None:
      self.preproc=lambda img,lbl: img,lbl
    else:
      self.preproc=preprocImgLbl
#    self.apex_opt_level=apex_opt_level
#    if self.apex_opt_level is not None:
#       self.net,self.optimizer=amp.initialize(self.net,self.optimizer,
#           opt_level=self.apex_opt_level)
    if learningRate is not None:
      self.learningRate = learningRate
      self.batchSize = batchSize
      self.alpha = alpha
      self.sgm = sgm
      self.a = a

      self.run = wandb.init(
        entity="phesox",
        project="maximin_training",
        config={
          "learning_rate": self.learningRate,
          "architecture": "U-Net",
          "dataset": "TopoMortar",
          "batch size": self.batchSize,
          "alpha": self.alpha,
          "sgm": self.sgm,
          "a": self.a
        }
      )
    else:
      self.learningRate = 0
      self.batchSize = 0
      self.alpha = 0

  def train(self, numiter):
    self.net.train()
    local_iter=0
    t0=time.time()
    itertime_array = []
    while local_iter<numiter:
      try:
        img, lbl=next(self.di)
        img,lbl=self.preproc(img,lbl)


        if local_iter % self.update_every==0:
          self.optimizer.zero_grad()

        out= self.net.forward(img)
        
        loss = self.crit(out, lbl)

#        if self.apex_opt_level is not None:
#            with amp.scale_loss(loss,self.optimizer) as scaled_loss:
#                scaled_loss.backward()
#        else:
#            loss.backward()
        loss.backward()


        if local_iter % self.update_every== self.update_every-1:
          self.optimizer.step()


        with torch.no_grad():
          self.logger.add(img,out,lbl,loss.item(),
                          net=self.net,optim=self.optimizer)
        
        local_iter+=1
        self.tot_iter+=1
        if self.lr_scheduler and self.lrStepPer=='batch':
            self.lr_scheduler.step()

        t1=time.time()
        itertime = 2
        if t1-t0>0.1:
          itertime=(t1-t0)/(self.tot_iter-self.prev_iter)

          bsize_steps = np.int64(np.ceil(350/self.batchSize))
          numbers = [self.tot_iter%bsize_steps,bsize_steps-1-self.tot_iter%10]
          text = 'Iter: '+str(self.tot_iter)+'\t['+str(numbers[0]*'='+'>'+numbers[1]*'.')+'] '+str(numbers[0])+'/'+str(bsize_steps)+'\tTime/iter: '+str(itertime)[:4]+" s"

          sys.stdout.write('\r'+text)


          t0=t1
          self.prev_iter=self.tot_iter
        
        itertime_array.append(itertime)


      except StopIteration:

        lastLoss, metrics=self.logger.logEpoch(net=self.net,optim=self.optimizer,
            scheduler=self.lr_scheduler)
        
        iteravg = np.mean(itertime_array)
        
        self.run.log({
          "acc": metrics[0],
          "recall": metrics[1],
          "dice": metrics[2],
          "cldice": metrics[3],
          "betti": metrics[4],
          "betti (topomortar)": metrics[5],
          "loss": metrics[6],
          "itertime": iteravg
        })

        itertime_array.clear()

        self.epoch+=1
        self.di=iter(self.dataLoader)
        if self.test_every and self.epoch%self.test_every==0:
          self.tester.test(self.net)
        if self.lr_scheduler and self.lrStepPer=='epoch':
          self.lr_scheduler.step(lastLoss)

        print("\nEpoch "+str(self.epoch)+" :")
