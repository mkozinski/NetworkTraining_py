import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.optim as optim
import numpy as np
import sys
import time

class trainer:

  def __init__(self, net, train_loader, optimizer, loss_function, logger, tester, test_every,lr_scheduler=None,
               lrStepPer='batch'):
    self.net=net
    self.dataLoader=train_loader
    self.optimizer=optimizer
    self.crit=loss_function.cuda()
    self.logger=logger
    self.di=iter(self.dataLoader)
    self.epoch=0
    self.tot_iter=0
    self.prev_iter=self.tot_iter
    self.test_every=test_every
    self.tester=tester
    self.lr_scheduler=lr_scheduler
    self.lrStepPer=lrStepPer # 'batch' or 'epoch'

  def train(self, numiter):
    self.net.train()
    local_iter=0
    t0=time.time()
    while local_iter<numiter:
      try:
        img, lbl=next(self.di)
        self.optimizer.zero_grad()
        img = Variable(img.cuda()) 
        out= self.net.forward(img)
        loss = self.crit(out, lbl.cuda())
        loss.backward()
        self.optimizer.step()
        self.logger.add(loss.data.cpu().numpy(),out,lbl)
        local_iter+=1
        self.tot_iter+=1
        if self.lr_scheduler and self.lrStepPer=='batch':
            self.lr_scheduler.step()
        t1=time.time()
        if t1-t0>3:
          sys.stdout.write('\rIter: %8d\tEpoch: %6d\tTime/iter: %6f' % (self.tot_iter, self.epoch, (t1-t0)/(self.tot_iter-self.prev_iter)))
          t0=t1
          self.prev_iter=self.tot_iter
      except StopIteration:
        lastLoss=self.logger.logEpoch(self.net)
        self.epoch+=1
        self.di=iter(self.dataLoader)
        if self.test_every and self.epoch%self.test_every==0:
          self.tester.test(self.net)
        if self.lr_scheduler and self.lrStepPer=='epoch':
          self.lr_scheduler.step(lastLoss)
