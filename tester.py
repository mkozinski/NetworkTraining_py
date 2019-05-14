import torch
from torch.autograd import Variable
import numpy as np
import sys
import time

class tester:

  def __init__(self, test_loader, logger):
    self.dataLoader=test_loader
    self.logger=logger

  def test(self, net):
    net.eval()
    with torch.no_grad():
      self.di=iter(self.dataLoader)
      local_iter=0
      t0=time.time()
      while True:
        try:
          data=next(self.di)
          img, lbl = data
          img, lbl = img.cuda(), lbl.long().cuda()
          img, lbl = img, lbl
          out= net.forward(img)
          self.logger.add(0,out,lbl)
          local_iter+=1
          t1=time.time()
          if t1-t0>3:
            sys.stdout.write('\rTest iter: %8d' % (local_iter))
            t0=t1
        except StopIteration:
          self.logger.logEpoch(net)
          break
    net.train()
  
