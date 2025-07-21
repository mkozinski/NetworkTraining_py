import os
import torch

from .writer_text import WriterText

class LoggerBasic:

    param_list=["loss",]

    def __init__(self, log_dir, name, saveNetEvery=500, saveAndKeep=False,
        writer=None):

        self.log_dir=log_dir
        
        self.writer=writer

        self.loss=0
        self.count=0
        self.saveNetEvery=saveNetEvery
        self.saveAndKeep=saveAndKeep
        self.epoch=0

        if self.writer is None:
            self.writer=WriterText(self.log_dir,"log_"+name+"Basic.txt",self.param_list)

    def add(self,img,output,target,l,net=None,optim=None):
        self.loss+=l
        self.count+=1

    def logEpoch(self,net=None,optim=None,scheduler=None):

        self.writer.write({"loss":self.loss/self.count})

        lastLoss=self.loss
        self.loss=0
        self.count=0
        self.epoch+=1
        if self.epoch % self.saveNetEvery == 0:
          if self.saveAndKeep:
            fname='epoch_'+str(self.epoch)+'.pth'
          else:
            fname='last.pth'
          if net:
            nfname='net_'+fname
            torch.save({'epoch': self.epoch, 'state_dict': net.state_dict()},
                       os.path.join(self.log_dir,nfname))
          if optim:
            ofname='optim_'+fname
            torch.save({'epoch': self.epoch, 'state_dict': optim.state_dict()},
                       os.path.join(self.log_dir,ofname))
          if scheduler:
            ofname='scheduler_'+fname
            torch.save({'epoch': self.epoch, 'state_dict': scheduler.state_dict()},
                       os.path.join(self.log_dir,ofname))
        return lastLoss
