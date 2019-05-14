import os
import torch

class LoggerBasic:
    def __init__(self, log_dir, name, saveNetEvery=500):
        self.log_dir=log_dir
        self.log_file=os.path.join(self.log_dir,"log"+name+".txt")

        text_file = open(self.log_file, "w")
        text_file.close()
        self.loss=0
        self.count=0
        self.saveNetEvery=saveNetEvery
        self.epoch=0

    def add(self,l,output,target):
        self.loss+=l
        self.count+=1

    def logEpoch(self,net):
        text_file = open(self.log_file, "a")
        text_file.write(str(self.loss/self.count))
        text_file.write('\n')
        text_file.close()
        lastLoss=self.loss
        self.loss=0
        self.count=0
        self.epoch+=1
        if self.epoch % self.saveNetEvery == 0:
          torch.save({'epoch': self.epoch,
                      'state_dict': net.state_dict()},
                       os.path.join(self.log_dir, 
                       'net_last.pth'))
        return lastLoss
