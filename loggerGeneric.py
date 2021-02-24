import os
import torch
from apex import amp

class LoggerGeneric:
    def __init__(self, log_dir, name, compute_logged_val, 
                 saveMin=False, saveMax=False, apex_opt_level=None):
        self.log_dir=log_dir
        self.log_file=os.path.join(self.log_dir,"log_"+name+".txt")
        self.name=name

        self.compute_logged_val=compute_logged_val

        text_file = open(self.log_file, "w")
        text_file.close()
        self.val=0
        self.count=0
        self.epoch=0
        self.min=float("inf")
        self.max=float("-inf")
        self.saveMin=saveMin
        self.saveMax=saveMax
        self.apex_opt_level = apex_opt_level

    def add(self,img,output,target,l,net=None,optim=None):
        self.val+=self.compute_logged_val(img,output,target,l,net,optim)
        self.count+=1

    def logEpoch(self,net=None,optim=None,scheduler=None):
        text_file = open(self.log_file, "a")
        text_file.write(str(self.val/self.count))
        text_file.write('\n')
        text_file.close()
        lastVal=self.val
        self.val=0
        self.count=0
        self.epoch+=1
        if self.saveMin and lastVal < self.min:
            self.min=lastVal
            filename_base = self.name + '_min.pth'

            if net:
                torch.save({'state_dict':net.state_dict()},
                           os.path.join(self.log_dir, 'net_' + filename_base))
            if optim:
                torch.save({'state_dict':optim.state_dict()},
                           os.path.join(self.log_dir, 'optim_' + filename_base))
            if scheduler:
                torch.save({'state_dict':optim.state_dict()},
                           os.path.join(self.log_dir, 'scheduler_' + filename_base))
            if self.apex_opt_level is not None:
                torch.save({'state_dict': amp.state_dict(), 'opt_level': self.apex_opt_level},
                           os.path.join(self.log_dir, 'amp_' + filename_base))

        if self.saveMax and lastVal > self.max:
            self.max=lastVal
            filename_base = self.name + '_max.pth'
            if net:
                torch.save({'state_dict': net.state_dict()},
                           os.path.join(self.log_dir, 'net_' + filename_base))
            if optim:
                torch.save({'state_dict': optim.state_dict()},
                           os.path.join(self.log_dir, 'optim_' + filename_base))
            if self.apex_opt_level is not None:
                torch.save({'state_dict': amp.state_dict(), 'opt_level': self.apex_opt_level},
                           os.path.join(self.log_dir, 'amp_' + filename_base))
        
        return lastVal
