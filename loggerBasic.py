import os
import torch
from apex import amp


class LoggerBasic:
    def __init__(self, log_dir, name, saveNetEvery=500, saveAndKeep=False, apex_opt_level=None, n_losses=1):
        self.log_dir=log_dir
        self.log_file=os.path.join(self.log_dir,"log_"+name+".txt")
        self.apex_opt_level = apex_opt_level

        text_file = open(self.log_file, "w")
        text_file.close()
        self.n_losses = n_losses
        self.loss = [0] * n_losses
        self.count = [0] * n_losses
        self.saveNetEvery=saveNetEvery
        self.saveAndKeep=saveAndKeep
        self.epoch=0

    def add(self, img, output, target, loss_components, net=None, optim=None):
        for i, loss_component in enumerate(loss_components):
            self.loss[i] += loss_component
            self.count[i] += 1

    def logEpoch(self, net=None, optim=None, scheduler=None):
        text_file = open(self.log_file, "a")
        losses = list(map(lambda loss_count: str(loss_count[0] / loss_count[1]), zip(self.loss, self.count)))
        text_file.write(','.join(losses))
        text_file.write('\n')
        text_file.close()
        lastLoss = sum(self.loss)
        self.loss = [0] * self.n_losses
        self.count = [0] * self.n_losses
        self.epoch += 1
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

            if self.apex_opt_level is not None:
                amp_filename = 'amp_' + fname
                torch.save({'epoch': self.epoch, 'state_dict': amp.state_dict(), 'opt_level': self.apex_opt_level},
                           os.path.join(self.log_dir, amp_filename))

        return lastLoss
