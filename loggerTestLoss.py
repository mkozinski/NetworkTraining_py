import os
import torch


class LoggerTestLoss:
    def __init__(self, log_dir, name, loss_function, save_net_every=500, save_and_keep=False):
        self.log_dir = log_dir
        self.log_file = os.path.join(self.log_dir, f'log_{name}.txt')
        text_file = open(self.log_file, 'w')
        text_file.close()

        self.loss_function = loss_function
        self.loss = 0
        self.count = 0
        self.save_net_every = save_net_every
        self.save_and_keep = save_and_keep
        self.epoch = 0

    def add(self, img, output, target, loss_components=None, net=None, optim=None):
        self.loss += self.loss_function(output, target).item()
        self.count += 1

    def logEpoch(self, net=None, optim=None, scheduler=None):
        with open(self.log_file, 'a') as text_file:
            text_file.write(f'{self.loss / self.count}\n')

        lastLoss = self.loss
        if self.epoch % self.save_net_every == 0:
            if self.save_and_keep:
                filename = f'epoch_{self.epoch}.pth'
            else:
                filename = 'last.pth'

            if net:
                nfname = 'net_' + filename
                torch.save({'epoch': self.epoch, 'state_dict': net.state_dict()},
                           os.path.join(self.log_dir, nfname))
            if optim:
                ofname = 'optim_' + filename
                torch.save({'epoch': self.epoch, 'state_dict': optim.state_dict()},
                           os.path.join(self.log_dir, ofname))
            if scheduler:
                sfname = 'scheduler_' + filename
                torch.save({'epoch': self.epoch, 'state_dict': scheduler.state_dict()},
                           os.path.join(self.log_dir, sfname))

        return lastLoss
