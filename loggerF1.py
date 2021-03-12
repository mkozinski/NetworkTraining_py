import os
import torch
from apex import amp
from .f1 import PRFromHistograms, F1FromPR


def reverse(t):
    idx = [i for i in range(t.size(0)-1, -1, -1)]
    idx = torch.LongTensor(idx)
    it  = t.index_select(0, idx)
    return it


class LoggerF1:
    def __init__(self,logdir,fname,preproc=lambda o,t: (o,t),
                 nBins=10000,saveBest=False, apex_opt_level=None):
        self.log_dir=logdir
        self.name=fname
        self.log_file=os.path.join(self.log_dir,"logF1_"+self.name+".txt")
        text_file = open(self.log_file, "w")
        text_file.close()
        self.preproc=preproc
        self.nBins=nBins
        self.hPos=torch.zeros(self.nBins)
        self.hNeg=torch.zeros(self.nBins)
        self.bestF1=0
        self.saveBest=saveBest
        self.apex_opt_level = apex_opt_level

    def add(self,img,output,target,l,net=None,optim=None):
        o,t=self.preproc(output,target)
        pos=o[t==1]
        neg=o[t==0]
        self.hPos+=pos.histc(self.nBins,0,1)
        self.hNeg+=neg.histc(self.nBins,0,1)

    def logEpoch(self,net=None,optim=None,scheduler=None):
        precision,recall=PRFromHistograms(self.hPos,self.hNeg)
        f1s=F1FromPR(precision,recall)
        f=f1s.max()
        text_file=open(self.log_file, "a")
        text_file.write('{}\n'.format(f))
        text_file.close()
        if self.saveBest and f > self.bestF1:
            self.bestF1 = f
            filename_base = self.name + '_bestF1.pth'
            if net:
                torch.save({'state_dict': net.state_dict()},
                           os.path.join(self.log_dir, 'net_' + filename_base))
            if optim:
                torch.save({'state_dict': optim.state_dict()},
                           os.path.join(self.log_dir, 'optim_' + filename_base))

            if self.apex_opt_level is not None:
                torch.save({'state_dict': amp.state_dict(), 'opt_level': self.apex_opt_level},
                           os.path.join(self.log_dir, 'amp_' + filename_base))

        self.hPos.zero_()
        self.hNeg.zero_()
