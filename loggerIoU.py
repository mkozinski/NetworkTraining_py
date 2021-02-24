import os
import torch
import numpy as np
import sklearn.metrics
from apex import amp

class LoggerIoU:
    def __init__(self, log_dir, name, nClasses, ignoredIdx, saveBest=False, 
                 preproc=lambda o,t: (o,t), apex_opt_level=None):
        self.log_dir=log_dir
        self.name=name
        self.log_file=os.path.join(self.log_dir,"logIou_"+self.name+".txt")
        text_file = open(self.log_file, "w")
        text_file.close()

        self.nClasses=nClasses
        self.confMat=np.zeros((nClasses,nClasses))
        self.ignoredIdx=ignoredIdx
        self.saveBest=saveBest
        self.bestIoU=0
        self.preproc=preproc
        self.apex_opt_level = apex_opt_level

    def add(self,img,output,target,loss,net=None,optim=None):
        output,target=self.preproc(output,target)

        output=output.cpu().data
        output=output.numpy()
        outputClass=np.argmax(output, axis=1)
        oc=outputClass.flatten()

        tc=target.cpu().data.numpy().flatten()

        oc_valid=oc[tc!=self.ignoredIdx]
        tc_valid=tc[tc!=self.ignoredIdx]

        self.confMat += sklearn.metrics.confusion_matrix(tc_valid, oc_valid, labels=np.array(range(self.nClasses)))

    def logEpoch(self,net=None,optim=None,scheduler=None):
        sums1=np.sum(self.confMat,axis=0)
        sums2=np.sum(self.confMat,axis=1)
        dg=np.diag(self.confMat)
        iou=np.zeros(dg.shape)
        iou=np.divide(dg.astype(np.float64),(sums1+sums2-dg).astype(np.float64),
                      out=iou, where=(dg!=0))
        text_file = open(self.log_file, "a")
        for i in range(self.nClasses):
            text_file.write('{}\t'.format(iou[i]))
        mean_iou=np.mean(iou)
        text_file.write('{}\n'.format(mean_iou))
        text_file.close()
        self.confMat.fill(0)
        if mean_iou > self.bestIoU:
            self.bestIoU=mean_iou
            filename_base = self.name + '_bestIoU.pth'
            if self.saveBest:
                torch.save({'state_dict': net.state_dict()},
                           os.path.join(self.log_dir, 'net_' + filename_base))
                if net:
                    torch.save({'state_dict': net.state_dict()},
                               os.path.join(self.log_dir, 'net_' + filename_base))
                if optim:
                    torch.save({'state_dict': optim.state_dict()},
                               os.path.join(self.log_dir, 'optim_' + filename_base))
                if self.apex_opt_level is not None:
                    torch.save({'state_dict': amp.state_dict(), 'opt_level': self.apex_opt_level},
                               os.path.join(self.log_dir, 'amp_' + filename_base))
