import os
import torch
import numpy as np
import sklearn.metrics

from .writer_text import WriterText

class LoggerIoU:
    def __init__(self, log_dir, name, nClasses, ignoredIdx, saveBest=False, 
                 preproc=lambda o,t: (o,t),writer=None):

        self.log_dir=log_dir
        self.name=name
        self.writer=writer

        self.nClasses=nClasses
        self.confMat=np.zeros((nClasses,nClasses))
        self.ignoredIdx=ignoredIdx
        self.saveBest=saveBest
        self.bestIoU=0
        self.preproc=preproc

        self.param_list=["IoU_"+str(i) for i in range(self.nClasses)]+["IoU",]

        if self.writer is None:
            self.writer=WriterText(self.log_dir,"log_"+name+"IoU.txt",self.param_list)

    def add(self,img,output,target,loss,net=None,optim=None):
        output,target=self.preproc(output,target)

        output=output.cpu().data
        output=output.numpy()
        outputClass=np.argmax(output, axis=1)
        oc=outputClass.flatten()

        tc=target.cpu().data.numpy().flatten()

        oc_valid=oc[tc!=self.ignoredIdx]
        tc_valid=tc[tc!=self.ignoredIdx]

        self.confMat+=sklearn.metrics.confusion_matrix
            (tc_valid,oc_valid,labels=np.array(range(self.nClasses)))

    def logEpoch(self,net=None,optim=None,scheduler=None):
        sums1=np.sum(self.confMat,axis=0)
        sums2=np.sum(self.confMat,axis=1)
        dg=np.diag(self.confMat)
        iou=np.zeros(dg.shape)
        iou=np.divide(dg.astype(np.float64),(sums1+sums2-dg).astype(np.float64),
                      out=iou, where=(dg!=0))
        mean_iou=np.mean(iou)

        log_dict={"IoU_"+str(i):iou[i] for i in range(self.nClasses)}
        log_dict["IoU"]=mean_iou
        self.writer.write(log_dict)

        self.confMat.fill(0)
        if mean_iou > self.bestIoU:
            self.bestIoU=mean_iou
            if self.saveBest:
                torch.save({'state_dict': net.state_dict()},
                            os.path.join(self.log_dir, 
                            'net_'+self.name+'_bestIoU.pth'))
                if net:
                    torch.save({'state_dict': net.state_dict()},
                                os.path.join(self.log_dir, 
                                'net_'+self.name+'_bestIoU.pth'))
                if optim:
                    torch.save({'state_dict': optim.state_dict()},
                                os.path.join(self.log_dir, 
                                'optim_'+self.name+'_bestIoU.pth'))
