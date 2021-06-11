import torch
import sys
import time


class Tester:
    def __init__(self, test_loader, logger, preproc=None, device=None):
        # test_loader
        # logger
        # preproc  -  a function for batch pre-processing
        #             takes a pair img,lbl, returns a pair img,lbl
        #             I use it mostly to copy the data to gpu
        self.dataLoader=test_loader
        self.logger=logger
        if preproc is None:
            self.preproc = lambda img_lbl_pair: img_lbl_pair
        else:
            self.preproc=preproc
        self.device = device

    def test(self, net):
        net.eval()
        with torch.no_grad():
            self.di = iter(self.dataLoader)
            local_iter = 0
            t0=time.time()
            while True:
                try:
                    img, lbl = self.preproc(next(self.di))

                    out = net(img)
                    self.logger.add(img,out,lbl,0,net=net)
                    local_iter+=1
                    t1=time.time()
                    if t1-t0>3:
                        sys.stdout.write('\rTest iter: %8d' % (local_iter))
                        t0=t1
                except StopIteration:
                    self.logger.logEpoch(net=net)
                    break
        net.train()
  
