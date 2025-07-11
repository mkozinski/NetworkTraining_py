import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.optim as optim
import numpy as np
import sys
import time
import wandb

from sys import path
path.append("../")
path.append("./")
from driveMaximin.cl_metric import cldDice
from ripser import lower_star_img

import cv2
import matplotlib.pyplot as plt

def betti_number(img_true, pred):
    diags_pred = lower_star_img(pred)[:-1]
    diags = lower_star_img(img_true)[:-1]
    return len(diags_pred) - len(diags)

def get_metrics(img_predicted, img_true, img_mask):
    '''
    binary segmented feature maps
    '''
    tp = float(np.logical_and(img_mask, np.logical_and(img_true, img_predicted)).sum())
    fp = float(np.logical_and(img_mask, np.logical_and(1 - img_true, img_predicted)).sum())
    tn = float(np.logical_and(img_mask, np.logical_and(1 - img_true, 1 - img_predicted)).sum())
    fn = float(np.logical_and(img_mask, np.logical_and(img_true,  1 - img_predicted)).sum())


    if (tp+fp+tn+fn) != 0:
      accuracy = (tp + tn) / (tp + fp + tn + fn)
    else:
      accuracy = float('NaN')

    if (tp+fn) != 0:
      recall = tp / (tp + fn)
    else:
      recall = float('NaN')
    
    if (2*tp+fp+fn) != 0:
      dice = 2 * tp / (2 * tp + fp + fn)
    else:
      dice = float('NaN')

    betti_dif = 0
    
    for i in np.random.uniform(0,383,100):
        i = int(i)
        betti_dif += abs(betti_number(img_true[i:i+64,i:i+64], img_predicted[i:i+64,i:i+64]))
    
    return accuracy, recall, dice , betti_dif/100

class trainer:

  def __init__(self, net, train_loader, optimizer, loss_function, logger, 
               tester, test_every,lr_scheduler=None, lrStepPer='batch', 
               preprocImgLbl=None, apex_opt_level=None, update_every=1,
               learningRate=1e-3, batchSize=5, alpha=0.5,sgm=5,a=0.8):
    # net
    # train_loader
    # optimizer
    # loss_function
    # logger
    # tester
    # test_every - a test is run every "test_every" epochs
    # lr_scheduler - 
    # lrStepPer - 'batch' or 'epoch' - make a learning rate update
    #             according to number of batches or epochs
    # preprocImgLbl - a function for batch preprocessing
    #                 I use it to copy the batches to gpu
    self.net=net
    self.dataLoader=train_loader
    self.optimizer=optimizer
    self.crit=loss_function
    self.logger=logger
    self.di=iter(self.dataLoader)
    self.epoch=0
    self.tot_iter=0
    self.prev_iter=self.tot_iter
    self.test_every=test_every
    self.tester=tester
    self.lr_scheduler=lr_scheduler
    self.lrStepPer=lrStepPer # 'batch' or 'epoch'
    self.update_every=update_every
    if preprocImgLbl==None:
      self.preproc=lambda img,lbl: img,lbl
    else:
      self.preproc=preprocImgLbl
#    self.apex_opt_level=apex_opt_level
#    if self.apex_opt_level is not None:
#       self.net,self.optimizer=amp.initialize(self.net,self.optimizer,
#           opt_level=self.apex_opt_level)
    if learningRate is not None:
      self.learningRate = learningRate
      self.batchSize = batchSize
      self.alpha = alpha
      self.sgm = sgm
      self.a = a

      self.run = wandb.init(
        entity="phesox",
        project="maximin_training",
        config={
          "learning_rate": self.learningRate,
          "architecture": "U-Net",
          "dataset": "DRIVE",
          "batch size": self.batchSize,
          "alpha": self.alpha,
          "sgm": self.sgm,
          "a": self.a
        }
      )
    else:
      self.learningRate = 0
      self.batchSize = 0
      self.alpha = 0

  def train(self, numiter):
    self.net.train()
    local_iter=0
    t0=time.time()
    metrics = []
    while local_iter<numiter:
      try:
        img, lbl=next(self.di)
        img,lbl=self.preproc(img,lbl)


        if local_iter % self.update_every==0:
          self.optimizer.zero_grad()

        out= self.net.forward(img)
        
        loss = self.crit(out, lbl)

#        if self.apex_opt_level is not None:
#            with amp.scale_loss(loss,self.optimizer) as scaled_loss:
#                scaled_loss.backward()
#        else:
#            loss.backward()
        loss.backward()


        if local_iter % self.update_every== self.update_every-1:
          self.optimizer.step()


        with torch.no_grad():
          self.logger.add(img,out,lbl,loss.item(),
                          net=self.net,optim=self.optimizer)
          o=out.unsqueeze(0).cuda()
          e=torch.exp(o)
          p=e/(1+e)


        # let's  compute all the metrics for this iteration
        lbimg=(lbl[0][0]*255).cpu()
        primg=(p[0][0][0]*255).cpu() 
        maskimg=(lbl[1][0]*255).cpu()

        #print("lb max min", lbl[0][0].max(), lbl[0][0].min())
        #print("lbimg max min", lbimg.max(), lbimg.min())

        pr = (255*(primg<150)).detach().numpy().astype(np.uint16)
        lb = lbimg.detach().numpy().astype(np.uint16)

        #print("lb max min", lb.max(), lb.min())
        #print("lbimg max min", lbimg.max(), lbimg.min())

        #cv2.imshow("test", torch.cat([lbl[0][0].cpu(),p[0][0][0].cpu()],dim=1).numpy())
        #cv2.imshow("lb", lb)
        #cv2.imshow("pr", pr.astype(np.uint16))
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

        img_pred = (pr > 250)
        img_true = (lb != 0)
        img_mask = maskimg.detach().numpy() 
        img_pred = np.logical_and(img_pred, img_mask)

        accuracy, recall, dice, betti = get_metrics(img_pred, img_true, img_mask)
        cldice = cldDice(img_pred, img_true)        

        local_iter+=1
        self.tot_iter+=1
        if self.lr_scheduler and self.lrStepPer=='batch':
            self.lr_scheduler.step()

        t1=time.time()
        itertime = 2
        if t1-t0>0.1:
          itertime=(t1-t0)/(self.tot_iter-self.prev_iter)

          bsize_steps = np.int64(np.ceil(20/self.batchSize))
          numbers = [self.tot_iter%bsize_steps,bsize_steps-1-self.tot_iter%10]
          text = 'Iter: '+str(self.tot_iter)+'\t['+str(numbers[0]*'='+'>'+numbers[1]*'.')+'] '+str(numbers[0])+'/'+str(bsize_steps)+'\tTime/iter: '+str(itertime)[:4]+" s"

          sys.stdout.write('\r'+text)


          t0=t1
          self.prev_iter=self.tot_iter
        
        metrics.append([accuracy, recall, dice, betti, cldice,loss.item(),itertime])


      except StopIteration:

        lastLoss=self.logger.logEpoch(net=self.net,optim=self.optimizer,
            scheduler=self.lr_scheduler)
        
        metrics2 = np.array(metrics)
        mean_metrics = [np.nanmean(metrics2[:,0]),
                        np.nanmean(metrics2[:,1]),
                        np.nanmean(metrics2[:,2]),
                        np.nanmean(metrics2[:,3]),
                        np.nanmean(metrics2[:,4]),
                        np.nanmean(metrics2[:,5]),
                        np.nanmean(metrics2[:,6])]
        
        self.run.log({
          "acc": mean_metrics[0],
          "recall": mean_metrics[1],
          "dice": mean_metrics[2],
          "betti": mean_metrics[3],
          "cldice": mean_metrics[4],
          "loss": mean_metrics[5],
          "itertime": mean_metrics[6]
        })

        metrics.clear()
        


        self.epoch+=1
        self.di=iter(self.dataLoader)
        if self.test_every and self.epoch%self.test_every==0:
          self.tester.test(self.net)
        if self.lr_scheduler and self.lrStepPer=='epoch':
          self.lr_scheduler.step(lastLoss)


        

        print("\nEpoch "+str(self.epoch)+" :")
