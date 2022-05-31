import torch
from torch.utils.data import Dataset
import numpy as np
from os import path

class DatasetDisk(Dataset):

  def __init__(self, img, lbl, augment,loadfun=np.load,rootdir=""):
    self.img=img
    self.lbl=lbl
    self.augment=augment
    self.loadfun=loadfun
    self.rootdir=rootdir

  def __len__(self):
    return len(self.lbl)

  def __getitem__(self, idx):

    img=self.loadfun(path.join(self.rootdir,self.img[idx]))
    lbl=self.loadfun(path.join(self.rootdir,self.lbl[idx]))

    i,l=self.augment(img,lbl)
    return  i,l

