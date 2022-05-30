import torch
from torch.utils.data import Dataset
import numpy as np

class DatasetDisk(Dataset):

  def __init__(self, img, lbl, augment,loadfun=np.load):
    self.img=img
    self.lbl=lbl
    self.augment=augment
    self.loadfun=loadfun

  def __len__(self):
    return len(self.lbl)

  def __getitem__(self, idx):

    img=self.loadfun(self.img[idx])
    lbl=self.loadfun(self.lbl[idx])

    i,l=self.augment(img,lbl)
    return  i,l

