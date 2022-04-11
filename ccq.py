
from math import ceil,floor
import torch as th
from torch import nn as nn
from torch.nn import functional as f

def getDisc2d(x,radius,padv):
    
    r=ceil(radius)
    y=f.pad(x,(r,r,r,r),mode='constant',value=padv)
    
    stack=[]
    
    inds=[slice(None,None)]*(x.dim()-2)+[slice(r,x.shape[-2]+r),slice(r,x.shape[-1]+r)]
    
    for i in range(-r,r+1):
        yi=th.roll(y,i,dims=-2)
        for j in range(-r,r+1):
            if (i*i+j*j)<=radius*radius:
                yij=th.roll(yi,j,dims=-1)
                stack.append(yij[inds])
                
    stacked=th.stack(stack,dim=-1)
    
    return stacked

def minValInDisc2d(x,radius):
    # for a 2D array x,
    # compute for each point the lowest value within a distance of less than `radius' pixels/voxels
    
    type_info=th.finfo(x.dtype) if x.dtype.is_floating_point else th.iinfo(x.dtype)
    maxval=type_info.max
    
    discs=getDisc2d(x,radius,maxval)
    
    minv, _=th.min(discs,dim=-1)
    return minv

def maxValInDisc2d(x,radius):
    # for a 2D array x,
    # compute for each point the highest value within a distance of less than `radius' pixels/voxels
    
    type_info=th.finfo(x.dtype) if x.dtype.is_floating_point else th.iinfo(x.dtype)
    minval=type_info.min
    
    discs=getDisc2d(x,radius,minval)
    
    maxv, _=th.max(discs,dim=-1)
    return maxv

from scipy.ndimage.morphology import distance_transform_edt
import numpy as np

def reverse(t):
    #idx = [i for i in range(t.size(0)-1, -1, -1)]
    #idx = th.LongTensor(idx)
    #it  = t.index_select(0, idx)
    it = t.flip([0])
    return it

def ccqHistograms(pr, gt, val_range, radius, pr_is_dm=True, gt_is_dm=True, nbins=10000):
    if gt_is_dm:
        gt_dm=gt
        gt= gt_dm==0
    else:
        maxval=th.max(gt)
        gt_dm=th.from_numpy(distance_transform_edt(maxval-gt.cpu().numpy())).to(gt)
        gt= gt==maxval
        
    gt_dilated=gt_dm<radius
    
    pr_close_to_gt=pr[gt_dilated]
    pr_far_from_gt=pr[th.logical_not(gt_dilated)]
    hist_pr_close_to_gt=th.histc(pr_close_to_gt,nbins,val_range[0],val_range[1]).long()
    hist_pr_far_from_gt=th.histc(pr_far_from_gt,nbins,val_range[0],val_range[1]).long()
    
    if pr_is_dm:
        pr_for_gt=minValInDisc2d(pr,radius)[gt]
        hist_pr_gt=th.histc(pr_for_gt,nbins,val_range[0],val_range[1]).long()
    else:
        pr_for_gt=maxValInDisc2d(pr,radius)[gt]
        hist_pr_gt=th.histc(pr_for_gt,nbins,val_range[0],val_range[1]).long()
        
        hist_pr_close_to_gt=reverse(hist_pr_close_to_gt).long()
        hist_pr_far_from_gt=reverse(hist_pr_far_from_gt).long()
        hist_pr_gt         =reverse(hist_pr_gt)         .long()
        
    #print("\n",pr_for_gt,"\n",hist_pr_gt)
     
    #return hist_pr_gt.cumsum(dim=0,dtype=th.long),\
    #       hist_pr_close_to_gt.cumsum(dim=0,dtype=th.long),\
    #       hist_pr_far_from_gt.cumsum(dim=0,dtype=th.long)
    return hist_pr_gt.cumsum(dim=0),\
           hist_pr_close_to_gt.cumsum(dim=0),\
           hist_pr_far_from_gt.cumsum(dim=0)
    
    
def ccqFromHistograms(hist_pr_gt,
                      hist_pr_close_to_gt,
                      hist_pr_far_from_gt):
    # length of matched reference/length of reference
    completeness=hist_pr_gt.double()/max(hist_pr_gt[-1],1.0)
    # length of matched extraction/length of extraction
    correctness =hist_pr_close_to_gt.double()/th.maximum(hist_pr_close_to_gt+hist_pr_far_from_gt,
                                                         th.tensor([1.0]).expand_as(hist_pr_close_to_gt))
    # length of matched extraction/(length of extraction + length of unmatched reference)
    quality     =hist_pr_close_to_gt.double()/th.maximum(hist_pr_close_to_gt+hist_pr_far_from_gt+hist_pr_gt[-1]-hist_pr_gt,
                                                         th.tensor([1.0]).expand_as(hist_pr_close_to_gt))
    
    return completeness,correctness,quality

def ccq(prs, gts, val_range, radius, pr_is_dm=True, gt_is_dm=True, nbins=10000):
    hist_pr_gt         =th.zeros(nbins,dtype=th.long)
    hist_pr_close_to_gt=th.zeros(nbins,dtype=th.long)
    hist_pr_far_from_gt=th.zeros(nbins,dtype=th.long)
    
    for pr,gt in zip(prs,gts):
        hg,hc,hf=ccqHistograms(pr, gt, val_range, radius, pr_is_dm=pr_is_dm, gt_is_dm=gt_is_dm, nbins=nbins)
        hist_pr_gt+=hg
        hist_pr_close_to_gt+=hc
        hist_pr_far_from_gt+=hf
        
    return ccqFromHistograms(hist_pr_gt,hist_pr_close_to_gt,hist_pr_far_from_gt)


