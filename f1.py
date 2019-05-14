import torch
import sys
import numpy as np

def reverse(t):
    idx = [i for i in range(t.size(0)-1, -1, -1)]
    idx = torch.LongTensor(idx)
    it  = t.index_select(0, idx)
    return it

def PRFromHistograms(hPos,hNeg):
    #print("hPos",hPos,"hNeg",hNeg)
    positives=hPos.sum()
    negatives=hNeg.sum()
    #print("positives, negatives", positives, negatives)
    truepositives =reverse(hPos.clone()).long().cumsum(dim=0)
    falsepositives=reverse(hNeg.clone()).long().cumsum(dim=0)
    predpositives=torch.add(truepositives,falsepositives)
    #protect against zero division
    predpositives[predpositives==0]=1
    precision=torch.Tensor.div(truepositives.float(),predpositives.float())
    recall   =torch.Tensor.div(truepositives.float(),positives.float())
    precision[precision<=0]=sys.float_info.epsilon
    recall   [recall   <=0]=sys.float_info.epsilon
    #print("precision,recall",precision,recall)
    return precision,recall

def PRFromDiscrete(nTruePos,nPredPos,nGtPos):
    truepositives =nTruePos
    falsepositives=nPredPos-nTruePos
    positives     =nGtPos
    predpositives =nPredPos
    if predpositives==0: predpositives =1 #protect against zero division
    precision=truepositives/float(predpositives)
    recall   =truepositives/float(positives)
    if precision<=0: precision=sys.float_info.epsilon
    if recall   <=0: recall   =sys.float_info.epsilon
    return precision,recall

def PRFromOutGt(outps,targs,nbins=10000):
    hPos=torch.zeros(nbins)
    hNeg=torch.zeros(nbins)
    for o,t in zip(outps,targs):
        pos=o[t==1]
        neg=o[t==0]
        hPos+=torch.from_numpy(pos.astype(np.float32)).histc(nbins,0,1)
        hNeg+=torch.from_numpy(neg.astype(np.float32)).histc(nbins,0,1)
    precision,recall=PRFromHistograms(hPos,hNeg)
    f1s=F1FromPR(precision,recall)
    f=f1s.max()
    return f

def F1FromPR(p,r):
    suminv=torch.pow(p,-1)+torch.pow(r,-1)
    f1s=torch.pow(suminv,-1).mul(2)
    #print("f1s",f1s)
    return f1s


