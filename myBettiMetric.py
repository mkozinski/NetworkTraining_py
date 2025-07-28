import gudhi as gd

from heapq  import merge
from bisect import bisect, bisect_left
from copy import copy

import numpy as np

def pers(patch):
    # patch should have values in [0,1] ?

    cc = gd.CubicalComplex(top_dimensional_cells=1-patch)
    pers=cc.persistence()
    
    bd0th=[(b,d) for k,(b,d) in pers if k==0]
    bd1st=[(b,d) for k,(b,d) in pers if k==1]

    return bd0th,bd1st
    
def persCurve(bdlist):
    # bdlist is a lost of pairs (b,d)
    # where b is birth time and d is death time

    # birth increases the number of homology classes by one
    # death decreases this number by one
    births=[(b, 1) for (b,d) in sorted(bdlist,key=lambda a:a[0])]
    deaths=[(d,-1) for (b,d) in sorted(bdlist,key=lambda a:a[1])]
        
    bds=list(merge(births,deaths,key=lambda a:a[0]))

    bettifun=np.array([float(b) for a,b in bds]).cumsum()
    bettithr=np.array([      a  for a,b in bds])
    
    return bettifun,bettithr,bds

def bettiError(bettifun,truebetti):
    return [abs(b-truebetti) for b in bettifun]

def bettiErrorIncr(bettierr):
    return [bettierr[0],]+[b2-b1 for b2,b1 in zip(bettierr[1:],bettierr[:-1])]

def cropFun2Range(x,y,start,stop,val_at_minInf=0.0):
    # x,y represent a piecewise-constant function
    # x is a list a of arguments where the function value changes
    # y is a list of function values for the arguments given in x
    #
    # val_at_minInf is the value of the function to the left 
    # from the first entry in x and y.
    #
    # start,stop represent an interval
    # to which the function should be "cropped"
    # start is the left boundary of the interval, a number
    # stop is the right boundary of the interval, a number
    #
    # returned is a cropped function, 
    # x2,y2
    # with the same interpretation as x,y 
    # the list x2 starts with the "start" argument
    # and ends with the largest argument smaller or equal to "stop"
    # the list y2 is trimmed accordingly
    
    x=copy(x)
    y=copy(y)
    x.insert(0,float("-inf"))
    y.insert(0,val_at_minInf)
    
    # the value at "start" is stored to the left from the first x
    # such that x>start 
    start_idx=bisect(x,start)-1 
    stop_idx =bisect(x,stop )

    x2=x[start_idx:stop_idx]
    y2=y[start_idx:stop_idx]

    x2[ 0]=start
    
    return x2,y2

def removeDuplicates(x,y):
    x2=[]
    y2=[]
    if len(x)>0:
        x2=[x[0],]
        y2=[y[0],]
        for u,v in zip(x[1:],y[1:]):
            if x2[-1]==u:
                y2[-1]=v
            else:
                x2.append(u)
                y2.append(v)
    return x2,y2

def bettiErrorMeanMin(bettierr,bettithr,start,stop):

    be,bt=removeDuplicates(bettithr,bettierr)
    be,bt=cropFun2Range(be,bt,start,stop)
    
    bettierrfun=list(zip(be,bt))
        
    area=[]
    for (t1,e1),(t2,e2) in zip(bettierrfun[:-1],bettierrfun[1:]):
        area.append(e1*(t2-t1))
    last=bettierrfun[-1]
    area.append(last[1]*(stop-last[0]))
    
    meanerr= sum(area)/(stop-start)
    
    minthr,minerr=min(bettierrfun,key=lambda a:a[1])
    
    return meanerr,minerr,minthr
    
def bettiMeanMin(pred,gt, start,stop):
    
    gt0,  gt1  =pers(gt)
    trueb0=len(gt0)
    trueb1=len(gt1)

    pred0,pred1=pers(pred)
    bfun0,bthr0,_=persCurve(pred0)
    bfun1,bthr1,_=persCurve(pred1)

    berr0=bettiError(bfun0,trueb0)
    berr1=bettiError(bfun1,trueb1)
    
    berr0mean,berr0min,berr0minThr=bettiErrorMeanMin(berr0,bthr0,start,stop)
    berr1mean,berr1min,berr1minThr=bettiErrorMeanMin(berr1,bthr1,start,stop)

    return berr0mean,berr1mean, berr0min,berr1min, berr0minThr,berr1minThr
    
def updateBettiError(bettiErrIncrFun0,bettiErrIncrFun1, pred,gt):
    
    gt0,  gt1  =pers(gt)
    trueb0=len(gt0)
    trueb1=len(gt1)

    pred0,pred1=pers(pred)
    bfun0,bthr0,_=persCurve(pred0)
    bfun1,bthr1,_=persCurve(pred1)

    berr0=bettiError(bfun0,trueb0)
    berr1=bettiError(bfun1,trueb1)

    berr0incr=bettiErrorIncr(berr0)
    berr1incr=bettiErrorIncr(berr1)

    bfun0incr=list(zip(bthr0,berr0incr))
    bfun1incr=list(zip(bthr1,berr1incr))

    bettiErrIncrFun0=list(merge(bettiErrIncrFun0,bfun0incr,key=lambda a:a[0]))
    bettiErrIncrFun1=list(merge(bettiErrIncrFun1,bfun1incr,key=lambda a:a[0]))

    return bettiErrIncrFun0, bettiErrIncrFun1

def aggregateIncrErr(bettiErrIncrFun):
    
    thr=[a for (a,b) in bettiErrIncrFun]
    err=list(np.array([b for (a,b) in bettiErrIncrFun]).cumsum())
    
    return thr,err

def incrError2MeanMin(bettiErrIncrFun0,bettiErrIncrFun1, start, stop):

    bthr0,berr0=aggregateIncrErr(bettiErrIncrFun0)
    bthr1,berr1=aggregateIncrErr(bettiErrIncrFun1)
    
    berr0mean,berr0min,berr0minThr=bettiErrorMeanMin(berr0,bthr0,start,stop)
    berr1mean,berr1min,berr1minThr=bettiErrorMeanMin(berr1,bthr1,start,stop)
    
    return berr0mean,berr1mean, berr0min,berr1min, berr0minThr,berr1minThr

class MyBettiMetric():

    def __init__(self,start,stop):
        self.bettiIncrErr0=[]
        self.bettiIncrErr1=[]
        self.start=start
        self.stop =stop
        self.numcalls=0

    def reset(self):
        self.bettiIncrErr0=[]
        self.bettiIncrErr1=[]
        self.numcalls=0

    def add(self,pred,gt):
        self.bettiIncrErr0,self.bettiIncrErr1=\
            updateBettiError(self.bettiIncrErr0,self.bettiIncrErr1, pred,gt)
        self.numcalls+=1

    def computeBettiErrors(self):
        be0mean,be1mean,be0min,be1min,be0minThr,be1minThr=\
            incrError2MeanMin(self.bettiIncrErr0,self.bettiIncrErr1, self.start, self.stop)
        return be0mean/self.numcalls, be1mean/self.numcalls, \
            be0min/self.numcalls,be1min/self.numcalls, \
            be0minThr,be1minThr

if __name__=='__main__':

    import unittest
    from bettiMetricsTopo import BettiErrorMetric
    import torch as th
    from torch.nn.functional import pad

    class TestMyBetti(unittest.TestCase):

        def testMeanIndiv(self):
            preds=np.load("bettiTest_preds.npy")
            gts  =np.load("bettiTest_gts.npy")

            threshold=0.77
            center = 1-threshold

            mbm=MyBettiMetric(center-1e-6,center+1e-6)
            
            for p,g in zip(preds,gts):
                o=pad(th.from_numpy(p[np.newaxis,np.newaxis]),(0,0, 0,0, 1,0))
                t=    th.from_numpy(g[np.newaxis,np.newaxis])

                bm =BettiErrorMetric()
                bm(o>=threshold,t)
                berrors=bm.aggregate()
                bmeanerr=th.mean(berrors.float(),dim=0)

                mbm.add(p,g)
                be0mean,be1mean,be0min,be1min,be0minThr,be1minThr=mbm.computeBettiErrors()
                mbm.reset()

#                print("betti0 error, reference/computed",bmeanerr[0],be0mean)
#                print("betti1 error, reference/computed",bmeanerr[1],be1mean)

                self.assertEqual(bmeanerr[0],be0mean)
                self.assertEqual(bmeanerr[1],be1mean)

        def testMinIndiv(self):
            preds=np.load("bettiTest_preds.npy")
            gts  =np.load("bettiTest_gts.npy")

            start=0.25
            stop =0.75

            mbm=MyBettiMetric(start,stop )
            
            for p,g in zip(preds,gts):
                mbm.add(p,g)
                be0mean,be1mean,be0min,be1min,be0minThr,be1minThr=mbm.computeBettiErrors()
                mbm.reset()

                o=pad(th.from_numpy(p[np.newaxis,np.newaxis]),(0,0, 0,0, 1,0))
                t=    th.from_numpy(g[np.newaxis,np.newaxis])

                bm =BettiErrorMetric()
                bm(o>=1-be0minThr,t)
                berrors=bm.aggregate()
                berror0=berrors[0][0]

                bm =BettiErrorMetric()
                bm(o>=1-be1minThr,t)
                berrors=bm.aggregate()
                berror1=berrors[0][1]

#                print("betti0 error, reference/computed",berror0,be0min)
#                print("betti1 error, reference/computed",berror1,be1min)

                self.assertEqual(berror0,be0min)
                self.assertEqual(berror1,be1min)

        def testMeanMean(self):
            preds=np.load("bettiTest_preds.npy")
            gts  =np.load("bettiTest_gts.npy")

            threshold=0.77
            center = 1-threshold

            mbm=MyBettiMetric(center-1e-6,center+1e-6)
            bm =BettiErrorMetric()
            
            for p,g in zip(preds,gts):
                o=pad(th.from_numpy(p[np.newaxis,np.newaxis]),(0,0, 0,0, 1,0))
                t=    th.from_numpy(g[np.newaxis,np.newaxis])

                bm(o>=threshold,t)
                mbm.add(p,g)

            be0mean,be1mean,be0min,be1min,be0minThr,be1minThr=mbm.computeBettiErrors()

            berrors=bm.aggregate()
            bmeanerr=th.mean(berrors.float(),dim=0)

#            print("betti0 error, reference/computed",bmeanerr[0],be0mean)
#            print("betti1 error, reference/computed",bmeanerr[1],be1mean)

            self.assertEqual(bmeanerr[0],be0mean)
            self.assertEqual(bmeanerr[1],be1mean)

        def testMeanMin(self):
            preds=np.load("bettiTest_preds.npy")
            gts  =np.load("bettiTest_gts.npy")

            start=0.25
            stop =0.75

            mbm=MyBettiMetric(start,stop )
            
            for p,g in zip(preds,gts):
                mbm.add(p,g)

            be0mean,be1mean,be0min,be1min,be0minThr,be1minThr=mbm.computeBettiErrors()
            threshold0=1-be0minThr
            threshold1=1-be1minThr

            bm0 =BettiErrorMetric()
            bm1 =BettiErrorMetric()

            for p,g in zip(preds,gts):

                o=pad(th.from_numpy(p[np.newaxis,np.newaxis]),(0,0, 0,0, 1,0))
                t=    th.from_numpy(g[np.newaxis,np.newaxis])

                bm0(o>=threshold0,t)
                bm1(o>=threshold1,t)

            berrors=th.mean(bm0.aggregate().float(),dim=0)
            berror0=berrors[0]

            berrors=th.mean(bm1.aggregate().float(),dim=0)
            berror1=berrors[1]

#            print("betti0 error, reference/computed",berror0,be0min)
#            print("betti1 error, reference/computed",berror1,be1min)

            self.assertEqual(berror0,be0min)
            self.assertEqual(berror1,be1min)

        def testBettiMean(self):
            bettierr=[1,2,3,2,0]
            bettithr=[1,2,6,8,9]
            
            start=2.2
            stop =6.7
            mean =(3.8*2 + 0.7*3)/4.5
            me,mi,mt=bettiErrorMeanMin(bettierr,bettithr,start,stop)
            self.assertAlmostEqual(me,mean)
            self.assertEqual( mi,2 )
            self.assertEqual( mt,start)
            
            start=2
            stop =6.7
            mean =(4*2 + 0.7*3)/4.7
            me,mi,mt=bettiErrorMeanMin(bettierr,bettithr,start,stop)
            self.assertAlmostEqual(me,mean)
            self.assertEqual(mi,2)
            self.assertEqual(mt,start)
            
            start=2.2
            stop =8
            mean =(3.8*2 + 2*3)/5.8
            me,mi,mt=bettiErrorMeanMin(bettierr,bettithr,start,stop)
            self.assertAlmostEqual(me,mean)
            self.assertEqual(mi,2)
            self.assertEqual(mt,start)
            
            start=2
            stop =8
            mean =(4*2 + 2*3)/6
            me,mi,mt=bettiErrorMeanMin(bettierr,bettithr,start,stop)
            self.assertAlmostEqual(me,mean)
            self.assertEqual(mi,2)
            self.assertEqual(mt,start)
            
            bettierr=[1,2,1,2,0]
            bettithr=[1,2,6,8,9]
            start=2.2
            stop =8.7
            mean =(3.8*2 + 2*1 + 0.7*2)/6.5
            me,mi,mt=bettiErrorMeanMin(bettierr,bettithr,start,stop)
            self.assertAlmostEqual(me,mean)
            self.assertEqual(mi,1)
            self.assertEqual(mt,6)
            
            bettierr=[1,2,1,2,3]
            bettithr=[1,2,6,8,9]
            start=2.2
            stop =11.1
            mean =(3.8*2 + 2*1 + 1*2+2.1*3)/8.9
            me,mi,mt=bettiErrorMeanMin(bettierr,bettithr,start,stop)
            self.assertAlmostEqual(me,mean)
            self.assertEqual(mi,1)
            self.assertEqual(mt,6)
            
            bettierr=[1]
            bettithr=[3]
            start=2.2
            stop =11.0
            mean =(0.8*0 + 8*1)/8.8
            me,mi,mt=bettiErrorMeanMin(bettierr,bettithr,start,stop)
            self.assertAlmostEqual(me,mean)
            self.assertEqual(mi,0)
            self.assertEqual(mt,2.2)
            
            bettierr=[1]
            bettithr=[3]
            start=3.2
            stop =11.0
            mean =(7.8*1)/7.8
            me,mi,mt=bettiErrorMeanMin(bettierr,bettithr,start,stop)
            self.assertAlmostEqual(me,mean)
            self.assertEqual(mi,1)
            self.assertEqual(mt,3.2)
            
            bettierr=[1]
            bettithr=[3]
            start=0.0
            stop =0.5
            mean =(0.5*0)/0.5
            me,mi,mt=bettiErrorMeanMin(bettierr,bettithr,start,stop)
            self.assertAlmostEqual(me,mean)
            self.assertEqual(mi,0.0)
            self.assertEqual(mt,0.0)
    

    unittest.main()
