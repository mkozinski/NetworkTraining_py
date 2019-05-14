import numpy as np
import NetworkTraining_py.cropRoutines as cropRoutines
import torch

def targetCoords(sourceCoords,validCoords):
  cc=sourceCoords
  vc=validCoords
  tc=[]
  for i in range(len(cc)):
    tc.append(slice(cc[i].start+vc[i].start,cc[i].start+vc[i].stop))
  return tc

def processChunk(inChunk,cropSize,marginSize,startDim,net,outChannels=None):
  nc=cropRoutines.noCrops(inChunk.shape,cropSize,marginSize,startDim)
  size=np.array(inChunk.shape)
  if outChannels:
    size[1]=outChannels
  outChunk=np.zeros(tuple(size))
  for i in range(nc):
    cc,vc=cropRoutines.cropCoords(i,cropSize,marginSize,inChunk.shape,startDim)
    tc=targetCoords(cc,vc)
    crop=inChunk[tuple(cc)]
    o=net.forward(torch.from_numpy(crop).cuda())
    tc[1]=slice(0,size[1])
    vc[1]=slice(0,size[1])
    outChunk[tuple(tc)]=o.cpu().data.numpy()[tuple(vc)]
  return outChunk

def processChunk_v2(inChunk,cropSize,marginSize,startDim,net,outDims=None):
  # outDims specify the size of the output along k first dimensions along which
  # there is no cropping; it is a tuple
  nc=cropRoutines.noCrops(inChunk.shape,cropSize,marginSize,startDim)
  size=inChunk.shape
  if outDims:
    size=[]
    for k in range(len(outDims)):
      size.append(outDims[k])
    for k in range(startDim,inChunk.ndim):
      size.append(inChunk.shape[k])
  outChunk=np.zeros(tuple(size))
  for i in range(nc):
    cc,vc=cropRoutines.cropCoords(i,cropSize,marginSize,inChunk.shape,startDim)
    tc=targetCoords(cc,vc)
    crop=inChunk[tuple(cc)]
    o=net.forward(torch.from_numpy(crop).cuda())
    if outDims:
      ttc=[]
      tvc=[]
      for k in range(len(outDims)):
        ttc.append(slice(0,outDims[k]))
        tvc.append(slice(0,outDims[k]))
      for k in range(startDim,inChunk.ndim):
        ttc.append(tc[k])
        tvc.append(vc[k])
      tc=ttc
      vc=tvc
    outChunk[tuple(tc)]=o.cpu().data.numpy()[tuple(vc)]
  return outChunk
