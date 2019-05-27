import numpy as np

def crop(img,inds,fill=0):
    # crop where the range of indexes can contain negative values
    # that is, we can request a crop that goes outside of the image frame
    # this means, negatice indeces to the left (top) of the image
    # and indeces larger than the image size to the right (bottom)
    # inds is a list of slice objects
    # fill is the value inpainted in the regions outside of the image frame
    outsize=[]
    dstinds=[]
    srcinds=[]
    for k in range(len(inds)):
        srcbeg=max(0,inds[k].start)
        srcend=min(img.shape[k],inds[k].stop)
        dstbeg=max(0,-inds[k].start)
        dstend=dstbeg+srcend-srcbeg
        dstinds.append(slice(dstbeg,dstend))
        srcinds.append(slice(srcbeg,srcend))
        outsize.append(inds[k].stop-inds[k].start)
    crop=np.full(tuple(outsize),fill,img.dtype)
    crop[tuple(dstinds)]=img[tuple(srcinds)]
    return crop, dstinds

