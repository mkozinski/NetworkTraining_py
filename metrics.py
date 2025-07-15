import numpy as np
from ripser import lower_star_img
import gudhi as gd
from driveMaximin.cl_metric import cldDice

def compute_betti(patch: np.array):
    cc = gd.CubicalComplex(top_dimensional_cells=1-patch)
    cc.compute_persistence()
    bnum = cc.persistent_betti_numbers(np.inf, -np.inf)[0]
    del cc
    return bnum

def betti_error_topo(y_true, y_pred):
  diff_tmp = 0
  #  ASSUMES SQUARE IMAGE
  psize = 64
  for i in np.random.uniform(0,y_true.shape[0]-psize-1,100):
     i=int(i)
     patch_true = 1*y_true[i:i+psize,i:i+psize]
     patch_pred = 1*y_pred[i:i+psize,i:i+psize]
     diff_tmp += np.abs(compute_betti(patch_true)-compute_betti(patch_pred))
  return diff_tmp/100.0 

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

    betti_topo = betti_error_topo(img_true, img_predicted)
    cldice = cldDice(img_predicted, img_true)    
    
    return accuracy, recall, dice , betti_dif/100, betti_topo, cldice