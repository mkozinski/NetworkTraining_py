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
  betti_true = compute_betti(y_true)
  betti_pred = compute_betti(y_pred)

  return np.abs(betti_true-betti_pred)

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

    betti_topo = betti_error_topo(img_true, img_predicted)
    cldice = cldDice(img_predicted, img_true)   

    betti_diff = abs(betti_number(img_true, img_predicted)) 
    
    return accuracy, recall, dice , betti_diff, betti_topo, cldice