import torch
import numpy as np
from scipy import ndimage
from sklearn import metrics
from easydict import EasyDict as edict
#import math
from ._utils import *

def get_activation(is_multiclass):
    if is_multiclass:
        act = nn.Softmax(dim=1)
    else:
        act = nn.Sigmoid()
    return act


def sigmoid(x):
  return 1 / (1 + np.exp(-x))


def mean_roc_auc(truths, predictions, do_sigmoid = True):
    """
    Calculating mean ROC-AUC:
        Asuming that the last dimension represent the classes
    """
    
    _truths      = np.array(deepcopy(truths))
    _predictions = np.array(deepcopy(predictions))  
    n_classes    = _predictions.shape[-1]

    if do_sigmoid:
        _predictions = sigmoid(_predictions)

    avg_roc_auc = 0 
    with_known  = 0
    aucs_array  = []
    for class_num in range(n_classes):
        
        
        tar  = (_truths[:,class_num] + _truths[:,class_num]**2 ) / 2
        mask =  _truths[:,class_num]**2 > 0

        if tar.sum() > 0 and (mask.sum() - tar.sum()) > 0: 
            
            auc = metrics.roc_auc_score(tar[mask], _predictions[:,class_num][mask])        
            avg_roc_auc += auc 
            with_known  += 1
            aucs_array.append(auc)
        else:
            print("Missing sufficent number of samples") # Need at least one positive and negative example to calculate ROC-AUC
    
    
    print("KNOWN: ", with_known, len(_truths), avg_roc_auc / (with_known+1e-8))
    
    return avg_roc_auc / (with_known+1e-8), aucs_array



class MultiLabelMaskClassificationMetrics:
    
    def __init__(self, n_classes, int_to_labels=None, act_threshold=0.5, mode=""):
        self.mode = mode
        self.prefix = ""
        if mode:
            self.prefix = mode + "_"       
        self.n_classes = n_classes
        if int_to_labels is None:
            int_to_labels = {val:'class_'+str(val) for val in range(n_classes)}
        self.labels = np.arange(n_classes)
        self.int_to_labels = int_to_labels
        self.truths      = np.empty((0,n_classes), int)
        self.predictions = np.empty((0,n_classes), float)
        self.activation = nn.Sigmoid()
        
        
    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))
        self.truths      = np.empty((0,self.n_classes), int)
        self.predictions = np.empty((0,self.n_classes), float)
    
    def add_preds(self, y_pred, y_true, using_knn=False):    
        y_true = y_true.detach().cpu().numpy()
        y_pred = y_pred.detach().cpu().numpy()
        self.truths      = np.vstack([self.truths, y_true])
        self.predictions = np.vstack([self.predictions, y_pred])
    
    def threshold_preds(self, preds):
        if isinstance(preds, torch.Tensor):
            _preds = preds.clone() > self.act_threshold
            return _preds.int().detach().cpu().numpy()
        elif isinstance(preds, np.ndarray):
            _preds = np.copy(preds) > self.act_threshold
            return _preds * 1            
        else:
            _preds = deepcopy(preds) > self.act_threshold
            return _preds * 1
    
    def evaluate_predictions(self, predictions, labels_l, compound_l):
        
        assays = ["assay_"+str(i) for i in range(len(predictions[0]))]
        preds = pd.DataFrame(predictions, columns=assays)
        preds["compound"] = compound_l
        preds[assays] = sigmoid(preds[assays].values)
        preds_combined = preds.groupby("compound").mean()

        labs = pd.DataFrame(labels_l, columns=assays)
        labs["compound"]  = compound_l
        labs_combined = labs.groupby("compound").mean()

        roc_auc, roc_auc_array = mean_roc_auc(labs_combined.values, preds_combined.values, do_sigmoid=False)        

        print(roc_auc, roc_auc_array)

        return roc_auc, roc_auc_array, labs_combined, preds_combined

    # Calculate and report metrics
    def get_value(self, use_dist=False, return_value=False):
        if use_dist:
            synchronize()
            truths      = np.array(sum(dist_gather(self.truths), []))
            predictions = np.array(sum(dist_gather(self.predictions), []))
        else:
            truths      = np.array(self.truths)
            predictions = np.array(self.predictions) 
               
                            
        roc_auc, roc_auc_array = mean_roc_auc(truths, predictions)        
        
        if return_value:
                
            return roc_auc,  edict({  self.prefix + "roc_auc" : round(roc_auc, 3),
                        self.prefix + "roc_auc_array" : wandb.Histogram(roc_auc_array)}
                    )   
            
        return edict({  self.prefix + "roc_auc" : round(roc_auc, 3),
                        self.prefix + "roc_auc_array" : wandb.Histogram(roc_auc_array)}
                    )   
        