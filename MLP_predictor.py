from __future__ import print_function, division

import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
from skimage import io, transform
from torch.autograd import Variable
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader

from collections import OrderedDict

import os
import math
import scipy as sp
import scipy.stats
import logging
import copy
import time
import argparse
 
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score, hamming_loss, roc_curve, auc, f1_score
from sklearn.metrics import average_precision_score
from scipy import interp

from utils._utils import *
from utils.metrics import MultiLabelMaskClassificationMetrics


 
torch.manual_seed(0)
np.random.seed(0)
 
import warnings

warnings.filterwarnings("ignore")


  
class BCEMASKEDLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__()
        
        self.BCE_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.gamma  = torch.tensor([gamma])
        self.thresh = torch.tensor([0.5])
        
    def forward(self, input, target):
        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})"
                             .format(target.size(), input.size()))
        
        # targets are represented as positive: 1, negative -1 and unknown 0

        mask =  target**2 # make all known activity values 1 and create a mask
        tar  = (target + mask)/2.0 # make a target matrix with all known actives
        
        input_masked  = input[mask > 0] # Select all known compound activity readouts, mask
        target_masked = tar[mask > 0]   # Select all known compound activity readouts, target

        basic_loss = self.BCE_loss(input_masked, target_masked)
        sig_out = torch.sigmoid(input_masked)

        focal_part = torch.pow(target_masked-sig_out, self.gamma)
        loss = (focal_part*basic_loss).mean()        

        return loss
 

class MLP(nn.Module):

    def __init__(self, input_size, hidden_size, num_hidden, num_classes, drop_prob=0.5):

        super(MLP, self).__init__()                   

        components_ = OrderedDict()

        components_["layer_1"] = nn.Linear(input_size, hidden_size) 
        components_["relu_1"]  = nn.ReLU()  
        components_["drop_1"]  = nn.Dropout(p=drop_prob)  


        for i in range(1, num_hidden - 1):

            components_["layer_"+str(i+1)] = nn.Linear(hidden_size, hidden_size) 
            components_["relu_" +str(i+1)] = nn.ReLU()  
            components_["drop_" +str(i+1)] = nn.Dropout(p=drop_prob)  

        components_["layer_out"] = nn.Linear(hidden_size, num_classes) 

        self.mlp_net = nn.Sequential(components_)

       

    def forward(self, x):                        

        return self.mlp_net(x)





def sigmoid(X):

    return 1/(1+np.exp(-X))



class VectorData(Dataset):

    """Dataset"""
    def __init__(self, dataset_info_splits, features,  assays, mode = "train", split_numbers = {"train":[0,1,2,3], "val":[4], "test":[5]}):

        self.mode = mode
        self.num_classes = len(assays)
        self.well_info      = pd.read_csv(dataset_info_splits)
        self.features       = pd.read_csv(features, index_col=0)
        self.data, self.int_to_labels, self.labels_to_int = self.get_data_as_list(dataset_info_splits, split_numbers=split_numbers, assays=assays,  un_identifier="Metadata_InChI", cmpd_identifier="Metadata_InChI")
        

    def get_data_as_list(self, dataset_info_splits, split_numbers, assays, un_identifier="Metadata_InChI", cmpd_identifier="Metadata_InChI"):
        
        assays = assays
        df = pd.read_csv(dataset_info_splits, index_col=0)
        
        int_to_labels = {i: x for i,x in enumerate(assays)}
        labels_to_int = {val: key for key, val in int_to_labels.items()}

        if self.mode == 'train':
            data = df[df.split_number.isin(split_numbers["train"])]
        elif self.mode in ['val', 'eval']:
            data = df[df.split_number.isin(split_numbers["val"])]
        else:
            data = df[df.split_number.isin(split_numbers["test"])]

        data = data.drop_duplicates(un_identifier)

        self.df = data

        labels    = data[assays].values.tolist()
        cmpds     = data[cmpd_identifier].values.tolist()
        
        data_list = [{'cmpds': cmp, 'label': label} for cmp, label in zip(cmpds, labels)]
                    
        return data_list, int_to_labels, labels_to_int

    def get_features(self, cmpd):
        return self.features.loc[cmpd].values

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        if torch.is_tensor(idx):
            idx = idx.tolist()

        cmpd    = self.data[idx]['cmpds']
        label   = torch.as_tensor(self.data[idx]['label'])
        feature = torch.as_tensor(self.get_features(cmpd))
        
        sample = {'features': feature, 'label': label, "cmpd": cmpd}
        
        return sample

 


class Predictor:

   

    def __init__(self, data_path, feature_path, hyper_params, settings, is_chem=True):
        
        if is_chem:
            train_data = VectorData(data_path, feature_path, split_numbers=settings["split_numbers"], mode="train")
            val_data   = VectorData(data_path, feature_path, split_numbers=settings["split_numbers"], mode="val")
            test_data  = VectorData(data_path, feature_path, split_numbers=settings["split_numbers"], mode="test")
        else:
            print("Cell Features")

        self.train_loader = DataLoader(train_data, batch_size=hyper_params['batch_size'], shuffle=True, num_workers=1)
        self.val_loader   = DataLoader(val_data,   batch_size=hyper_params['batch_size'], shuffle=False, num_workers=1)
        self.test_loader  = DataLoader(test_data,  batch_size=hyper_params['batch_size'], shuffle=False, num_workers=1)

        self.mlp = MLP(hyper_params['input_size'], hyper_params['hidden_size'], hyper_params['num_hidden'], train_data.num_classes)

        self.deep_copy()
 
        self.metric_train = MultiLabelMaskClassificationMetrics(train_data.num_classes, train_data.int_to_labels, mode="train")
        self.metric_val   = MultiLabelMaskClassificationMetrics(train_data.num_classes, train_data.int_to_labels, mode="val")
        self.metric_test  = MultiLabelMaskClassificationMetrics(train_data.num_classes, train_data.int_to_labels, mode="test")
        self.criterion    = BCEMASKEDLoss()
        if hyper_params['optimizer'] == "SGD": 
            self.optimizer    = torch.optim.SGD(self.mlp.parameters(), lr=hyper_params['learning_rate'], weight_decay=hyper_params['weight_decay'])
        else:
            self.optimizer    = torch.optim.Adam(self.mlp.parameters(), lr=hyper_params['learning_rate'], weight_decay=hyper_params['weight_decay'])

        self.num_epochs = hyper_params['num_epochs']

        self.current_epoch = 0
        self.loss_array = []
        self.loss_test  = []

   

    def deep_copy(self):
        self.best_model = copy.deepcopy(self.mlp)

    def train(self):

        start_epoch = self.current_epoch
        best_val_roc_auc = 0.0
        patience = 0
        for epoch in range(self.current_epoch,start_epoch+self.num_epochs):
            
            self.mlp.train()
            for i, batch in enumerate(self.train_loader):  

                features = batch['features'].float()
                labels   = batch['label'].float()
                
                self.optimizer.zero_grad()                         
                outputs = self.mlp(features)                       
                loss    = self.criterion(outputs, labels)          
                
                loss.backward()          
                self.optimizer.step()    

                self.metric_train.add_preds(outputs, labels)
                if (i+1) % 5 == 0:                

                    print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'

                         %(epoch+1, start_epoch+self.num_epochs, i+1, len(self.train_loader), loss.item()))
                    print(self.metric_train.get_value())
                    self.metric_train.reset()
                    break

                self.loss_array.append(loss.item())



            if epoch % 1 == 0 and epoch > 0:
                self.mlp.eval()
                with torch.no_grad():
                    test_loss_avg = []
                    for j, batch in enumerate(self.val_loader):   

                        features = batch['features'].float()
                        labels   = batch['label'].float()

                        outputs = self.mlp(features)
                        loss = self.criterion(outputs, labels)       

                        test_loss_avg.append(loss.item())
                        self.metric_val.add_preds(outputs, labels)
                    self.loss_test.append(np.array(test_loss_avg).mean())
                    print(self.loss_test[-1])
                    val_roc_auc, res_str = self.metric_val.get_value(return_value=True)
                    print(res_str)
                    self.metric_val.reset()

                    if val_roc_auc > best_val_roc_auc:
                        print('best yet')
                        self.deep_copy()
                        best_val_roc_auc = val_roc_auc
                        patience = 0
                    else:
                        patience += 1

            if patience > 8 and epoch > 0:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = param_group['lr']*0.5
                    print("New learning rate: ", param_group['lr'])
                    patience = 0

            self.current_epoch += 1

           

    def test(self,current=False,val=False):

        if current:
            self.test_mlp = self.mlp
        else:
            self.test_mlp = self.best_model

        if val:
            self.loader = self.val_loader
        else:
            self.loader = self.test_loader

        predictions = []
        labels_l    = []
        compound_l  = []

        self.test_mlp.eval()
        
        with torch.no_grad():
            for i, batch in enumerate(self.loader): 

                features   = batch['features'].float()
                labels     = batch['label'].float()
                cmpds      = batch['cmpd']

                outputs = self.test_mlp(features)

                predictions += outputs.tolist()
                labels_l    += labels.tolist()
                compound_l  += cmpds

        perf  = self.metric_test.evaluate_predictions(predictions, labels_l, compound_l)

        return perf




# Define params and paths below
# This assumes that the training file is prepared and the the features have been prepared

data_path    = "/path/to/metadata/with_splits_ids_and_labels.csv" 
feature_path = "/path/to/features.csv"



hyper_params = {'input_size':    1024,
                'hidden_size':    512,   
                'num_hidden':       3,
                'num_epochs':     200,    
                'batch_size':      64,    
                'learning_rate':  2.0,
                'weight_decay':   0.0 
}
settings = {"split_numbers":{"train":[0,1,2,3], "val":[4], "test":[5]}}

pred = Predictor(data_path, feature_path, hyper_params, settings)

pred.train()
perf = pred.test()


