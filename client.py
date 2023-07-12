#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
import torch
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
import os,time,re,tempfile,json, pickle
import copy
#from replay_parser import parse_replay_syn
#from torch.multiprocessing import Process,Queue
from util import log

 
class local_dataset(torch.utils.data.Dataset):
    def __init__(self, wholedata, indices):
        self.all_data = wholedata
        self.indices = indices

    def __getitem__(self, index):
        img, label = self.all_data[self.indices[index]]
        return img, label

    def __len__(self):
        return len(self.indices)

    def some_function(self):
        pass


def p_local_train_0(model, device, idx, trainset_full, params, count, data_q):
    '''
    implement one local trainning for each algorithm
    simplest version
    fedavg and fedensemble
    '''
    #log("client %s started preparing"%count)

    trainset = local_dataset(trainset_full,idx)
    #log("client %s created dataset"%count)

    the_model = model#copy.deepcopy(model)
    #log("client %s copied model"%count)

    if params['local_optim'] == 'adam':
        optimizer = torch.optim.Adam(the_model.parameters(), lr=params['initial_lr'])
    elif params['local_optim'] == 'sgd':
            optimizer = torch.optim.SGD(the_model.parameters(), momentum=params['momentum'], lr=params['initial_lr'], weight_decay=params['weight_decay'])
    else:
        raise Exception('No such optimizer: '+params['local_optim'])
    the_model = the_model.to(device)
    the_model.train()
        #self.model.train()
        
    #log("client %s moved model to gpu"%count)
    
    data_loader = torch.utils.data.DataLoader(dataset=trainset,
                                                  batch_size=params['batch_size'],
                                                  shuffle=True)
        #criterion = nn.CrossEntropyLoss()
        #the_model = self.model.to(device)
        #sample_intervals = self.dataset.__len__()//optimizer_method['number_of_samples_sent_back']

    #log("client %s prepared for training"%count)
    for j in range(params['local_epoch']):
        tot_loss = 0
        nos = 0
           
        for i, (images, labels) in enumerate(data_loader):
            images = images.to(device)
            labels = labels.to(device)

            output = the_model(images)
            local_loss = nn.CrossEntropyLoss()(output, labels)
                      
            optimizer.zero_grad()
            local_loss.backward()
            optimizer.step()
            tot_loss += local_loss.item()
            nos += len(labels)

    log("Client %s finished training"% count,logfile=params['LOGFILE'])
    res = {
            'updated_model_weights':the_model.cpu().state_dict(),
            'average_training_loss':tot_loss/nos,
            'count':count
    } 
    PATH = params['prefix']+"%s.pkl"% (count)       
    torch.save(res, PATH)
    data_q.put([PATH])

