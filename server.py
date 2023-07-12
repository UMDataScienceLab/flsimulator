#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
import torch
import os,time,re,tempfile,json,pickle
import math
from mip import Model, xsum, BINARY

#from replay_parser import parse_replay_syn
from torch.multiprocessing import Process,Queue
import torch.nn as nn
import torch.nn.functional as F

from util import log
import torchvision
from torchvision import transforms

from client import *
from network import *
import modeloperations as mo

def get_algorithm_class(algorithm_name):
    """Return the algorithm class with the given name."""
    if algorithm_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]

def split_dataset(dataset, params):
    how_many_clients = params['number_of_clients']
    idx_list = []

    len_ds = dataset.__len__()
    log('There are {} train datapoints'.format(len_ds),logfile=params['LOGFILE'])
    mu = len_ds / (how_many_clients)

    if params['noniid'] == 1 and params['DATASET'] in {'MNIST','CIFAR10'}:
        # if we want to use non iid partition
        idx=[]
        #test_idx = []
        if params['DATASET'] == 'MNIST':
            labels = dataset.train_labels.numpy()
            #test_labels = test_dataset.train_labels.numpy()
        else:
            labels = np.array(dataset.targets)
            #test_labels = test_dataset.targets
        for i in range(10):
            print(np.where(labels == i)[0])
            idx.append(np.where(labels == i)[0])
            #test_idx.append(np.where(test_labels == i)[0])
        for i in range(10):
            random.shuffle(idx[i])
            #random.shuffle(test_idx[i])
        start = [0 for i in range(10)]
        #test_start = [0 for i in range(10)]

        # use integer programming to determine the assignment
        c = 10
        lc = params['largest_categories']
        queens = Model()
        x = [[queens.add_var('x({},{})'.format(i, j), var_type=BINARY)
                  for j in range(c)] for i in range(how_many_clients)]
        # lc per row
        for i in range(how_many_clients):
            queens += xsum(x[i][j] for j in range(c)) == lc, 'row({})'.format(i)

        # (how_many_clients*lc)/(10) per column
        for j in range(c):
            queens += xsum(x[i][j] for i in range(how_many_clients)) == (how_many_clients*lc)/c, 'col({})'.format(j)
        queens.optimize()

        res = []
        if queens.num_solutions:
            for i, v in enumerate(queens.vars):
                if v.x >= 0.99:
                    res.append([i // c, i % c])
        else:
            raise Exception('Cannot find a proper assignment')
            #raise Exception('Assignment created')

        # print('length of res is:{}'.format(len(res)))
        dt = int(mu / lc)
        #test_dt = int(test_dataset.__len__() / (lc*how_many_clients))

        #calculate statistics about assignment
        all_assignment = set()
        for i in range(0,len(res),lc):
            to_concatenate = []
            #test_to_concatenate = []
            which_client = i//lc
            which_class = []
            for j in range(lc):
            # print(res[i*lc+j][1])
                try:
                    which_class.append(res[i+j][1])
                    st = start[res[i+j][1]]
                    #test_st = test_start[res[i + j][1]]

                except:
                    print('error at this position:')
                    print(i  + j)
                    print("i is {}, lc is {}, j is {}".format(i,lc,j))
                    print(res[i+j][1])

                to_concatenate.append(idx[res[i+j][1]][st:(st+dt)])
                #test_to_concatenate.append(test_idx[res[i+j][1]][test_st:(test_st+test_dt)])

                start[res[i+j][1]] = st + dt
                #test_start[res[i + j][1]] = test_st + test_dt

            which_class = np.sort(np.array(which_class))
            #hash_class = sum([c**i*which_class[i] for i in range(len(which_class))])
            #all_assignment.add(hash_class)
                
   
            index_i = copy.deepcopy(np.concatenate((to_concatenate)))
            idx_list.append(index_i)
            #test_index_i = copy.deepcopy(np.concatenate((test_to_concatenate)))

            #index_i = copy.deepcopy(np.concatenate((idx[slice[0]][st1:(st1+dt)],idx[slice[1]][st2:(st2+dt)])))
            #client_i = client(None, dataset, index_i, test_dataset, test_index_i, args)
            #client_list.append(client_i)
            #start[slice[0]] = st1 + dt
            #start[slice[1]] = st2 + dt
        if len(idx_list) > 5:
            print('{} clients are created. Distribution is non-iid'.format(len(idx_list)))
        else:
            raise Exception('Assignment created')


        print('{} clients are created. Distribution is non-iid'.format(len(idx_list)))
        #print('all assignments')
        #print(all_assignment)
        print('length')
        print(len(all_assignment))
        return idx_list#client_list, np.ones(len(client_list))/len(client_list), validation_client
    # if we want to use iid partition
    delta = int(np.floor(np.sqrt(12/(how_many_clients*how_many_clients+2))*mu*params['variance_ratio']))
    if mu - delta*how_many_clients*0.5<1:
        delta = int(np.floor(2*len_ds/how_many_clients**2-2/how_many_clients))
    rd_list = list(range(len_ds))
    random.shuffle(rd_list)
    step = int(np.floor(mu))-int(delta*how_many_clients/2)#int(len(dataset)/how_many_clients)
    begin = 0
    test_begin = 0
    test_dt = 0#int(test_dataset.__len__()/how_many_clients)
    for i in range(how_many_clients):
        index_i = rd_list[begin:(begin+step)]
        idx_list.append(index_i)
        begin+=step
        test_begin+=test_dt
        #impv.append(step/len_ds)
        step += delta
    #print('Data set distributed with std'+str(np.sqrt(np.var(impv))))
    #params['std'] = np.sqrt(np.var(impv))
    params['total_data'] = how_many_clients
    return idx_list

class fedavg(torch.nn.Module):
    def __init__(self, trainset, testset,params):
        super(fedavg, self).__init__()
        if params['model'] == 'Resnet':
            self.model = Resnet()
        self.trainset = trainset
        self.testset = testset
        self.params = params

    def train_model(self):
        """
            multi-processing implementation of fedavg
        """
        log("playing with %s"%(2),logfile=self.params['LOGFILE'])
        idx_list = split_dataset(self.trainset, self.params)
        device = torch.device('cuda')
        subprocess_num = self.params['subprocess_num']
        tot_client = len(idx_list)
        log("there are %s clients"%tot_client,logfile=self.params['LOGFILE'])
        data_q=Queue()
        #tot_client = 2

        testloader = torch.utils.data.DataLoader(dataset=self.testset,
                                                  batch_size=self.params['batch_size'],
                                                  shuffle=True)
        
        for numepoch in range(self.params['epochs']):
            log("[%s/%s]: training started"%(numepoch, self.params['epochs']),logfile=self.params['LOGFILE'])
            weight_list = []
            loss_list = []
            plist = []
            count = 0
            for j in range(tot_client//subprocess_num):
                plist = []
                for i in range(subprocess_num):
                    #p_local_train_0(model, device, idx, trainset, params, data_q):
                    plist.append(Process(target=p_local_train_0,args=(copy.deepcopy(self.model), device, idx_list[count], self.trainset, self.params,count, data_q)))
                    plist[i].start()
                    time.sleep(0.5)
                    log("client %s started training"%count,logfile=self.params['LOGFILE'])

                    count += 1
                for i in range(subprocess_num):
                    for nres in data_q.get(True):
                        #log(nres)
                        res = torch.load(nres)
                        weight_list.append(res['updated_model_weights'])
                        loss_list.append(res['average_training_loss'])
                for p in plist:
                    p.join(0.01)
                
            # take average
            self.model = mo.take_average(weight_list, self.model)
            self.model = self.model.to(device)
            train_loss = np.mean(np.array(loss_list)) 
            test_acc = accuracy(self, testloader, None, device)           
            log(" [%s/%s] finished, train loss %.6f, test acc %s"%(numepoch, self.params['epochs'], train_loss, test_acc),logfile=self.params['LOGFILE'])
            self.model = self.model.cpu()

        #os.system(mount_command)

    def predict(self, x): 
        return self.model(x)

class fedavgRW(torch.nn.Module):
    def __init__(self, trainset, testset,params):
        super(fedavgRW, self).__init__()
        if params['model'] == 'Resnet':
            self.model = Resnet()
        self.trainset = trainset
        self.testset = testset
        self.params = params

    def train_model(self):
        """
            multi-processing implementation of fedavg random walk
        """
        log("playing with %s"%(2),logfile=self.params['LOGFILE'])
        idx_list = split_dataset(self.trainset, self.params)
        device = torch.device('cuda')
        subprocess_num = 1#self.params['subprocess_num']
        tot_client = len(idx_list)
        log("there are %s clients"%tot_client,logfile=self.params['LOGFILE'])
        data_q=Queue()
        current_point = int(np.random.rand(1)[0]*tot_client)
        #tot_client = 2

        testloader = torch.utils.data.DataLoader(dataset=self.testset,
                                                  batch_size=self.params['batch_size'],
                                                  shuffle=True)
        
        for numepoch in range(self.params['epochs']):
            log("[%s/%s]: random walk started"%(numepoch, self.params['epochs']),logfile=self.params['LOGFILE'])
            weight_list = []
            loss_list = []
            plist = []
            count = 0
            
            plist = []
            for i in range(subprocess_num):
                    #p_local_train_0(model, device, idx, trainset, params, data_q):
                plist.append(Process(target=p_local_train_0,args=(copy.deepcopy(self.model), device, idx_list[current_point], self.trainset, self.params,count, data_q)))
                plist[i].start()
                time.sleep(0.5)
                log("client %s started training"%current_point,logfile=self.params['LOGFILE'])

                count += 1
            for i in range(subprocess_num):
                for nres in data_q.get(True):
                        #log(nres)
                    res = torch.load(nres)
                    weight_list.append(res['updated_model_weights'])
                    loss_list.append(res['average_training_loss'])
            for p in plist:
                p.join(0.01)
                
            # take average
            self.model = mo.take_average(weight_list, self.model)
            self.model = self.model.to(device)
            train_loss = np.mean(np.array(loss_list)) 
            test_acc = accuracy(self, testloader, None, device)           
            log(" [%s/%s] finished, train loss %.6f, test acc %s"%(numepoch, self.params['epochs'], train_loss, test_acc),logfile=self.params['LOGFILE'])
            self.model = self.model.cpu()
            current_point += np.random.choice(2)*2-1
            current_point = current_point % tot_client

        #os.system(mount_command)

    def predict(self, x): 
        return self.model(x)

class fedensembleRW(torch.nn.Module):
    def __init__(self, trainset, testset,params):
        super(fedensembleRW, self).__init__()
        if params['model'] == 'Resnet':
            self.model_list = nn.ModuleList([Resnet() for i in range(params['K'])])
        #self.model_list = model_list
        self.trainset = trainset
        self.testset = testset
        self.params = params

    def train_model(self):
        """
            multi-processing implementation of fedensemble random walk
        """
        log("playing with %s"%(2),logfile=self.params['LOGFILE'])
        idx_list = split_dataset(self.trainset, self.params)
        device = torch.device('cuda')
        subprocess_num = self.params['subprocess_num']
        tot_client = len(idx_list)
        log("there are %s clients"%tot_client,logfile=self.params['LOGFILE'])
        data_q=Queue()
        
        #tot_client = 2
        model_pos = np.random.choice(tot_client, self.params['K'])
        testloader = torch.utils.data.DataLoader(dataset=self.testset,
                                                  batch_size=self.params['batch_size'],
                                                  shuffle=True)
        
        for numepoch in range(self.params['epochs']):
            log("[%s/%s]: training started"%(numepoch, self.params['epochs']),logfile=self.params['LOGFILE'])
            weight_list = []
            loss_list = []
            plist = []
            which_model_list = []
            count = 0
            for j in range(math.ceil(self.params['K']/subprocess_num)):
                plist = []
                for i in range(min(subprocess_num,self.params['K'] - subprocess_num*(j))):
                    #p_local_train_0(model, device, idx, trainset, params, data_q):
                    plist.append(Process(target=p_local_train_0,args=(copy.deepcopy(self.model_list[count]), device, idx_list[model_pos[count]], self.trainset, self.params,count, data_q)))
                    plist[i].start()
                    time.sleep(0.5)
                    log("client %s started training with model %s "%(model_pos[count], count),logfile=self.params['LOGFILE'])

                    count += 1
                for i in range(min(subprocess_num,self.params['K'] - subprocess_num*(j))):
                    for nres in data_q.get(True):
                        #log(nres)
                        res = torch.load(nres)
                        weight_list.append(res['updated_model_weights'])
                        loss_list.append(res['average_training_loss'])
                        which_model_list.append(res['count'])

                for p in plist:
                    p.join(0.01)

            model2pos = {which_model_list[ii]:ii for ii in range(len(which_model_list))}
            # take average
            for k in range(self.params['K']):
                self.model_list[k] = mo.take_average([weight_list[model2pos[k]]], self.model_list[k])
            self.model_list = self.model_list.to(device)
            train_loss = np.mean(np.array(loss_list)) 
            log("evaluating model...",logfile=self.params['LOGFILE'])
            test_acc = accuracy(self, testloader, None, device)           
            log(" [%s/%s] finished, train loss %.6f, test acc %s"%(numepoch, self.params['epochs'], train_loss, test_acc),logfile=self.params['LOGFILE'])
            self.model_list = self.model_list.cpu()
            for k in range(self.params['K']):
                model_pos[k] += np.random.choice(2)*2-1
                model_pos[k] = model_pos[k]%tot_client


        #os.system(mount_command)

    def predict(self, x):
        return sum(self.model_list[k](x) for k in range(self.params['K']) ) 
        #return self.model(x)

class fedensemble(torch.nn.Module):
    def __init__(self, trainset, testset,params):
        super(fedensemble, self).__init__()
        if params['model'] == 'Resnet':
            self.model_list = nn.ModuleList([Resnet() for i in range(params['K'])])
        #self.model_list = model_list
        self.trainset = trainset
        self.testset = testset
        self.params = params

    def train_model(self):
        """
            multi-processing implementation of fedensemble
        """
        #log("playing with %s"%(2),logfile=self.params['LOGFILE'])
        idx_list = split_dataset(self.trainset, self.params)
        device = torch.device('cuda')
        subprocess_num = self.params['subprocess_num']
        tot_client = len(idx_list)
        log("there are %s clients"%tot_client,logfile=self.params['LOGFILE'])
        data_q=Queue()
        
        #tot_client = 2
        #model_pos = np.random.choice(tot_client, self.params['K'])
        testloader = torch.utils.data.DataLoader(dataset=self.testset,
                                                  batch_size=self.params['batch_size'],
                                                  shuffle=True)
        
        for numepoch in range(self.params['epochs']):
            log("[%s/%s]: training started"%(numepoch, self.params['epochs']),logfile=self.params['LOGFILE'])
            weight_list = []
            loss_list = []
            plist = []
            which_model_list = []
            tempfilenames = []
            if numepoch % self.params['K'] == 0:
                training_schedule = []
                for nc in range(tot_client):
                    tsc = list(range(self.params['K']))
                    random.shuffle(tsc)
                    training_schedule.append(tsc)
            count = 0
            for j in range(tot_client//subprocess_num):
                plist = []
                for i in range(subprocess_num):
                    #p_local_train_0(model, device, idx, trainset, params, data_q):
                    plist.append(Process(target=p_local_train_0,args=(copy.deepcopy(self.model_list[training_schedule[count][numepoch%self.params['K']]]), device, idx_list[count], self.trainset, self.params,count, data_q)))
                    plist[i].start()
                    time.sleep(0.5)
                    log("client %s started training with model %s "%(training_schedule[count][numepoch%self.params['K']], count),logfile=self.params['LOGFILE'])

                    count += 1

                for i in range(subprocess_num):
                    for nres in data_q.get(True):
                        #log(nres)
                        tempfilenames.append(nres)
                        res = torch.load(nres)
                        weight_list.append(res['updated_model_weights'])
                        loss_list.append(res['average_training_loss'])
                        which_model_list.append(res['count'])

                for p in plist:
                    p.join(0.01)
            
            count2pos = {which_model_list[ii]:ii for ii in range(len(which_model_list))}

            # take average

            for k in range(self.params['K']):
                m2pk = []
                for j in range(tot_client):
                    if training_schedule[j][numepoch%self.params['K']] == k:
                        m2pk.append(count2pos[j])
                modeliupdated = [weight_list[ii] for ii in m2pk]
                self.model_list[k] = mo.take_average(modeliupdated, self.model_list[k])
            #for k in range(self.params['K']):
            #    self.model_list[k] = mo.take_average([weight_list[model2pos[k]]], self.model_list[k])
            self.model_list = self.model_list.to(device)
            train_loss = np.mean(np.array(loss_list)) 
            log("evaluating model...",logfile=self.params['LOGFILE'])
            test_acc = accuracy(self, testloader, None, device)           
            log(" [%s/%s] finished, train loss %.6f, test acc %s"%(numepoch, self.params['epochs'], train_loss, test_acc),logfile=self.params['LOGFILE'])
            self.model_list = self.model_list.cpu()

            for dfile in tempfilenames:
                os.system('rm %s'%dfile)
            #for k in range(self.params['K']):
            #    model_pos[k] += np.random.choice(2)*2-1
            #    model_pos[k] = model_pos[k]%tot_client


        #os.system(mount_command)

    def predict(self, x):
        return sum(self.model_list[k](x) for k in range(self.params['K']) ) 
        #return self.model(x)


def accuracy(network, loader, weights, device):
    correct = 0
    total = 0
    weights_offset = 0

    network.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            p = network.predict(x)
            if weights is None:
                batch_weights = torch.ones(len(x))
            else:
                batch_weights = weights[weights_offset : weights_offset + len(x)]
                weights_offset += len(x)
            batch_weights = batch_weights.to(device)
            if p.size(1) == 1:
                correct += (p.gt(0).eq(y).float() * batch_weights.view(-1, 1)).sum().item()
            else:
                correct += (p.argmax(1).eq(y).float() * batch_weights).sum().item()
            total += batch_weights.sum().item()
    network.train()
    return correct / total

