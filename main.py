import torch
import torch.nn as nn
import torch.nn.functional as F
#import torch.optim as optim
import numpy as np
import random
import time
from server import *

if __name__=='__main__':
    torch.multiprocessing.set_start_method('spawn')


    params = {
        'DATASET': 'CIFAR10', 
        'local_optim': 'sgd', 
        'initial_lr': 0.005, 
        'weight_decay': 1e-06, 
        'momentum': 0.9, 
        'local_epoch': 10, 
        'batch_size': 32, 
        'epochs': 500, 
        'number_of_clients': 10, 
        'subprocess_num': 2, # or 1 or more, depending on your momory
        'noniid': 0, 
        'largest_categories': 5, 
        'K': 5, 
        'variance_ratio': 0.2, 
        'prefix': 'fedtemp/', 
        'buffersize': '5G', 
        'algorithm': 'fedensemble', #'fedavg'
        'seed': 0, 
        'model': 'Resnet', 
    }


    random.seed(params['seed'])
    np.random.seed(params['seed'])
    torch.manual_seed(params['seed'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    jour = "logs/"+time.strftime("%Y-%m-%d-jour-%H-%M-%S", time.localtime())+'.log'
    global LOGFILE
    LOGFILE = jour
    params['LOGFILE'] = LOGFILE

    log("training started!",logfile=LOGFILE)
    log("hyper parameters:",logfile=LOGFILE)
    log(params,logfile=LOGFILE)
    # load dataset
    if params['DATASET'] == 'CIFAR10':
        transform = transforms.Compose([
            transforms.Pad(4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        trainset = torchvision.datasets.CIFAR10(root='data', train=True,
                                                download=True, transform=transform)

        testset = torchvision.datasets.CIFAR10(root='data', train=False,
                                               download=True, transform=transform)
        #print(trainset)
    algorithm_class = get_algorithm_class(params['algorithm'])
    algorithm = algorithm_class(trainset, testset, params)
    algorithm.train_model()