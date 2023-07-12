import pickle
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from sklearn.decomposition import PCA
import copy

def mul(net1, net2):
    result = pickle.loads(pickle.dumps(net1))
    net1_dict = net1.state_dict()
    net2_dict = net2.state_dict()
    weight_keys = list(net1_dict.keys())
    result_dict = result.state_dict()
    for key in weight_keys:
        result_dict[key] = net1_dict[key] * net2_dict[key]
    result.load_state_dict(result_dict)
    return result

def take_average(nets_list, model):    
    result_dict = model.state_dict()   
    N = len(nets_list) 
    with torch.no_grad():        
        #nets_dict = [neti.state_dict() for neti in net_list]
        weight_keys = list(result_dict.keys())
        for key in weight_keys:
            result_dict[key] *= 0
            for j in range(N):
                result_dict[key] = result_dict[key] + 1/N*nets_list[j][key]
        model.load_state_dict(result_dict)
        return model

def scalar_mul(scalar_list, net_list, rg=False):
    result = copy.deepcopy(net_list[0])
    with torch.no_grad():
        result_dict = result.state_dict()
        nets_dict = [neti.state_dict() for neti in net_list]
        weight_keys = list(result_dict.keys())
        for key in weight_keys:
            result_dict[key] *= 0
            for j in range(len(scalar_list)):
                result_dict[key] = result_dict[key] + scalar_list[j]*nets_dict[j][key]
        result.load_state_dict(result_dict)
        return result
        
def scalar_mul_no_buffer(scalar_list, net_list, rg=False):
    result = copy.deepcopy(net_list[0])#pickle.loads(pickle.dumps(net_list[0]))
    with torch.no_grad():
        worker_params = [list(x.parameters()) for x in net_list]
        for i, params in enumerate(result.parameters()):
            params.data = 0 * params.data
            for j in range(len(scalar_list)):
                params.data = params.data + worker_params[j][i] * scalar_list[j]
        return result



def inner_p(net1, net2, rg=False):
    result = torch.zeros(1)[0]
    if rg:
        result.requires_grad = True
    if next(net1.parameters()).is_cuda:
        #print('something is on cuda')
        result = result.cuda()
    net1_params = list(net1.parameters())
    net2_params = list(net2.parameters())

    for i in range(len(net1_params)):
        result = result + (net1_params[i]*net2_params[i]).sum()
    #print("inner product: {}".format(result))
    #print(result)
    return result


def net_flatten(net):
    newnet = pickle.loads(pickle.dumps(net))
    n_dict = newnet.state_dict()
    w_keys = list(n_dict.keys())
    res = torch.flatten(n_dict[w_keys[0]])
    for i in range(1,len(w_keys)):
        res = torch.cat((res,torch.flatten(n_dict[w_keys[0]])))
    return res


def gradient_flatten(net):
    res = torch.tensor([])#torch.flatten(n_dict[w_keys[0]])
    for p in net.parameters():
        res = torch.cat((res,torch.flatten(p.grad.data)))
    return res


def reconstruct_gradient(net):
    res = copy.deepcopy(net)
    with torch.no_grad():
        for (p1,p2) in zip(res.parameters(), net.parameters()):
            p1.data *= 0
            p1.data += p2.grad.data
    return res


def reset(model):
    for layer in model.children():
        if hasattr(layer, "reset_parameters"):
            layer.reset_parameters()

def gaussian_sampling(dwt, model):
    coefs = torch.ones(len(dwt))/len(dwt)
    average_dw = scalar_mul(coefs, dwt)
    centralized_dw = [scalar_mul([1, -1], [dw, average_dw]) for dw in dwt]
    var = torch.tensor([inner_p(cdw, cdw) for cdw in centralized_dw])
    std = torch.sqrt(var.mean())
    deltaw = copy.deepcopy(model)
    reset(deltaw)
    norm = inner_p(deltaw, deltaw)
    print("norm is {}".format(norm))
    z = torch.randn(1).item()
    #z = 0
    return scalar_mul([1,1,std/torch.sqrt(norm)*z], [model,average_dw, deltaw])


def add(net1, net2):
    result = pickle.loads(pickle.dumps(net1))
    net1_dict = net1.state_dict()
    net2_dict = net2.state_dict()
    weight_keys = list(net1_dict.keys())
    result_dict = result.state_dict()
    for key in weight_keys:
        result_dict[key] = net1_dict[key] + net2_dict[key]
    result.load_state_dict(result_dict)
    return result


def gradient_l2_norm(model):
    norm = 0
    for p in model.parameters():
        norm += p.grad.data.norm(2).item() ** 2
    return norm

def gradient_norm_lw(model):
    res = []
    for p in model.parameters():
        res.append(p.grad.data.norm(2).item())
    return res

def model_std(model_list):
    new_list = pickle.loads(pickle.dumps(model_list))
    wt = np.zeros(len(new_list))
    center_of_mass = scalar_mul(wt,new_list)
    var = 0
    for i in new_list:
        diff = scalar_mul([1,-1],[i,center_of_mass])
        var = var + inner_p(diff,diff)
    return torch.sqrt(var/(len(model_list)-1))


def grad_over_model(numerator, denominator):
    result = pickle.loads(pickle.dumps(denominator))
    #denominator.zero_grad()
    numerator.backward()
    res_params = list(result.parameters())
    deno_params = list(denominator.parameters())
    for i in range(len(res_params)):
        res_params[i] = deno_params[i].grad
    result.zero_grad()
    denominator.zero_grad()
    return result



if __name__ == '__main__':
    from network import CNNMnist, Twohiddenlayerfc
    ml = []
    for i in range(10):
        ml.append(Twohiddenlayerfc())
    #ms = model_star(ml)
    import copy
    m11=copy.deepcopy(ml[0])
    res = m11(torch.randn(10))
    res = res.sum()
    res.backward()
    ms = weight_params_pca(ml,7)
    print('eigen values: {}'.format(ms))
