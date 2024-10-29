import numpy as np
import torch
import torch.nn as nn
np.random.seed(131)
torch.manual_seed(131)

def get_loss(gamma1,gamma2,prob_ini,prob_all_s,prob_all_h,prob_all_y):
    t1=torch.sum(torch.exp(gamma1)*(prob_ini+prob_all_h[:,0]+prob_all_y[:,0][:,None]))
    t2=0
    for i in range(1,prob_all_s.shape[1]):
        t2=t2+torch.sum(torch.exp(gamma2[:,i-1])*(prob_all_s[:,i]+prob_all_h[:,i][:,None]+prob_all_y[:,i][:,None,None]))
    return t1,t2

def get_cross_entropy(pos,pri):
    en=0
    for k in range(pos.shape[0]):
        for i in range(pos.shape[1]):
            en=en+pos[k,i]*pri[k,i]
    return en.sum()