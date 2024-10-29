import numpy as np
import torch
import torch.nn as nn
np.random.seed(131)
torch.manual_seed(131)

def dis_forward_pass(prob_all_s,prob_initial,prob_all_h,prob_all_y):
    T=prob_all_s.shape[1]
    N=prob_all_s.shape[2]
    forward_prob=torch.ones((prob_all_s[:,:,:,0].shape))
    for j in range(T):
        if j==0:
            forward_prob_each=prob_initial+prob_all_h[:,j]+prob_all_y[:,j,None]
            forward_prob[:,j]=forward_prob_each-torch.logsumexp(forward_prob_each,1)[:,None]
        else:
            np_=prob_all_s[:,j,:]+forward_prob[:,j-1][:,None,:]
            forward_prob_each=torch.logsumexp(prob_all_h[:,j,:,None]+prob_all_y[:,j,None,None]+np_,axis=-1)

            forward_prob[:,j]=forward_prob_each-torch.logsumexp(forward_prob_each,1)[:,None]
    return forward_prob
            
def dis_backward_pass(prob_all_s,prob_initial,prob_all_h,prob_all_y):
    T=prob_all_s.shape[1]
    N=prob_all_s.shape[2]
    backward_prob=torch.ones((prob_all_s[:,:,:,0].shape))
    for j in range(T-1,-1,-1):
        if j==T-1:
            backward_prob_each=torch.zeros((prob_all_s.shape[0],N))
            backward_prob[:,j]=backward_prob_each
        else:
            np_=backward_prob[:,j+1,:,None]+prob_all_s[:,j+1,:,:]+prob_all_h[:,j+1,:,None]+prob_all_y[:,j+1,None,None]
            backward_prob_each=torch.logsumexp(np_,axis=1)
            backward_prob[:,j]=backward_prob_each-torch.logsumexp(backward_prob_each,axis=1)[:,None]
    return backward_prob

def get_gamma(forward_prob,backward_prob,prob_all_s,prob_all_h,prob_all_y):

    forward_expprob=torch.exp(forward_prob-torch.logsumexp(forward_prob,2)[:,:,None])
    backward_expprob=torch.exp(backward_prob-torch.logsumexp(backward_prob,2)[:,:,None])
    gamma1=torch.zeros(forward_prob.shape[0],forward_prob.shape[2])
    for n in range(forward_prob.shape[2]):
        gamma1[:,n]=forward_prob[:,0,n]+backward_prob[:,0,n]
    gamma1=gamma1-torch.logsumexp(gamma1,1)[:,None]

    gamma2=torch.zeros((forward_prob.shape[0],forward_prob.shape[1]-1,forward_prob.shape[2],forward_prob.shape[2]))
    for t in range(forward_prob.shape[1]-1):
        for i in range(forward_prob.shape[2]):
            for j in range(forward_prob.shape[2]):
                gamma2[:,t,i,j]=forward_prob[:,t,i]+prob_all_s[:,t,i,j]+backward_prob[:,t+1,j]+prob_all_h[:,t+1,j]+prob_all_y[:,t+1]
    sum_gamma2=torch.zeros((forward_prob.shape[0],forward_prob.shape[1]-1))
    for t in range(forward_prob.shape[1]-1):
        sum_gamma2_each=torch.zeros((forward_prob.shape[0],forward_prob.shape[2],forward_prob.shape[2]))
        for k in range(forward_prob.shape[2]):
            for w in range(forward_prob.shape[2]):
                sum_gamma2_each[:,k,w]=forward_prob[:,t,k]+prob_all_s[:,t,k,w]+backward_prob[:,t+1,w]+prob_all_h[:,t+1,w]+prob_all_y[:,t+1]
        sum_gamma2[:,t]=torch.logsumexp(sum_gamma2_each.reshape(forward_prob.shape[0],forward_prob.shape[2]*forward_prob.shape[2]),1)
    gamma2=gamma2-sum_gamma2[:,:,None,None]
    return gamma1,gamma2