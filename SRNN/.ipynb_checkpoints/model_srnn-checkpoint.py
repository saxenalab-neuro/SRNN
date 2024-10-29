import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import os
from SRNN import baum_welch
np.random.seed(131)
torch.manual_seed(131)
class Emission(nn.Module):
    def __init__(self,input_shape,hidden_shape):
        super(Emission, self).__init__()
        self.input_shape=input_shape
        self.hidden_shape=hidden_shape
        self.fc1 = nn.Linear(self.hidden_shape,32)
        self.fc2 = nn.Linear(32,64)
        self.fc3 = nn.Linear(64,self.input_shape)
        self.relu=nn.ReLU()
        
    def forward(self, sampled_h_):
        emission_mean=self.fc1(sampled_h_)
        emission_mean=self.relu(emission_mean)
        emission_mean=self.fc2(emission_mean)
        emission_mean=self.relu(emission_mean)
        emission_mean=self.fc3(emission_mean)
        return emission_mean
    
class Model(nn.Module):
    def __init__(self,input_shape,num_tv,hidden_shape):
        super(Model, self).__init__()
        self.input_shape=input_shape
        self.hidden_shape=hidden_shape
        self.num_tv=num_tv
        self.rnns = nn.ModuleList([nn.RNN(input_size=self.input_shape, hidden_size=self.hidden_shape, num_layers=1, batch_first=True) for i in range(self.num_tv)])
        self.emission = Emission(self.input_shape,self.hidden_shape)
        self.transitions = nn.RNNCell(self.hidden_shape, self.num_tv*self.num_tv)
        self.initials = nn.Parameter(torch.randn(self.num_tv), requires_grad=True)


    def forward(self, x_input,y_train,sampled_h,device):
    
        batches = int(x_input.shape[0])
        dtype = torch.float32
        h0_transitions=torch.zeros([batches,self.num_tv*self.num_tv],device=device)
        prob_initial=torch.zeros((x_input.shape[0],self.num_tv))
        for k in range(batches):
            prob_initial[k]=self.initials-torch.logsumexp(self.initials,0)
        prob_all_s=torch.zeros((x_input.shape[0],x_input.shape[1],self.num_tv,self.num_tv))
        prob_all_h=torch.zeros((x_input.shape[0],x_input.shape[1],self.num_tv))
        prob_all_y=torch.zeros(x_input.shape[0],x_input.shape[1])
        # state_prob=prob_initial
     
        for j in range(x_input.shape[1]):
            
            if j==0:
                for i in range(self.num_tv):
                    covariance_matrix=(1e-4)*torch.eye(self.hidden_shape,device=device)
                    infer_dist=torch.distributions.multivariate_normal.MultivariateNormal(sampled_h[:,0,:], covariance_matrix)
                    prob_all_h[:,j,i]=infer_dist.log_prob(sampled_h[:,j,:])
                    for g in range(prob_all_s.shape[0]):
                        prob_all_s[g,j]=torch.eye(self.num_tv,device=device)
                    
            else:
                trans_prob=torch.reshape(self.transitions(sampled_h[:,j-1,:],h0_transitions),(-1,self.num_tv,self.num_tv))
                prob_all_s[:,j]=trans_prob-torch.logsumexp(trans_prob,axis=1)[:,None,:]
                for i in range(self.num_tv):
                    x_out,h_out=self.rnns[i](x_input[:,j:j+1,:],sampled_h[:,j-1,:].unsqueeze(0).contiguous())
                    covariance_matrix=(1e-4)*torch.eye(self.hidden_shape,device=device)
                    infer_dist=torch.distributions.multivariate_normal.MultivariateNormal(x_out[:,0,:], covariance_matrix)
                    prob_all_h[:,j,i]=infer_dist.log_prob(sampled_h[:,j,:])
            
            emission_mean=self.emission(sampled_h[:,j:j+1,:])
            covariance_matrix=(1e-4)*torch.eye(self.input_shape,device=device)
            emission_dist=torch.distributions.multivariate_normal.MultivariateNormal(emission_mean[:,0,:], covariance_matrix)
            prob_all_y[:,j]=emission_dist.log_prob(y_train[:,j,:])
        forward_prob=baum_welch.dis_forward_pass(prob_all_s,prob_initial,prob_all_h,prob_all_y)
        backward_prob=baum_welch.dis_backward_pass(prob_all_s,prob_initial,prob_all_h,prob_all_y)
        gamma1,gamma2=baum_welch.get_gamma(forward_prob,backward_prob,prob_all_s,prob_all_h,prob_all_y)

        return prob_initial,prob_all_s,prob_all_h,prob_all_y,gamma1,gamma2,forward_prob,backward_prob
    
    def get_posterior_lk(self,forward_prob,backward_prob):
        
        posterior_lk=forward_prob+backward_prob
        posterior_lk=posterior_lk-torch.logsumexp(posterior_lk,axis=2)[:,:,None]
        return posterior_lk
