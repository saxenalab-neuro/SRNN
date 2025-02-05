import numpy as np
import torch
import torch.nn as nn
np.random.seed(131)
torch.manual_seed(131)

# Inference network
class RNNInfer(nn.Module):
    def __init__(self,input_shape,hidden_shape):
        super(RNNInfer, self).__init__()
        self.input_shape=input_shape
        self.hidden_shape=hidden_shape
        # We use bidirectional RNN in the inference network, if using forward-rnn only (e.g., for causal inference), please delete backward_rnn
        self.forward_rnn = nn.RNN(input_size=self.input_shape, hidden_size=self.hidden_shape, num_layers=1, batch_first=True) # forward rnn
        self.backward_rnn = nn.RNN(input_size=self.input_shape, hidden_size=self.hidden_shape, num_layers=1, batch_first=True) # backward rnn
        self.rnn_mean=nn.RNN(input_size=self.hidden_shape, hidden_size=self.hidden_shape, num_layers=1, batch_first=True) # simple rnn (output layers)
        self.lc1=nn.Linear(self.hidden_shape+self.hidden_shape,64)
        self.lc2=nn.Linear(64,self.hidden_shape)
        self.relu=nn.ReLU()



    def forward(self, y_train):
        device = device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        batches = int(y_train.shape[0])
        dtype = torch.float32

        h_f0 = torch.zeros([1,batches, self.hidden_shape],device=device)
        h_b0 = torch.zeros([1,batches,self.hidden_shape],device=device)
        h0=torch.zeros([1,batches,self.hidden_shape],device=device)
        f_out,_=self.forward_rnn(y_train,h_f0) # forward pass
        b_out,_=self.backward_rnn(torch.flip(y_train,dims=(1,)),h_b0) # backward pass
        bi_output=f_out+torch.flip(b_out,dims=(1,)) # bidirectional output
        mean_out=torch.zeros((batches,y_train.shape[1],self.hidden_shape),device=device)
        sampled_h=torch.zeros((batches,y_train.shape[1],self.hidden_shape),device=device)
        h_last=h0
        sampled_h_=torch.zeros(sampled_h[:,0:1,:].shape,device=device)
        for i in range(y_train.shape[1]):
            emb=torch.cat((bi_output[:,i:i+1,:],sampled_h_),axis=-1)
            mean_out_,h_=self.rnn_mean(self.lc2(self.relu(self.lc1(emb))),h_last)
            h_last=h_
            covariance_matrix=(1e-4)*torch.eye(self.hidden_shape,device=device)
            ep=torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros((mean_out_[:,0].shape),device=device), covariance_matrix).sample()
            sampled_h_=mean_out_+ep[:,None,:]*covariance_matrix.diag() # reparameterization trick in order to compute gradients
            sampled_h[:,i,:]=sampled_h_[:,0,:]
            mean_out[:,i,:]=mean_out_[:,0,:]
        infer_dist=torch.distributions.multivariate_normal.MultivariateNormal(mean_out, covariance_matrix) # we need this distribution to compute the entropy in our loss
        return infer_dist,sampled_h,mean_out
