# -*- coding: utf-8 -*-
"""demo_lorenz


# Switching Recurrent Neural Networks (SRNNs) Tutorial on Lorenz Attractor

### Step 1: Package load
In this step, we load the packages of SRNNs. **Please note** the initializatins are very important to SRNNs as well as other SSM models to shorten training time and avoid stuck of training as we discussed in our paper. If you use HMM as a initialization, you have two options:
<br>
(1) We provide HMM package from Linderman SSM, you may **import ssm** to test whether the SSM package is installed. If so, you can specify the initialization method to be 'hmm' in step 5.
<br>
(2) If the ssm doesn't work for you or you would like to use your own initialization (e.g., your own HMM, your own labels, etc.), please change the initialization method to be 'defined'.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import os
from SRNN import model_srnn
from SRNN import inference_network
from SRNN import initialization
from SRNN import train
from SRNN import generative_check
from sklearn.metrics import mean_squared_error

np.random.seed(131)
torch.manual_seed(131)

"""### Step 2: Data load
In this step, we load and visualize the data of lorenz attractor.
"""

y_c=np.load('./data/lorenz_new.npy') #You may change this 'y_c' to your own data, the data has to be in size of (#samples*#time points*#features).

"""We then split the data into training and testing, i.e., 17 trials in training and 1 trials in testing. 'jobid' is to set which trial in testing."""

jobid=0
train_data=np.delete(y_c,(int(jobid)),axis=0)
test_data=y_c[int(jobid):int(jobid)+1]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype = torch.float32
X_train=torch.tensor(0*train_data,dtype=dtype,device=device)
y_train=torch.tensor(train_data,dtype=dtype,device=device)
y_test=torch.tensor(test_data,dtype=dtype,device=device)
X_test=torch.tensor(0*test_data,dtype=dtype,device=device)
### beh_all_train and beh_all_test are behavioral labels, we don't have to use them in lorenz attractor.
beh_all_train=None
beh_all_test=None

"""### Step 3: Hyperparameters"""

input_shape=X_train.shape[2] # Input shape of SRNNs, but the models are input free.
num_tv=2 # Number of RNNs in SRNNs.
hidden_shape=3 # Number of hidden states of SRNNs.
ini_epochs=3000 # Epochs in initialization stage, can be longer than training stage.
coef_cross=5e-1 # Coefficient of initialization, larger coef_cross means larger constraint on posterior states in initialization.
epochs=1000 # Epochs in training stage.
lr=0.001 # Learning rate

"""### Step 4: Define SRNN and Inference Networks"""

model = model_srnn.Model(input_shape,num_tv,hidden_shape).to(device)
rnninfer=inference_network.RNNInfer(input_shape,hidden_shape).to(device)


"""### Step 5: Initialization"""

optimizer = torch.optim.Adam(list(model.parameters())+list(rnninfer.parameters()) ,lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.8)

model_ini,rnninfer_ini,mse_all_ini,error_all_ini,mse_all_test_ini,error_all_test_ini,loss_all_ini,pos_test_all_ini=initialization.run(model,
                                                                                                                     rnninfer,
                                                                                                                     optimizer,
                                                                                                                     scheduler,
                                                                                                                     X_train,
                                                                                                                     y_train,
                                                                                                                     X_test,
                                                                                                                     y_test,
                                                                                                                     beh_all_train,
                                                                                                                     beh_all_test,
                                                                                                                     num_tv,
                                                                                                                     coef_cross,
                                                                                                                     ini_epochs,
                                                                                                                     device,
                                                                                                                     method='hmm',
                                                                                                                     t_load=None)

"""Now, we can test the SRNN after initialization."""

y_pred_test_ini,pos_test_ini,sampled_h_test_ini=train.eval_(model_ini,rnninfer_ini,X_test,y_test,device)

"""#### We also include another tutorial using 'random' initailization, SRNNs are also able to identify correct states.

### Step 6: Training
"""

model_trained,rnninfer_trained,mse_all_train,error_all_train,mse_all_test,error_all_test,loss_all,pos_test_all=train.train_(model_ini,
                                                                                                                    rnninfer_ini,
                                                                                                                    optimizer,
                                                                                                                    scheduler,
                                                                                                                    X_train,
                                                                                                                    y_train,
                                                                                                                    X_test,
                                                                                                                    y_test,
                                                                                                                    beh_all_train,
                                                                                                                    beh_all_test,
                                                                                                                    num_tv,
                                                                                                                    epochs,
                                                                                                                    device)

"""Now, we can test the SRNN after training.

### Step 7: Analysis
"""

y_pred_test,pos_test,sampled_h_test=train.eval_(model_trained,rnninfer_trained,X_test,y_test,device)


torch.save({
            'num_tv':num_tv,
            'hidden_shape':hidden_shape,
            'y_test':y_test.cpu().detach().numpy(),
            'X_test':X_test.cpu().detach().numpy(),
            'model_state_dict_ini': model_ini.state_dict(),
            'rnninfer_state_dict_ini': rnninfer_ini.state_dict(),
            'optimizer_state_dict_ini': optimizer.state_dict(),
            'mse_all_ini':mse_all_ini,
            'error_all_ini':error_all_ini,
            'mse_all_test_ini':mse_all_test_ini,
            'error_all_test_ini':error_all_test_ini,
            'loss_train_ini':loss_all_ini,
            'pos_test_all_ini':pos_test_all_ini,
            'model_state_dict': model_trained.state_dict(),
            'rnninfer_state_dict': rnninfer_trained.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'mse_all':mse_all_train,
            'error_all':error_all_train,
            'mse_all_test':mse_all_test,
            'error_all_test':error_all_test,
            'loss_train':loss_all,
            'pos_test_all':pos_test_all,
            }, './result/model_'+str(jobid)+'.pt')




