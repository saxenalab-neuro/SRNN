import numpy as np
import torch
import torch.nn as nn
from SRNN import loss_function
from sklearn.metrics import mean_squared_error
from sklearn.metrics import balanced_accuracy_score
import time
from SRNN import utils
np.random.seed(131)
torch.manual_seed(131)



def train_(model,rnninfer,optimizer,scheduler,X_train,y_train,X_test,y_test,beh_labels,beh_labels_test,num_tv,epochs,device):
    mse_all=np.ones(epochs)
    error_all=np.ones(epochs)
    mse_all_test=np.ones(epochs)
    error_all_test=np.ones(epochs)
    loss_save=np.ones(epochs)
    start_time=time.time()
    pos_test_save_all=np.zeros((epochs,y_test.shape[0],y_test.shape[1],num_tv))
    for epoch in range(epochs):
        model.train()
        rnninfer.train()
        optimizer.zero_grad()
        infer_dist,sampled_h,mean_out=rnninfer(y_train)
        prob_ini,prob_all_s,prob_all_h,prob_all_y,gamma1,delta1,fwp,bwp = model(X_train,y_train,sampled_h,device)
        t1,t2=loss_function.get_loss(gamma1,delta1,prob_ini.cpu(),prob_all_s,prob_all_h,prob_all_y)
        posterior_lk=model.get_posterior_lk(fwp,bwp)
        loss_all=-(t1.mean()+t2.mean()+infer_dist.entropy().mean())
        loss_all.backward(retain_graph=True)
        if epoch%100==0:
            print(f"Epoch {epoch+1}/{epochs}, loss = {loss_all}")
            end_time = time.time()
            if epoch!=0:
                utils.compute_time(start_time,end_time,epochs,epoch+1)
        loss_save[epoch]=loss_all
        optimizer.step()
        scheduler.step()
        y_pred_train,pos_train,_=eval_(model,rnninfer,X_train,y_train,device)
        mse_,error_=compute_metric(y_train,y_pred_train,pos_train,beh_labels)
        mse_all[epoch]=mse_
        error_all[epoch]=error_.mean()

        y_pred_test,pos_test,_=eval_(model,rnninfer,X_test,y_test,device)
        mse_test_,error_test_=compute_metric(y_test,y_pred_test,pos_test,beh_labels_test)
        mse_all_test[epoch]=mse_test_
        error_all_test[epoch]=error_test_.mean()
        pos_test_save_all[epoch]=pos_test
    return model,rnninfer,mse_all,error_all,mse_all_test,error_all_test,loss_save,pos_test_save_all


def eval_(model,rnninfer,X_test,y_test,device):
    model.eval()
    rnninfer.eval()
    infer_dist_test,sampled_h_test,mean_out_test=rnninfer(y_test)
    prob_ini_test,prob_all_s_test,prob_all_h_test,prob_all_y_test,gamma1_test,delta1_test,fwp_test,bwp_test = model(X_test,y_test,sampled_h_test,device)
    
    t1_test,t2_test=loss_function.get_loss(gamma1_test,delta1_test,prob_ini_test.cpu(),prob_all_s_test,prob_all_h_test,prob_all_y_test)
    posterior_lk_test=model.get_posterior_lk(fwp_test,bwp_test)
    pos_test=posterior_lk_test.cpu().detach().numpy()
    
    emission_mean=model.emission(sampled_h_test)
    covariance_matrix=(1e-20)*torch.eye(model.input_shape,device=device)
    emission_dist=torch.distributions.multivariate_normal.MultivariateNormal(emission_mean[:,:,:], covariance_matrix)
    y_pred_test=emission_dist.sample().cpu().detach().numpy()

    return y_pred_test,pos_test,sampled_h_test.cpu().detach().numpy()

def compute_metric(y_train,y_pred_train,pos_train,beh_labels):
    mse_=mean_squared_error(y_pred_train.reshape(-1,y_pred_train.shape[2]),y_train.cpu().detach().numpy().reshape(-1,y_train.shape[2]))
    if beh_labels!=None:
        error_=compute_error(np.argmax(pos_train,axis=-1),beh_labels)
    else:
        error_=np.ones(2)
    return mse_,error_

def compute_error(pos_,truth_):
    srnn_acc_curve=np.ones(pos_.shape[0])
    for i in range(pos_.shape[0]):
        srnn_acc_curve[i]=balanced_accuracy_score(truth_[i], pos_[i])
    return 1-srnn_acc_curve