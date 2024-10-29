import numpy as np
import torch
import torch.nn as nn
import time
from SRNN import loss_function
from SRNN import train
from SRNN import utils
np.random.seed(131)
torch.manual_seed(131)

# Reorder the discrete states in sequence 
def reorder_state(states):
    states_copy=states.copy()
    for i in range(len(states)):
        if i==0:
            states_copy[i]=0
            check_num=0
        else:
            if states[i] in states[:i]:
                idx_=np.where(states==states[i])[0][0]
                states_copy[i]=states_copy[idx_]
            else:
                check_num=check_num+1
                states_copy[i]=check_num
    return states_copy
# Discrete states initialized using Hidden Markov Model (HMM)
# Requires ssm package
def ini_hmm(y_train,num_states):
    import ssm
    obs_dim = np.shape(y_train)[-1] 
    hmm = ssm.HMM(num_states, obs_dim, observations="gaussian")
    hmm_lls = hmm.fit(y_train,method='em', num_iters=20, init_method="kmeans")
    hmm_z_all=[]
    for jobid in range(len(y_train)):
        hmm_z = hmm.most_likely_states(y_train[jobid])
        hmm_z_all.append(reorder_state(hmm_z))
    return np.array(hmm_z_all)
# Discrete states initialized using Kmeans
def ini_kmeans(y_train,num_states):
    Ts = [data.shape[0] for data in y_train]
    
    from sklearn.cluster import KMeans
    km = KMeans(num_states)
    km.fit(np.vstack(y_train))
    zs = np.split(km.labels_, np.cumsum(Ts)[:-1])
    return np.array(zs)
# Discrete states initialized randomly
def ini_random(y_train,num_states):
    random_z_all=[]
    for jobid in range(len(y_train)):
        random_z=np.random.choice(num_states, np.shape(y_train)[1], replace=True)
        random_z_all.append(reorder_state(random_z))
    return np.array(random_z_all)
# Discrete states initialized using mean, e.g., 000111222333444
def ini_mean(y_train,num_states):
    mean_z_all=[]
    mean_z=np.zeros(np.shape(y_train)[1])
    lo=int(np.shape(y_train)[1]/num_states)
    for i in range(1,num_states):
        mean_z[int(i*lo):int(i*lo+lo)]=i
    for jobid in range(len(y_train)):
        mean_z_all.append(mean_z)
    return np.array(mean_z_all)
# One hot coding, we need this to initialize the probability model, for example, P(zt=i)=1, P(zt=j)=0, etc.   
def one_hot(hmm_z_all,num_tv):
    pp=np.zeros((hmm_z_all.shape[0],hmm_z_all.shape[1],num_tv))

    for kk in range(hmm_z_all.shape[0]):
        t_all=hmm_z_all[kk]
        for i in range(hmm_z_all.shape[1]):

            pp[kk,i,int(t_all[i])]=1
    return pp
# Initialization training
def run(model,rnninfer,optimizer,scheduler,X_train,y_train,X_test,y_test,beh_labels,beh_labels_test,num_tv,coef_cross,epochs,device,method='hmm',t_load=None):
    if method=='hmm':
        z_all=ini_hmm(list(y_train.cpu().detach().numpy()),num_tv)
        pp=one_hot(z_all,num_tv)
    if method=='kmeans':
        z_all=ini_kmeans(list(y_train.cpu().detach().numpy()),num_tv)
        pp=one_hot(z_all,num_tv)
    if method=='random':
        z_all=ini_random(list(y_train.cpu().detach().numpy()),num_tv)
        pp=one_hot(z_all,num_tv)
    if method=='mean':
        z_all=ini_mean(list(y_train.cpu().detach().numpy()),num_tv)
        pp=one_hot(z_all,num_tv)
    if method=='defined':
        z_all=t_load
        pp=one_hot(z_all,num_tv)
    if method=='uniform':
        pp=(1/num_tv)*np.ones((y_train.shape[0],y_train.shape[1],num_tv))
    pp=torch.tensor(pp)
    mse_all=np.ones(epochs)
    error_all=np.ones(epochs)
    mse_all_test=np.ones(epochs)
    error_all_test=np.ones(epochs)
    loss_save=np.ones(epochs)
    start_time = time.time()
    pos_test_save_all=np.zeros((epochs,y_test.shape[0],y_test.shape[1],num_tv))
    for epoch in range(epochs):
        model.train()
        rnninfer.train()
        optimizer.zero_grad()
        infer_dist,sampled_h,mean_out=rnninfer(y_train)
        prob_ini,prob_all_s,prob_all_h,prob_all_y,gamma1,delta1,fwp,bwp = model(X_train,y_train,sampled_h,device)
        t1,t2=loss_function.get_loss(gamma1,delta1,prob_ini.cpu(),prob_all_s,prob_all_h,prob_all_y)
        posterior_lk=model.get_posterior_lk(fwp,bwp)
        cross_en=loss_function.get_cross_entropy(posterior_lk,pp)
        loss_all=-(t1.mean()+t2.mean()+infer_dist.entropy().mean()+coef_cross*(cross_en))
        loss_all.backward(retain_graph=True)
        if epoch%100==0:
            print(f"Epoch {epoch+1}/{epochs}, loss = {loss_all}")
            end_time = time.time()
            if epoch!=0:
                utils.compute_time(start_time,end_time,epochs,epoch+1)
        loss_save[epoch]=loss_all
        optimizer.step()
        scheduler.step()
        y_pred_train,pos_train,_=train.eval_(model,rnninfer,X_train,y_train,device)
        mse_,error_=train.compute_metric(y_train,y_pred_train,pos_train,beh_labels)
        mse_all[epoch]=mse_
        error_all[epoch]=error_.mean()

        y_pred_test,pos_test,_=train.eval_(model,rnninfer,X_test,y_test,device)
        mse_test_,error_test_=train.compute_metric(y_test,y_pred_test,pos_test,beh_labels_test)
        mse_all_test[epoch]=mse_test_
        error_all_test[epoch]=error_test_.mean()
        pos_test_save_all[epoch]=pos_test
    return model,rnninfer,mse_all,error_all,mse_all_test,error_all_test,loss_save,pos_test_save_all