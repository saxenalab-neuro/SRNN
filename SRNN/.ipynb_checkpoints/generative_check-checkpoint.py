import numpy as np
import torch
import torch.nn as nn
np.random.seed(131)
torch.manual_seed(131)

def run(model,rnninfer,X_test,y_test,pos_test,device):
    model.eval()
    rnninfer.eval()
    infer_dist_test,sampled_h_test,mean_out_test=rnninfer(y_test)
    pos_infer=np.argmax(pos_test[0],axis=1)
    generative_h=[]
    for j in range(X_test.shape[1]):
        if j==0:
            covariance_matrix=(1e-4)*torch.eye(model.hidden_shape,device=device)
            infer_dist=torch.distributions.multivariate_normal.MultivariateNormal(sampled_h_test[:,0,:], covariance_matrix)
            generative_h.append(infer_dist.sample().cpu().detach().numpy()[0])
        else:
            x_out,h_out=model.rnns[int(pos_infer[j])](X_test[:,j:j+1,:],sampled_h_test[:,j-1,:].unsqueeze(0).contiguous())
            covariance_matrix=(1e-4)*torch.eye(model.hidden_shape,device=device)
            infer_dist=torch.distributions.multivariate_normal.MultivariateNormal(x_out[:,0,:], covariance_matrix)
            generative_h.append(infer_dist.sample().cpu().detach().numpy()[0])   

    
    generative_h_test=torch.tensor(np.array(generative_h)[None],device=device)
    emission_mean=model.emission(generative_h_test)
    covariance_matrix=(1e-20)*torch.eye(model.input_shape,device=device)
    emission_dist=torch.distributions.multivariate_normal.MultivariateNormal(emission_mean[:,:,:], covariance_matrix)
    y_pred_test=emission_dist.sample().cpu().detach().numpy()
    return y_pred_test,np.array(generative_h)



