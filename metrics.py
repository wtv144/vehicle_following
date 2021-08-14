import torch
import torch.nn as nn
import numpy as np
import math
class nll(nn.Module):
    def __init__(self):
        super(nll,self).__init__()
        
        #pred is a distribution, fit it better to a target
    def forward(self, V_pred, V_trgt):
    # 2d tensor of mux, muy, sx, sy, corr
    #(batch, timesteps, (x,y)) tgt
    #(batch, timesteps, (5d distribution tensor))
        normx = V_trgt[...,0]- V_pred[...,0]
        normy = V_trgt[...,1]- V_pred[...,1] #changed :, to ...,
        sx = torch.exp(V_pred[...,2]) #sx
        sy = torch.exp(V_pred[...,3]) #sy
        corr = torch.tanh(V_pred[...,4]) #corr
        sxsy = sx*sy
        z = (normx/sx)**2 + (normy/sy)**2 - 2*((corr*normx*normy)/sxsy)
        negRho = 1 - corr**2

        # Numerator
        result = torch.exp(-z/(2*negRho))
        # Normalization factor
        denom = 2 * np.pi * (sxsy * torch.sqrt(negRho))

        # Final PDF calculation
        result = result / denom

        # Numerical stability
        epsilon = 1e-5

        result = -torch.log(torch.clamp(result, min=epsilon))
        result = torch.mean(result)
        print(result)
        return result 
#Uses descriptions from here: http://ai.stanford.edu/blog/trajectory-forecasting/ 
    
