from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as F
from torchvision import transforms, utils
import pandas as pd
import torch

class TS_Data(Dataset):
    def __init__(self,fname, inpt, out):
        relevant_cols = ['SCC_Follow_Info_6','SCC_Follow_Info_7','SCC_Follow_Info_8', 'VDS_Chassis_CG_Position_0','VDS_Chassis_CG_Position_1', 'VDS_Chassis_CG_Position_2']
        df = pd.read_csv(fname)
        self.obs_len = inpt
        self.pred_len = out
        dat_cols = ['SCC_Follow_Info_6','SCC_Follow_Info_7','SCC_Follow_Info_8']
        target_cols = [ 'VDS_Chassis_CG_Position_0','VDS_Chassis_CG_Position_1', 'VDS_Chassis_CG_Position_2']
        #do relative positioning for all of these 
        data_df = df[dat_cols]
        target_df = df[target_cols]
        ddf = data_df.to_numpy()   #convert both to a numpy array
        tdf = target_df.to_numpy()
        self.d_df = ddf[1:,:] - ddf[:-1,:] #subtract previous row from current 
        self.t_df = tdf[1:,:] - tdf[:-1,:] # same as above
        
    def __getitem__(self,idx):
        in_temp = self.d_df[idx: idx+self.obs_len]
        out_temp = self.t_df[idx+self.obs_len+1:idx+self.obs_len+1+self.pred_len]

        d_df = torch.from_numpy(in_temp).float()
        tg_df = torch.from_numpy(out_temp).float()
        #do any normalization?
        
        return d_df, tg_df
    
    def __len__(self):
        return self.t_df.shape[0]-(self.obs_len+self.pred_len)-1 #modify to prevent out of bounds