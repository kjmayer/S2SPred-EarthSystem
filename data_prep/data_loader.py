import torch
import numpy as np
import xarray as xr

class GetData():
    def __init__(self,dir,var,mem):
        data = xr.open_dataset(dir+var+'_anom_'+mem+'_1950-2014.nc')[var]
        self.data = data

        # need to standardize anom input and output data
        # need to -1 to 1 normalize standardized anoms and raw climo input
        # need to get input and output size... 
        # ...to get the number of nodes in input and output layers

        


class MakeDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(torch.from_numpy(X), dtype = torch.float32).unsqueeze(1)
        self.y = torch.tensor(torch.from_numpy(X), dtype = torch.float32) # unsqueeze?
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return TensorDataset(self.X,self.y)