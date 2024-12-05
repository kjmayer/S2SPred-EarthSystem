import torch
import numpy as np
import xarray as xr

def _makefipath(dir,var,finames):
    return [dir+var+'/'+var+finames[ifi] for ifi in range(len(finames))]

def lead_shift(data,lead,forward=True):
    if forward: # output
        data_shift = data[lead:]
    else: #input
        data_shift = data[:-lead]
    return data_shift

def concat_input(X,Y,dim_name='features'):
    firstdim = X.dims[0]
    thirddim = X.dims[-2]
    forthdim = X.dims[-1]
    inputs = xr.concat([X,Y],dim=dim_name).transpose(firstdim,dim_name,thirddim,forthdim)
    return inputs
        
class GetXData():
    
    def __init__(self, dir, var, finames,
                 train=False,
                 trainmean=None,
                 trainstd=None,
                 trainmin=None,
                 trainmax=None,
                 climo=False):
        
        self.climo = climo
        self.train = train
        self.var = var

        if not self.climo and not self.train:
            # self.trainmean = trainmean
            # self.trainstd = trainstd
            self.trainmin = trainmin
            self.trainmax = trainmax
        
        if len(finames) > 1:
            data = xr.open_mfdataset(_makefipath(dir, var, finames),
                                     concat_dim='mem',
                                     combine="nested",
                                     parallel=True)[var]
            data = data.stack(s=('mem', 'time')).transpose('s', 'lat', 'lon')
            data = data.reset_index(['s'])
        else:
            data = xr.open_dataset(dir+var+'/'+var+finames[0])[var]
        self.data = data

        # self.process()
    
    # def standardize(self):
    #     if self.train:
    #         self.trainmean = self.data.mean('s')
    #         self.trainstd = self.data.std('s')
    #     return (self.data - self.trainmean)/self.trainstd

    def minmax_normalize(self):
        if self.climo:
            self.climomin = self.data.min('dayofyear')
            self.climomax = self.data.max('dayofyear')
            return((self.data - self.climomin)/(self.climomax - self.climomin))
        else:
            if self.train:
                self.trainmin = self.data.min('s')
                self.trainmax = self.data.max('s')
            return((self.data - self.trainmin)/(self.trainmax - self.trainmin))
            # self.trainmin = self.datastd.min('s')
            # self.trainmax = self.datastd.max('s') 
            # return((self.datastd - self.trainmin)/(self.trainmax - self.trainmin))       

    def __len__(self):
        return len(self.datanorm)
    
    def __getitem__(self,idx):
        # if not self.climo:
        #     self.datastd = self.standardize()
        self.datanorm = self.minmax_normalize()

        ############################################
        if self.train and not self.climo:
            return self.datanorm, self.trainmin, self.trainmax #self.trainmean, self.trainstd,
        if self.climo:
            return self.datanorm, self.climomin, self.climomax
        else:
            return self.datanorm

class GetYData():
    
    def __init__(self, dir, var, finames,
                 train=False,
                 trainmean=None,
                 trainstd=None,
                 norm=False,
                 trainmax=None,
                 trainmin=None,
                ):
        
        self.train = train
        self.var = var
        self.norm = norm
        
        # if not self.train:
        #     self.trainmean = trainmean
        #     self.trainstd = trainstd
        if not self.train and self.norm:
            self.trainmax = trainmax
            self.trainmin = trainmin
        
        if len(finames) > 1:
            data = xr.open_mfdataset(_makefipath(dir, var, finames),
                                     concat_dim='mem',
                                     combine="nested",
                                     parallel=True)[var]
            data = data.stack(s=('mem', 'time')).transpose('s', 'lat', 'lon')
            data = data.reset_index(['s'])
        else:
            data = xr.open_dataset(dir+var+'/'+var+finames[0])[var]
        self.data = data
    
    # def standardize(self): # this will error if only one finame (s needs to be time)
    #     if self.train:
    #         self.trainmean = self.data.mean('s')
    #         self.trainstd = self.data.std('s')
    #     return (self.data - self.trainmean)/self.trainstd
        
    def minmax_normalize(self): # this will error if only one finame (s needs to be time)
        if self.train:
            self.trainmin = self.data.min('s')
            self.trainmax = self.data.max('s')
        return((self.data - self.trainmin)/(self.trainmax - self.trainmin))
        #     self.trainmin = self.datastd.min('s')
        #     self.trainmax = self.datastd.max('s')
        # return((self.datastd - self.trainmin)/(self.trainmax - self.trainmin))  

    def __len__(self):
        return len(self.datanorm)
    
    def __getitem__(self, idx):
        # self.datastd = self.standardize()
        if self.norm:
            self.datanorm = self.minmax_normalize()

        ############################################
        if self.train and not self.norm:
            return self.datastd  # , self.trainmean, self.trainstd
        elif self.train and self.norm:
            return self.datanorm, self.trainmax, self.trainmin  # self.trainmean, self.trainstd,
        # elif not self.train and not self.norm:
        #     return self.datastd
        elif not self.train and self.norm:
            return self.datanorm
        else:
            print('WARNING: nothing to return')

############################################
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, X, y, lat_map, lon_map):
        self.X = torch.tensor(X, dtype = torch.float32)
        self.y = torch.tensor(y, dtype = torch.float32)
        self.lat_map = torch.tensor(lat_map, dtype = torch.float32).unsqueeze(0)
        self.lon_map = torch.tensor(lon_map, dtype = torch.float32).unsqueeze(0)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        input = self.X[idx]
        target = self.y[idx]

        lat_map = self.lat_map  # Shape (1, H, W)
        lon_map = self.lon_map  # Shape (1, H, W)
        input_wlat = torch.cat((input, lat_map), dim=0)  # Shape (C+1, H, W)
        input_wlatlon = torch.cat((input_wlat, lon_map), dim=0)  # Shape (C+2, H, W)
        
        return input_wlatlon, target





