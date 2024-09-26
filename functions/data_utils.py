import xarray as xr
import pandas as pd
import numpy as np
from numpy.polynomial import polynomial
import os
import matplotlib.pyplot as plt

def is_month(data, months):
    i_timedim = np.where(np.asarray(data.dims) == 'time')[0][0]
    if i_timedim == 0:
        data = data[data.time.dt.month.isin(months)]
    elif i_timedim == 1:
        data = data[:,data.time.dt.month.isin(months)]
    return data

def detrend_members(data, ensmean_data, npoly=3):
    '''
    detrend ensemble member using polynomial fit (for each doy) to the ensemble mean
    
    data: [member, time, lat, lon] or [member, time]
        ensemble members to detrend 
    
    ensmean_data: [time, lat, lon] or [time]
        ensemble mean 
    
    npoly: [int] 
        order of polynomial, default = 3rd order
    '''
    
    # stack lat and lon of ensemble mean data
    if len(ensmean_data.shape) == 3:
        ensmean_data = ensmean_data.stack(z=('lat', 'lon'))
 
    # stack lat and lon of member data & grab doy information
    if len(data.shape) >= 3:
        data = data.stack(z=('lat', 'lon'))
    temp = data['time.dayofyear']
    
    # grab every Xdoy from ensmean, fit npoly polynomial
    # subtract polynomial from every Xdoy from members
    detrend = []
    for label,ens_group in ensmean_data.groupby('time.dayofyear'):
        Xgroup = data.where(temp == label, drop = True)
        
        curve = polynomial.polyfit(np.arange(0, ens_group.shape[0]), ens_group, npoly)
        trend = polynomial.polyval(np.arange(0, ens_group.shape[0]), curve, tensor=True)
        if len(ensmean_data.shape) >= 2: #combined lat and lon, so now 2-3
            trend = np.swapaxes(trend,0,1) #only need to swap if theres a space dimension

        diff = Xgroup - trend
        detrend.append(diff)

    detrend_xr = xr.concat(detrend,dim='time').unstack()
    detrend_xr = detrend_xr.sortby('time')
    
    return detrend_xr

