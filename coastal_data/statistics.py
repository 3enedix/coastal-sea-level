import numpy as np

def rms(ts):
    return round(np.sqrt(np.nanmean(ts **2)),3)

def std(ts): # Variation in one timeseries
    err = ts - np.nanmean(ts)
    return rms(err)

def moving_average(ts, n):
    '''
    ts: array, timeseries
    n: filterlength
    '''
    # fill the values that are cut off by the 'valid' option with nans:
    if n % 2 == 0:
        fill1 = np.full(int(n/2), np.nan)
        fill2 = np.full(int(n/2)-1, np.nan)
    else:
        fill1 = np.full(int(n/2), np.nan)
        fill2 = fill1
        
    filt = np.ones(n)/n
    ts_sm_filt = np.convolve(ts, filt, 'valid')
    ts_sm = np.concatenate((fill1, ts_sm_filt, fill2))
    
    return ts_sm

def compute_trend(x, y):
    '''
    Estimates the trend of a linear function estimated with least squares through a given set of points.
    
    Input
    ------------------------
    x: array or list of timestamps in [years]
    y: array or list of values in [units]
    
    Output
    ------------------------
    trend of the linear function in [units]/[years]
    '''
    
    
    # remove nans
    idx_nan = np.nonzero(np.isnan(y))[0]
    x = np.delete(x, idx_nan)
    y = np.delete(y, idx_nan)
    
    # A-Matrix
    # Linear function: y = m*x + t
    # Parameters: slope m, y-intercept t -> columns
    # Measurements -> rows
    # column 1: partial derivative df/dm = x
    # column 2: partial derivative df/dt = 1

    col1 = x
    col2 = np.ones(len(y))

    A = np.concatenate((col1[:, None], col2[:, None]), axis=1)
    
    N = np.matmul(np.transpose(A), A) # A'A
    n = np.matmul(np.transpose(A), y) # A'l
    xs = np.linalg.solve(N,n)
    
    return round(xs[0], 4)