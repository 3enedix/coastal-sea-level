import numpy as np
import pandas as pd
from scipy.sparse import lil_array, csc_array, linalg
from scipy.spatial import cKDTree
from shapely import get_coordinates

import pdb

# ===================================================================================
# Metrics to compare two datasets
# ===================================================================================

def compute_mean_error(diff, print_result=True):
    '''
    Mean error
    diff - array like vector of differences
    '''
    me = np.nanmean(diff) # Mean error
    if print_result:
        print(f'Mean error: {round(me,2)} m')
    return me

def compute_mean_absolute_error(diff, print_result=True):
    '''
    Mean absolute error
    diff - array like vector of differences
    '''
    mae = np.nanmean(abs(diff))
    if print_result:
        print(f'Mean absolute error: {round(mae,2)} m')
    return mae

def compute_rmse(diff, print_result=True):
    '''
    Root mean square error
    diff - array like vector of differences
    '''
    rmse = np.sqrt(np.nanmean(diff**2))
    if print_result:
        print(f'Root mean square error: {round(rmse,2)} m')
    return rmse

def compute_mad_mean(diff, print_result=True):
    '''
    Mean absolute deviation from the mean
    diff - array like vector of differences
    '''
    mad_mean = np.nanmean(abs(diff - np.nanmean(diff)))
    if print_result:
        print(f'Mean absolute deviation from mean: {round(mad_mean,2)} m')
    return mad_mean

def compute_mad_med(diff, print_result=True):
    '''
    Median absolute deviation from the median
    diff - array like vector of differences
    '''
    mad_med = np.nanmedian(abs(diff - np.nanmedian(diff)))
    if print_result:
        print(f'Median absolute deviation from median: {round(mad_med,2)} m')
    return mad_med

def print_stats(diff, fn=None, path=None):
    me = compute_mean_error(diff)

    diff_debias = diff - me
    print("---De-bias with mean---")
    me = compute_mean_error(diff_debias)
    mae = compute_mean_absolute_error(diff_debias)
    rmse = compute_rmse(diff_debias)
    mad_mean = compute_mad_mean(diff_debias)
    mad_med = compute_mad_med(diff_debias)

    if path:       
        with open(f'{path}{fn}.txt', 'w') as f:
            f.write(f'Mean error: {round(me,2)} m')
            f.write(f'\nMean absolute error: {round(mae,2)} m')
            f.write(f'\nMean absolute error: {round(mae,2)} m')
            f.write(f'\nRoot mean square error: {round(rmse,2)} m')
            f.write(f'\nMean absolute deviation from mean: {round(mad_mean,2)} m')
            f.write(f'\nMedian absolute deviation from median: {round(mad_med,2)} m')

def all_stats_per_year(rts_data, jarkus_gdf, year_list, reduce_mean=False, static=False):
    '''
    Computes all statistics per year.
    
    Input
    ------------------------
    rts_data: array like vector of differences
    jarkus_gdf: array like vector of differences
    year_list: list of years
    static: boolean, if True, the static data is used
    
    Output
    ------------------------
    stats_df: dataframe with the results
    all_diffs: dictionary with all differences
    '''

    print_result = False
    all_diffs = {}
    stats_df = pd.DataFrame(index=year_list)
    
    for year in year_list:
        if static:            
            diff = rts_data - jarkus_gdf[str(year)]
        else:
            diff = rts_data[str(year)] - jarkus_gdf[str(year)]
        all_diffs[str(year)] = diff # for aggregate histogram

        if reduce_mean:
            me = compute_mean_error(diff, print_result)
            diff = diff - me
            
        stats_df.loc[str(year), 'me'] = compute_mean_error(diff, print_result)
        stats_df.loc[str(year), 'mae'] = compute_mean_absolute_error(diff, print_result)
        stats_df.loc[str(year), 'rmse'] = compute_rmse(diff, print_result)
        stats_df.loc[str(year), 'mad_mean'] = compute_mad_mean(diff, print_result)
        stats_df.loc[str(year), 'mad_med'] = compute_mad_med(diff, print_result) 
     
    return stats_df, all_diffs

# ===================================================================================
# Timeseries analysis
# ===================================================================================

def rms(ts):
    return round(np.sqrt(np.nanmean(ts **2)),3)

def std(ts): # Variation in one timeseries
    err = ts - np.nanmean(ts)
    return rms(err)

def RMSE_timeseries(ts1, ts2):
    '''
    ts1: array, timeseries 1
    ts2: array, timeseries 2
    '''
    if len(ts1) != len(ts2):
        raise ValueError('Timeseries have different lengths.')
    
    ts1_red = ts1 - np.nanmean(ts1)
    ts2_red = ts2 - np.nanmean(ts2)
    rmse = np.sqrt(np.nanmean((ts1_red - ts2_red)**2))   
    return rmse

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

def three_sigma_outlier_rejection(ts):
    sigma3 = 3*std(ts)
    ts_red = ts[np.abs(ts-np.nanmedian(ts)) <= sigma3]
    return ts_red

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
    covariance matrix (of trend and y-axis-intercept)
    array of differences between observation and model (verbesserungen v)
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

    A = np.transpose(np.vstack((col1, col2)))
    
    N = np.matmul(np.transpose(A), A) # A'A
    n = np.matmul(np.transpose(A), y) # A'l
    xs = np.linalg.solve(N,n)
    
    return round(xs[0], 4)

def compute_trend_with_error(x, y, P=None):
    '''
    Estimates the trend of a linear function estimated with least squares through a given set of points.
    Same as compute_trend, but additionally returns the covariance matrix.
    
    Input
    ------------------------
    x: array or list of timestamps in [years]
    y: array or list of values in [units]
    P: covariance matrix of y-values (optional)    
    
    Output
    ------------------------
    trend of the linear function in [units]/[years]
    covariance matrix (of trend and y-axis-intercept)
    array of differences between observation and model (verbesserungen v)
    '''    

    # remove nans
    idx_nan = np.nonzero(np.isnan(y))[0]
    x = np.delete(x, idx_nan)
    y = np.delete(y, idx_nan)
    
    if P is None:
        P = np.diag(np.ones(len(y)))
    
    # A-Matrix
    # Linear function: y = m*x + t
    # Parameters: slope m, y-intercept t -> columns
    # Measurements -> rows
    # column 1: partial derivative df/dm = x
    # column 2: partial derivative df/dt = 1
    
    col1 = x - np.mean(x)
    col2 = np.ones(len(y))
    
    A = np.vstack((col1, col2)).T
    
    N = A.T @ P @ A
    n = A.T @ P @ y
    xs, _, _, _ = np.linalg.lstsq(N,n, rcond=-1)
    
    # Compute covariance matrix
    # Verbesserungen v=Ax-b (distance of observations to the estimated regression line)
    v = A @ xs - y
    n_obs = len(y)
    n_unknown = 2
    v02 = (v.T @ v)/ (n_obs - n_unknown) # v'v/f Varianzfaktor a posteriori
    Qxx = np.linalg.inv(N) # Kofaktormatrix der ausgegelichenen Unbekannten
    Cxx = v02 * Qxx # Kovarianzmatrix der ausgegelichenen Unbekannten
    
    return round(xs[0], 4), Cxx, v

def compute_periodic_signal_and_trend(t, l, f):
    '''
    Estimates a periodic signal of frequency f and the trend of a discrete function with least squares.
    
    Input
    ------------------------
    t: array or list of timestamps (preferably [years])
    l: array or list of values in [units]
    
    Output
    ------------------------
    amplitude in [units]
    phase in rad
    trend in [units]/[years]
    offset in [units]

    covariance matrix of all 4 unknowns
    '''
        
    # A-matrix
    # 1. column: df/dy = cos(2 pi f t) (y = A sin omega t)'
    # 2. column: df/dy = sin (2 pi f t) (y = B cos omega t)'
    # 3. column: df/dy = t (y = C t)' (trend)
    # 4. column: df/dy = 1 (y = D)' (offset)
    
    col1 = np.cos(2*np.pi*f*t)
    col2 = np.sin(2*np.pi*f*t)
    col3 = t
    col4 = np.ones(len(t))
    
    A = np.transpose(np.vstack((col1, col2, col3, col4)))
    
    N = np.matmul(np.transpose(A), A) # A'A
    n = np.matmul(np.transpose(A), l) # A'l
    xs = np.linalg.solve(N,n)
    
    a, b = xs[0], xs[1]
    c, d = xs[2], xs[3] # trend, offset
    amplitude = np.sqrt(a**2 + b**2)
    phase = np.arctan2(a, b)
    
    return amplitude, phase, c, d

def first_derivative(x, y, n):
    '''
    Input
    x - array of x-values
    y - array of y-values
    n - int, 0.5 filterlength (n = 1: no smoothing)

    Output
    m - first derivative
    '''
    df = np.concatenate((np.ones(n),-np.ones(n)))
    
    dy = np.convolve(y, df, 'valid')
    dx = np.convolve(x, df, 'valid')

    m = dy / dx
    
    return m



