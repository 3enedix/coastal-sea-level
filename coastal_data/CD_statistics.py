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

def compute_trend_with_error(x, y):
    '''
    Estimates the trend of a linear function estimated with least squares through a given set of points.
    Same as compute_trend, but additionally returns the covariance matrix.
    
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

    # Compute covariance matrix
    # Verbesserungen v=Ax-b (distance of observations to the estimated regression line)
    v = np.matmul(A, xs) - y
    n_obs = len(y)
    n_unknown = 2
    v02 = np.matmul(np.transpose(v), v) / (n_obs - n_unknown) # v'v/f Varianzfaktor a posteriori
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