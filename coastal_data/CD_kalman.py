from shapely import get_coordinates
from scipy.spatial import cKDTree
from scipy.sparse import lil_array, eye_array, csc_matrix, issparse
from sksparse.cholmod import cholesky
from sksparse.cholmod import CholmodNotPositiveDefiniteError

import numpy as np
import pandas as pd

from coastal_data import CD_matrix_tools

# ===================================================================================
# Filter (forward & backward)
# ===================================================================================

def KF(x_k, sigma_xx_k, T, q, l, A, sigma_ll, eps_factor):
    '''
    (Forward) Kalman Filter
    
    Input
    -----
    x_k - Numpy array of state in iteration k
    sigma_xx_k - Covariance matrix of the state, diagonal sparse matrix (at least at initialisation)
    T - Transition matrix (u x u)
    q - Process noise
    l - Vector of observations
    A - Designmatrix
    sigma_ll - Covariance matrix of the observations

    Output
    -----
    x_up - updated state
    sigma_xx_up - covariance matrix of the updated state
    x_p - predicted state
    sigma_xx_p - covariance matrix of the predicted state
    '''
    # Remove NaNs for matrix multiplication
    idx, = np.where(np.isnan(x_k))
    x_k[idx] = 0

    # Prediction
    x_p = T @ x_k 
    sigma_xx_p = T @ sigma_xx_k @ T.T + q * eye_array(len(x_k))

    # Update
    d = l - A @ x_p
    sigma_dd = A @ sigma_xx_p @ A.transpose() + sigma_ll
    
    sigma_dd = csc_matrix(sigma_dd)
    factor = cholesky(sigma_dd)
    b = A @ sigma_xx_p
    b = csc_matrix(b)
    K = factor(b).T

    x_up = x_p + K @ d

    sigma_xx_up = sigma_xx_p - K @ sigma_dd @ K.transpose()
    if issparse(sigma_xx_up):
        sigma_xx_up = CD_matrix_tools.sparsify_matrix(sigma_xx_up, eps_factor)

    # Re-insert the NaNs (so to not to confuse them with elevation=0)
    x_up[idx] = np.nan
    x_p[idx] = np.nan

    return x_up, sigma_xx_up, x_p, sigma_xx_p

def RTS_smoother(T, xup_0, sigma_xx_up_0, xpred_1, sigma_xx_pred_1, xs_1, sigma_xx_s_1, eps_factor):
    '''
    Rauch-Tung-Striebel Smoother (Kalman Smoother, fixed-interval smoother)
    Backward run, requires a forward run with the KF function
    (in a loop, both functions cover one single step in the filter)

    General variable naming:
    '0' refers to the current step ('k' in the equations)
    '1' refers to the previous step of the backwards filter ('k+1' in the equations)
    As it is a backward filter, each iteration goes from k+1 to k.

    Input
    -----
    xup_0            Updated state at time k
    sigma_xx_up_0    Covariance matrix of the update state at time k
    xpred_1          Predicted state at time k+1
    sigma_xx_pred_1  Covariance matrix of the predicted state at time k+1
    xs_1             Smoothed state at time k+1
    sigma_xx_s_1     Covariance matrix of the smoothed state at time k+1

    Output
    -----
    xs_0             Smoothed state at time k
    sigma_xx_s_0     Covariance matrix of the smoothed state at time k
    '''
   # Smoother gain matrix Ks
    try:
        # Solve system with cholesky factorisation if sigma_xx_pred_1 is positive definite
        factor = cholesky(sigma_xx_pred_1)
        b = T @ sigma_xx_up_0
        b = csc_matrix(b)
        Ks = factor(b).T
    except: # CholmodNotPositiveDefiniteError:
        # Use pseudoinverse from SVD instead
        sigma_xx_pred_1_inv = CD_matrix_tools.pseudoinverse(sigma_xx_pred_1)      
        Ks = sigma_xx_up_0 @ T.T @ sigma_xx_pred_1_inv
        Ks = CD_matrix_tools.sparsify_matrix(Ks, eps_factor)
        
    # Remove NaNs for matrix multiplication
    idx, = np.where(np.isnan(xup_0))
    
    xup_0.loc[idx] = 0
    xs_1.loc[idx] = 0
    xpred_1.loc[idx] = 0

    # Smoothed state 
    xs_0 = xup_0 + Ks * (xs_1 - xpred_1)

    # Covariance matrix of the smoothed state
    diff = sigma_xx_s_1 - sigma_xx_pred_1
    sigma_xx_s_0 = sigma_xx_up_0 + Ks @ diff @ Ks.T

    sigma_xx_s_0 = CD_matrix_tools.sparsify_matrix(sigma_xx_s_0, eps_factor)

    # Re-insert the NaNs (so to not to confuse them with elevation=0)
    xs_0.loc[idx] = np.nan

    return xs_0, sigma_xx_s_0
# ===================================================================================
# Designmatrix
# ===================================================================================

def build_designmatrix_nn(x_state, int_pc):
    '''
    Build A-matrix with nearest neighbours.
    The states are the elevations in each grid point.
    The observations are the heights assigned to shoreline points (intertidal point cloud from waterline method) at one point in time.
    For each observation, find the closest grid point with nearest neighbours.
    -> Each row has exactly one 1 (at the closest grid point), the rest are zeros.
    
    Input
    -----
    x_state: GeoDataFrame, one row per grid point, the geometry column contains POINT objects
             with the coordinates of the target grid
    int_pc: Intertidal point cloud (= observation) for one point in time (-> one shoreline observations),
            GeoDataFrame with one shoreline point per row, geometry column with POINT objects
            contains the coordinates of the shoreline points
    '''
    # Number of unknowns and observations
    n_obs = len(int_pc)
    n_unk = len(x_state)
    
    # Coordinates of state vector and observations
    x_coords = get_coordinates(x_state.geometry) # state vector
    int_pc_coords = get_coordinates(int_pc.geometry) # observations

    # Nearest neighbour lookup
    tree = cKDTree(x_coords)
    dist, idx_unk = tree.query(int_pc_coords, k=1) # idx_unk is the column index for A
    
    # Artificial row index
    idx_obs = np.arange(0, n_obs)
    
    A = lil_array((n_obs, n_unk))
    A[idx_obs, idx_unk] = 1
    A = A.tocsr()

    return A

# ===================================================================================
# Transition matrix
# ===================================================================================

def build_transitionmatrix(T_is_identity, x_state, max_distance, w_id):
    '''
    Build transition matrix either as the identity matrix, or with spatial averaging.
    
    Input
    ------
    T_is_identity - boolean, True if T is the identity matrix
        (predicted state = previous updated state), 
        False if T models spatial averaging
    x_state - GeoDataFrame, one row per grid point, the geometry column contains POINT objects
        with the coordinates of the target grid
    max_distance - float, maximum distance to identify neighbouring points
        All points within `max_distance` around the point of iterest (identical point)
        are considered neighbouring points
    w_id - float, weight of the identical point
        (residual weight is equally distributed between neighbouring points)

    Output
    ------
    T - Transition matrix (in scipy sparse format)
    '''
    # Unity matrix
    if T_is_identity:
        T = eye_array(len(x_state))

    # T with spatial averaging
    if not T_is_identity:
        # Find neighbouring points
        x_state_28992 = x_state.to_crs(epsg=28992) # sjoin_nearest requires non-geographic crs
        right = np.array(x_state_28992.geometry)
        row_idx, col_idx = x_state_28992.sindex.query(right, predicate='dwithin', distance=max_distance)
        
        # Norm each line to sum = 1
        rc_idx_df = pd.DataFrame({'row_idx':row_idx, 'col_idx':col_idx})
        count = rc_idx_df.value_counts('row_idx') # number of entries per row
        
        # weight left to distribute among the non-identical points
        w_res = 1 - w_id
        
        # Initialise T
        n = len(x_state)
        T = lil_array((n, n))
        
        # iterate through indices
        for i, df in rc_idx_df.groupby('row_idx'):
            # number of non identical points
            n_nonid = count[i] - 1
            # weight for each of the non-identical points
            w_nonid = w_res/n_nonid
            
            # Set weight for the identical point
            T[i,i] = w_id
            # Identify the column indices that are not the identical point
            col_idx_thisrow = df.loc[df['row_idx'] != df['col_idx'], 'col_idx'].values
            # Set weights of non-identical points
            T[i,col_idx_thisrow] = w_nonid
    return T

# ===================================================================================
# Helper functions
# ===================================================================================

def track_updated_points(updated_points, A, year):
    '''
    Track which points have been updated by an observation.
    The DataFrame 'updated_points" is input and output
        - Has similar structure as x_state with one grid point per row,
          and one year per column
        - Is updated in a loop, where each iteration is one year, and adds one column
        - 0 entry: point was updated, entry 1: point was not updated
        - 'updated_points' additionally contains a column 'counter' that counts how
          many times a point was not updated in a row
          
    Input
    ------
    updated_points - DataFrame, initialise at the beginning of the loop as
        updated_points = pd.DataFrame({'counter':np.full(len(x_state), 0)}, index=range(0,len(x_state)))
    A - Designmatrix
    year - float, year of the iteration

    Output
    ------
    updated_points, with new column for the year of the iteration,
        and counter column updated
    '''
    _, idx_col = A.nonzero() # Find column index where A >= 1
    
    updated_points.loc[:, str(year)] = 1 # 1 everywhere
    updated_points.loc[idx_col, str(year)] = 0
    # Setting nan would remove the counter for this point
    # -> if there is only on observation in the entire period, then this is kept
    
    # Increase counter for how many times in row a point was not updated
    updated_points['counter'] = updated_points['counter'] + updated_points[str(year)]

    return updated_points

def infuse_pseudo_observation(updated_points, max_noupdate, x_state, int_pc):
    '''
    If a grid point was not updated with an observation for max_noupdate times in row
    (as counted by the counter column in updated_points), infuse a pseudo observation
    taken from the previous updated state. Extend the observation vector int_pc
    with the pseudo observation.

    Input
    -----
    updated_points - DataFrame, initialised at beginning of the loop, updated in
        function 'track_updated_points'
    max_noupdate - int, maximum number of times a point can be not updated in a row
        before a pseudo observation is infused
    x_state - GeoDataFrame containing the state vector as an elevation grid with
        one grid point per row
    int_pc - Observation vector, GeoDataFrame with one row per observation point
        (shoreline points assigned with sea level). int_pc is extended
        with the pseudo observation.

    Ouput
    -----
    updated_points - counter column updated with 0 if a pseudo observation was infused
    int_pc - Observation vector, extended with the pseudo observation
    '''
    idx_infuse, = np.where(updated_points['counter'] == max_noupdate)
                
    col_idx_last = x_state.columns.get_loc(x_state.columns[-1]) # idx of last column in x_state
    col_idx_geom = x_state.columns.get_loc('geometry')
    prev_state_infuse = x_state.iloc[idx_infuse, [col_idx_last, col_idx_geom]].dropna() # this .dropna triggers the strange future warning
    prev_state_infuse = prev_state_infuse.rename(columns={x_state.columns[-1]:'ssh', 'geometry':'coords'}).set_geometry('coords')    
    
    int_pc = pd.concat([df for df in [int_pc, prev_state_infuse] if df is not None and not df.empty])

    updated_points.loc[idx_infuse, 'counter'] = 0

    return updated_points, int_pc







