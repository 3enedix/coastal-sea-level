from shapely import get_coordinates
from scipy.spatial import cKDTree
from scipy.sparse import lil_array, eye_array, csc_matrix, issparse, diags_array
from sksparse.cholmod import cholesky
from sksparse.cholmod import CholmodNotPositiveDefiniteError

import numpy as np
import pandas as pd
import time
import pickle
import xarray as xr
import geopandas as gpd
import rasterio
from collections import OrderedDict

from geocube.api.core import make_geocube

from coastal_data import CD_matrix_tools, CD_geometry, CD_combine_data, CD_jarkus_tools

import pdb

# ===================================================================================
# Entire forward/backward run
# ===================================================================================
def forward_run(year_start, year_end, init, std_init, rs_shoreline, seg_length, alt, tidal_corr,
                  epsg, T_is_identity, max_distance, w_id, max_noupdate, std_pseudobs,
                  sigma_l, sigma_q, eps_factor):
    start_time = time.perf_counter()

    # Initialise state vector
    init_df = init.to_dataframe()
    x = init_df.index.get_level_values('x')
    y = init_df.index.get_level_values('y')    
    x_state = gpd.GeoDataFrame({'init': init_df.band_data.values}, geometry=gpd.points_from_xy(x, y))
    x_state = x_state.set_crs(epsg)

    # Covariance matrix of the initial state
    sigma_xx_init_df = gpd.GeoDataFrame({'sigma': std_init**2}, geometry=gpd.points_from_xy(x, y), index=x_state.index)
    sigma_xx_init = diags_array(sigma_xx_init_df.sigma)
    # sigma_xx_init = np.diag(sigma_xx_init_df.sigma) # dense array format

    # "Iteration 0"
    x_k = x_state['init'].copy()
    sigma_xx_k = sigma_xx_init

    # Transition matrix
    T = build_transitionmatrix(T_is_identity, x_state, max_distance, w_id)

    # Initialise updated points, and dicts to save predicted and updated covariance matrices
    updated_points = pd.DataFrame({'counter': np.full(len(x_state), 0)}, index=range(0, len(x_state)))
    sigma_xx_up, sigma_xx_pred = {}, {}    

    for i, year in enumerate(range(year_start, year_end+1)):      
        startdate = str(year) + '-01-01' # '1993-01-01'
        enddate = str(year+1) + '-01-01' # '1994-01-01'

        # Observations
        int_pc = CD_combine_data.waterline_method_period(rs_shoreline, seg_length, alt, tidal_corr['corr_eot[cm]'], startdate, enddate)
        if int_pc is not None:
            int_pc = int_pc.set_crs(epsg)
            n_obs = len(int_pc)
        else: # no shorelines in this period
            continue

        # For T with spatial averaging, infuse a pseudo observation if point was not
        # updated for the last max_noupdate iterations        
        if (not T_is_identity) and (year > year_start):
            updated_points, int_pc = infuse_pseudo_observation(updated_points, max_noupdate, x_state, int_pc)

        # Observation vector and design matrix
        l = int_pc['ssh']
        A = build_designmatrix_nn(x_state, int_pc)
        updated_points = track_updated_points(updated_points, A, year)

        # sigma_ll: Covariance matrix of the observations
        if (not T_is_identity) and (year > year_start):
            # Combine variances of observations and pseudo-observations
            vars_comb = np.concatenate([np.ones(n_obs) * sigma_l, np.ones(len(int_pc)-n_obs) * std_pseudobs**2])
            sigma_ll = diags_array(vars_comb)
        else:
            sigma_ll = eye_array(n_obs) * sigma_l

        # Forward KF
        x_up_temp, sigma_xx_up_temp, x_pred_temp, sigma_xx_pred_temp = KF(x_k, sigma_xx_k, T, sigma_q, l, A, sigma_ll, eps_factor)

        # Overwrite for next iteration
        x_k = x_up_temp
        sigma_xx_k = sigma_xx_up_temp 

        # Save predicted and updated state and covariance matrix
        x_state['x_pred_'+str(year)] = x_pred_temp
        sigma_xx_pred[str(year)] = csc_matrix(sigma_xx_pred_temp) 

        x_state['x_up_'+str(year)] = x_up_temp
        sigma_xx_up[str(year)] = csc_matrix(sigma_xx_up_temp)

    end_time = time.perf_counter()
    ex_time = end_time - start_time
    print("Forward run done. Needed ", round(ex_time,1), "seconds.")

    return x_state, sigma_xx_up, sigma_xx_pred, updated_points, T

def backward_run(x_state, sigma_xx_up, sigma_xx_pred, year_end, T, eps_factor):
    start_time = time.perf_counter()
    x_state_s = x_state[['geometry', 'x_up_'+str(year_end)]].copy()
    x_state_s = x_state_s.rename(columns={'x_up_'+str(year_end) : 'x_s_'+str(year_end)})

    # Initialise covariance matrix of the smoothed state with covMatrix of the updated state
    sigma_xx_s = OrderedDict()
    sigma_xx_s[str(year_end)] = sigma_xx_up[str(year_end)]

    # Find out for which years the KF function yielded results (2012 is missing)
    col_idx = ['x_up' in _ for _ in x_state.columns]
    str_list = x_state.loc[:,col_idx].columns.to_list()
    year_list = [int(_.replace('x_up_', '')) for _ in str_list]

    for i in range(len(year_list)-1, 0, -1):
        x_state_s['x_s_'+str(year_list[i-1])], sigma_xx_s[str(year_list[i-1])] = \
            RTS_smoother(T=T,
                         xup_0=x_state['x_up_'+str(year_list[i-1])].copy(),
                         sigma_xx_up_0=sigma_xx_up[str(year_list[i-1])].copy(),
                         xpred_1=x_state['x_pred_'+str(year_list[i])].copy(),
                         sigma_xx_pred_1=sigma_xx_pred[str(year_list[i])].copy(),
                         xs_1=x_state_s.iloc[:,-1].copy(),
                         sigma_xx_s_1=list(sigma_xx_s.items())[-1][1].copy(),
                         eps_factor=eps_factor)
    end_time = time.perf_counter()
    ex_time = end_time - start_time
    print("Backward run done. Needed ", round(ex_time,1), "seconds.")
    
    return x_state_s, sigma_xx_s

# ===================================================================================
# Initial state
# ===================================================================================

def initial_state(epsg, buffer_size, smooth_factor, resolution, c_shorelines, alt, path_input, path_output, fn_init, fn_gebco, fn_ddtm):
    '''
    No return, saves result as TIF in path_output.
    '''
    start_time = time.process_time()
    # 1. Create target grid
    if epsg == 4326:
        buffer_size = CD_geometry.dist_meter_to_dist_deg(buffer_size) # [deg]
        resolution = CD_geometry.dist_meter_to_dist_deg(resolution) # [deg]
    
    # alpha (how much slack around the concave hull)
    if epsg == 4326:
        alpha = 100
    elif epsg == 28992:
        alpha = 3e-4    

    # Polgon around the shorelines
    poly_shorelines = CD_geometry.get_area_covered_by_shorelines(c_shorelines, alpha=alpha, buffer_size=0)
    # # Polygon of target area (shoreline polygon with buffer)
    # poly_buffered = poly_shorelines.buffer(buffer_size)

    # Buffer around median shoreline 
    med_sl = CD_geometry.median_shoreline_from_transect_intersections(c_shorelines, spacing=100, transect_length=5000, smooth_factor=smooth_factor)
    poly_buffered = med_sl.buffer(buffer_size)

    # Buffer around zero-contour
    # poly_buffered = zcontour.buffer(buffer_size)
    
    # Grid inside the target polygon
    x_poly, y_poly, x_full, y_full = CD_geometry.create_target_grid(poly_buffered, resolution=resolution)
    
    pickle.dump(med_sl, open(path_output + f'med_sl{epsg}.pkl', 'wb'))
    pickle.dump(poly_buffered, open(path_output + f'poly_buffered_{epsg}.pkl', 'wb'))
    pickle.dump(poly_shorelines, open(path_output + f'poly_sl_{epsg}.pkl', 'wb'))
    pickle.dump(zip(x_poly, y_poly), open(path_output+f'xy_poly_{epsg}.pkl', 'wb'))

    # 2. Get global DEMS
    fn = 'gebco_2023_n53.5439_s53.2693_w5.0194_e5.6607.nc'
    gebco = xr.open_dataset(path_input+fn_gebco)
    fn = 'DeltaDTM_v1_0_N53E005.tif'
    ddtm = xr.open_dataset(path_input+fn_ddtm).squeeze()

    if epsg != 4326:
        poly_4326 = CD_geometry.transform_polygon(poly_buffered, epsg, 4326)
    else:
        poly_4326 = poly_buffered

    # If kernel dies, there are too many points for 'gpd.points_from_xy(dem_df.x, dem_df.y)'
    # Shutting all other kernels can help
    ddtm_gdf = CD_geometry.cut_DEM_to_target_area(ddtm, 'band_data', poly_4326, 'ddtm')
    gebco_gdf = CD_geometry.cut_DEM_to_target_area(gebco, 'elevation', poly_4326, 'gebco')

    del ddtm, gebco
    
    if epsg != 4326:
        gebco_gdf = gebco_gdf.to_crs(epsg)
        ddtm_gdf = ddtm_gdf.to_crs(epsg)

    # 3. Interpolate global DEMs individually
    gebco_interp = CD_geometry.interpolate_dem(gebco_gdf, x_poly, y_poly)
    ddtm_interp = CD_geometry.interpolate_dem(ddtm_gdf, x_poly, y_poly)

    del ddtm_gdf, gebco_gdf

    # 4. Remove bias between GEBCO and DeltaDTM
    # Find where GEBCO and DeltaDTM intersect
    inter = gpd.sjoin(gebco_interp, ddtm_interp, how='inner', predicate='intersects', lsuffix='gebco', rsuffix='ddtm')

    diff = inter.elevation_gebco - inter.elevation_ddtm
    bias = diff.mean()
    print(f'Bias GEBCO - DeltaDTM: {round(bias,2)} m')
    
    # Remove the bias from GEBCO (and overwrite dataframe variable)
    gebco_interp.elevation = gebco_interp.elevation - bias

    # 5. Remove overlapping points and piece together
    idx_geb_overlap = gebco_interp.index.isin(inter.index_ddtm)
    gebco_noverlap = gebco_interp[~idx_geb_overlap]
    
    comb_interp = pd.concat([ddtm_interp, gebco_noverlap])

    del gebco_interp, ddtm_interp

    # 6. [Smooth] removed, see notebook 52_initial_state for the code if spatial smoothing desired

    # 7. De-bias with altimetry
    
    # Get elevations around the median shoreline
    # Buffer around median shoreline with the resolution of the target grid
    med_sl_buffer = med_sl.buffer(resolution)
    # Get DEM in the shoreline buffer zone
    dem_in_slzone = comb_interp[comb_interp.intersects(med_sl_buffer)]

    # Compute bias
    msl = np.nanmean(alt.values)
    diff = dem_in_slzone.elevation - msl

    # Remove bias
    comb_interp.elevation = comb_interp.elevation - diff.mean()

    # 8. Export as TIF
    comb_interp = comb_interp.set_crs(epsg)
    cube = make_geocube(vector_data=comb_interp,
                     measurements=['elevation'],
                     resolution=(-resolution,resolution))    
    cube.rio.to_raster(path_output + fn_init)
    # metadata
    with open (path_output+fn_init+'.txt', 'w') as f:
            f.write(f'Initial state as combination of GEBCO and DeltaDTM. Created on {time.strftime("%Y-%m-%d_%H:%M:%S")}')
    end_time = time.process_time()
    ex_time = end_time - start_time
    print("Done. Needed ", round(ex_time/60,2), "minutes.")

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

def interpolate_results(jarkus_gdf, x_state, x_state_s, updated_points):
    '''
    Interpolate results from forward Kalman filter and backward RTS smoother
    onto the JARKUS grid.
    '''
    start_time = time.perf_counter()
    kf_interp = gpd.GeoDataFrame(
        {'geometry':gpd.points_from_xy(jarkus_gdf.geometry.x, jarkus_gdf.geometry.y)}).set_crs('4326')
    rts_interp = gpd.GeoDataFrame(
        {'geometry':gpd.points_from_xy(jarkus_gdf.geometry.x, jarkus_gdf.geometry.y)}).set_crs('4326')

    # kf_interp_up = gpd.GeoDataFrame(
    #     {'geometry':gpd.points_from_xy(jarkus_gdf.geometry.x, jarkus_gdf.geometry.y)}).set_crs('4326')
    # rts_interp_up = gpd.GeoDataFrame(
    #     {'geometry':gpd.points_from_xy(jarkus_gdf.geometry.x, jarkus_gdf.geometry.y)}).set_crs('4326')

    year_list = [int(col.split('_')[-1]) for col in x_state.columns if col.startswith('x_up_')]
    for year in year_list:
        kf_interp[str(year)] = CD_jarkus_tools.interpolate_irregular_grid_onto_JARKUS(
                            x_state, f'x_up_{year}', jarkus_gdf.geometry.x, jarkus_gdf.geometry.y)
        rts_interp[str(year)] = CD_jarkus_tools.interpolate_irregular_grid_onto_JARKUS(
                            x_state_s, f'x_s_{year}', jarkus_gdf.geometry.x, jarkus_gdf.geometry.y)

        # following part looks wrong in the plots, compared to kf/rts_interp
        # # only updated points
        # idx_update, = np.where(updated_points[str(year)] == 0)
        # poly_up = CD_geometry.concave_hull_alpha_shape(x_state.iloc[idx_update].geometry, alpha=100)
        # poly_up = CD_geometry.transform_polygon(poly_up, 4326, 28992)

        # kf_interp_up[str(year)] = CD_jarkus_tools.interpolate_irregular_grid_onto_JARKUS(
        #     x_state.iloc[idx_update], f'x_up_{year}', jarkus_gdf.geometry.x, jarkus_gdf.geometry.y)    
        # kf_interp_up.loc[~kf_interp_up.intersects(poly_up), year] = np.nan # set extrapolated points outside the polygon to NaN

        # rts_interp_up[str(year)] = CD_jarkus_tools.interpolate_irregular_grid_onto_JARKUS(
        #     x_state_s.iloc[idx_update], f'x_s_{year}', jarkus_gdf.geometry.x, jarkus_gdf.geometry.y)
        # rts_interp_up.loc[~rts_interp_up.intersects(poly_up), year] = np.nan # set extrapolated points outside the polygon to NaN
    end_time = time.perf_counter()
    ex_time = end_time - start_time
    print("Interpolation done. Needed ", round(ex_time,1), "seconds.")
    
    return kf_interp, rts_interp #, kf_interp_up, rts_interp_up





