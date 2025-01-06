import pandas as pd
import geopandas as gpd
import numpy as np
from shapely import Point

from coastal_data import CD_statistics

import pdb

def get_altimetry_timeseries(alt_data, labels, epsg, gridsize, period_covered):
    '''
    From along-track altimetry data, generate a timeseries.

    Input
    ------
    alt_data - xarray Dataset (from netcdf)
    labels - dict
    epsg - dict
    gridsize - dict
    period_covered - dict

    Output
    ------
    '''

    # Convert data into geodataframe
    alt_df = alt_data.to_dataframe().reset_index()
    alt_gdf = gpd.GeoDataFrame({
                               'ssh':alt_df[labels['ssh']],
                               'mssh':alt_df[labels['mssh']],
                               'sla':alt_df[labels['ssh']] - alt_df[labels['mssh']],
                                },
                           geometry = gpd.points_from_xy(alt_df[labels['lon']], alt_df[labels['lat']]))
    
    alt_gdf = alt_gdf.set_index(pd.to_datetime(alt_df[labels['time']], utc=True))
    
    alt_gdf = alt_gdf.set_crs(epsg['in'])
    alt_gdf = alt_gdf.to_crs(epsg['out'])
    
    # opt: filter out points in certain areas (provide mask)

    # Chessboard binning
    alt_gdf, centers = chessboard_binning(alt_gdf, gridsize)
    alt_gdf = get_distance_to_cell_centers(alt_gdf, centers)

    # Statistics per cell (for now only min and max date of the timeseries)
    cell_numbers = np.sort((alt_gdf['cell'].dropna().unique()))
    cell_stats = pd.DataFrame(np.nan, index=pd.Index(cell_numbers, name='cell'),\
                            columns=['date_min', 'date_max', 'number_values(months)', 'R_psmsl', 'p_psmsl', \
                           'RMSE_psmsl', 'alt_trend', 'psmsl_trend', 'trend_diff'])

    # Attempts to avoid the dtype incompatible future warning, but stays Nattype
    # cell_stats['date_min'] = pd.to_datetime(cell_stats['date_min'])
    # cell_stats['date_max'] = pd.to_datetime(cell_stats['date_max'])

    # cell_stats = cell_stats.astype({'date_max':'datetime64[s]'})
    # cell_stats = cell_stats.astype({'date_min':'datetime64[us]'})
    
    for cell, df in alt_gdf.groupby('cell'):
        # Altimetry timeseries
        df_red = df.copy()
        df_red['sla_red'] = CD_statistics.three_sigma_outlier_rejection(df['sla'])

        cell_stats.loc[cell, ('date_min')] = pd.to_datetime(df_red.index.min())
        cell_stats.loc[cell, ('date_max')] = pd.to_datetime(df_red.index.max())
    
    # Identify cells that cover the minimum period
    date_begin_latest = pd.to_datetime(period_covered['min'], utc=True)
    date_end_earliest = pd.to_datetime(period_covered['max'], utc=True)
    idx_long, = np.where((cell_stats['date_min'] <= date_begin_latest) & (cell_stats['date_max'] >= date_end_earliest))
    # cell_stats_long = cell_stats.iloc[idx_long]
    
    return alt_gdf

def remove_season_and_trend(ts):
    '''
    Remove trend and seasonal signal
    '''
    f = 1 # frequency [years]
    l = ts.dropna()
    t = l.index.year + (l.index.dayofyear - 1)/365.25
    amplitude, phase, trend, offset = CD_statistics.compute_periodic_signal_and_trend(t, l, f)
    model = amplitude * np.sin(2*np.pi*f * t + phase) + trend*t + offset
    l_corr = l - model
    return l_corr

def interpolate_tg_to_alt(alt_index, tg_index, tg_data):
    tg_at_alt_time_interp = np.interp(alt_index, tg_index, tg_data)
    tg_at_alt_time = pd.Series(tg_at_alt_time_interp).set_axis(alt_index)
    return tg_at_alt_time

def regular_vector(data_min, data_max, spacing_desired):
    '''
    Define a regular vector where the distance between two values
    is closest to spacing desired (in [km]).
    '''
    diff = data_max - data_min

    # starting values
    dist = diff
    i = 1 # starting value for number of cells
    eps = 0.1 * spacing_desired * 1000 # [m]
    while dist > spacing_desired * 1000 + eps:
        dist = diff / i
        i = i + 1
    # print('distance between two values [km]:' , dist/1000)

    vec = np.linspace(data_min, data_max, i)
    return vec

def chessboard_binning(alt_gdf, gridsize):
    # cell centers
    centers = gpd.GeoDataFrame(columns=['cell', 'center'], geometry='center')
    centers = centers.set_index('cell')

    # x and y vector
    x_vec = regular_vector(alt_gdf.geometry.x.min(), alt_gdf.geometry.x.max(), gridsize['x'])
    y_vec = regular_vector(alt_gdf.geometry.y.min(), alt_gdf.geometry.y.max(), gridsize['y'])
    
    # Assign a cell number to each row of the altimetry geodataframe
    cell_nr = 1
    for i_x in range(1, len(x_vec)):
        for i_y in range(1, len(y_vec)):
            x1 = x_vec[i_x-1]
            x2 = x_vec[i_x]
            
            y1 = y_vec[i_y-1]
            y2 = y_vec[i_y]
            
            idx = (alt_gdf.geometry.x > x1) & (alt_gdf.geometry.x < x2) \
                & (alt_gdf.geometry.y > y1) & (alt_gdf.geometry.y < y2)

            alt_gdf.loc[idx, 'cell'] = int(cell_nr)
            centers.loc[cell_nr] = Point((x1+x2)/2, (y1+y2)/2)
        
            cell_nr = cell_nr + 1
        
    return alt_gdf, centers

def get_distance_to_cell_centers(alt_gdf, centers):
    dist = np.zeros(len(alt_gdf))
    for i in range(0, len(alt_gdf)):
        cell_nr = alt_gdf.loc[alt_gdf.index[i], 'cell']
        if np.isnan(cell_nr):
            dist[i] = np.nan
            continue
        dist[i] = alt_gdf.geometry.iloc[i].distance(centers.loc[cell_nr].center)
    
    alt_gdf.loc[:,'dist2center'] = dist
    
    return alt_gdf









