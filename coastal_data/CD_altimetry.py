import pandas as pd
import geopandas as gpd
import numpy as np
from shapely import Point
from scipy import stats

import matplotlib.pyplot as plt

from cartopy.feature import LAND
import cartopy.crs as ccrs


from coastal_data import CD_statistics, CD_helper_functions

import pdb

# matplotlib fontsizes
SMALL_SIZE = 15
MEDIUM_SIZE = 20
BIGGER_SIZE = 25
plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)    # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)   # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)   # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)   # legend fontsize
plt.rc('figure', titlesize=MEDIUM_SIZE)  # fontsize of the figure title

def get_altimetry_timeseries_with_TG(alt_data, labels, epsg, tg, gridsize, period_covered, temp_average='ME'):
    '''
    Create a timeseries from along-track altimetry data
    that fits best to a nearby tide gauge
    in terms of correlation, RMSE and trend difference.

    Input
    ------
    alt_data - xarray Dataset (from netcdf)
    labels - dict
    epsg - dict
    tg - pandas Series of corrected tide gauge data (corrected for IB and VLM) with DatetimeIndex
    gridsize - dict
    period_covered - dict
    temp_average - string, period over which to average in pd.Grouper
    (see https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases)

    Output
    ------
    alt_ex_temp_av - pandas DataFrame with columns time, sla_temp_av, ssh_temp_av and msss_temp_av
    '''
    plt.close('all')
    
    alt_gdf = gpd.GeoDataFrame({
                           'ssh':alt_data[labels['ssh']],
                           'mssh':alt_data[labels['mssh']],
                           'sla':alt_data[labels['ssh']] - alt_data[labels['mssh']],
                            },
                       geometry = gpd.points_from_xy(alt_data[labels['lon']], alt_data[labels['lat']]))    
    alt_gdf = alt_gdf.set_index(pd.to_datetime(alt_data[labels['time']], utc=True))
    
    alt_gdf = alt_gdf.set_crs(epsg['in'])
    alt_gdf = alt_gdf.to_crs(epsg['out'])
    
    # opt: filter out points in certain areas (provide mask)

    # Chessboard binning
    alt_gdf, centers, x_vec, y_vec = chessboard_binning(alt_gdf, gridsize)
    alt_gdf = get_distance_to_cell_centers(alt_gdf, centers) # is there another way to add a single column that is easier on the memory? Does this store alt_gdf 2 times?

    # Statistics per cell
    cell_numbers = np.sort((alt_gdf['cell'].dropna().unique()))
    cell_stats = pd.DataFrame(np.nan, index=pd.Index(cell_numbers, name='cell'),
                            columns=['date_min', 'date_max', 'number_values(months)', 'R', 'p',
                                     'RMSE', 'alt_trend', 'tg_trend', 'trend_diff'])

    # Attempts to avoid the dtype incompatible future warning, but stays Nattype
    # cell_stats['date_min'] = pd.to_datetime(cell_stats['date_min'])
    # cell_stats['date_max'] = pd.to_datetime(cell_stats['date_max'])

    # cell_stats = cell_stats.astype({'date_max':'datetime64[s]'})
    # cell_stats = cell_stats.astype({'date_min':'datetime64[us]'})
    
    for cell, df in alt_gdf.groupby('cell'):
        # Get one altimetry timeseries per cell
        df_red = df.copy()
        df_red['sla_red'] = CD_statistics.three_sigma_outlier_rejection(df['sla'])
        # df_red_des_det = remove_season_and_trend(df_red)

        # Temporal averages weighted with inverse distance to cell
        df_temp_av = pd.DataFrame()
        for time, df_period in df_red.groupby(pd.Grouper(freq=temp_average)):
            if len(df_period) > 0:
                weights = 1/df_period['dist2center']
                temp_av = (df_period['sla_red'] * weights).sum()/weights.sum()
            else:
                continue
            df_temp_av_temp = pd.DataFrame({'temp_av':temp_av}, index=[time])
            if not df_temp_av_temp.empty:
                df_temp_av = pd.concat([df_temp_av, df_temp_av_temp])
                
        # Comparison statistics between altimetry and tide gauge
        if len(df_temp_av) >= 5:
            fill_cell_stats(cell_stats, cell, tg, df_temp_av, df_red, temp_average)
        
            # trends altimetry
            x_alt = CD_helper_functions.datetime_to_decimal_numbers(df_temp_av['temp_av'].index)
            trend_alt = CD_statistics.compute_trend(x_alt, df_temp_av['temp_av'].values*1000) # [mm/year]
        
            # trend PSMSL for same time period
            idx_tg_in_alt_period = np.where((tg.index > df_temp_av['temp_av'].index[0]) \
                                   & (tg.index < df_temp_av['temp_av'].index[-1]))

            if len(idx_tg_in_alt_period[0]) >= 5:
                tg_red_to_alt = tg.iloc[idx_tg_in_alt_period]
                x_tg = CD_helper_functions.datetime_to_decimal_numbers(tg_red_to_alt.index)

                trend_tg = CD_statistics.compute_trend(x_tg, tg_red_to_alt.values*10)
            else:
                trend_tg = np.nan
        else:
            trend_alt = np.nan
        
        cell_stats.loc[cell, ('alt_trend')] = trend_alt
        cell_stats.loc[cell, ('tg_trend')] = trend_tg
        cell_stats.loc[cell, ('trend_diff')] = trend_alt - trend_tg
        cell_stats.loc[cell, ('date_min')] = pd.to_datetime(df_red.index.min())
        cell_stats.loc[cell, ('date_max')] = pd.to_datetime(df_red.index.max())
        cell_stats.loc[cell, ('number_values(months)')] = len(df_temp_av.index)
   
    # Identify cells that cover the minimum period
    date_begin_latest = pd.to_datetime(period_covered['min'], utc=True)
    date_end_earliest = pd.to_datetime(period_covered['max'], utc=True)
    idx_long, = np.where((cell_stats['date_min'] <= date_begin_latest) & (cell_stats['date_max'] >= date_end_earliest))
    cell_stats_long = cell_stats.iloc[idx_long]

    idx_long_in_altgdf = np.where(alt_gdf.cell.isin(cell_stats.iloc[idx_long].index))
    alt_gdf_long = alt_gdf.iloc[idx_long_in_altgdf]

    
    # List 10 cells with highest R, lowest RMSE and smallest trend difference
    nr = 10 # show the x best candidates
    print(cell_stats['R'].iloc[idx_long].sort_values().iloc[-nr:])
    print(cell_stats['RMSE'].iloc[idx_long].sort_values().iloc[:nr])
    print(np.abs(cell_stats['trend_diff'].iloc[idx_long]).sort_values().iloc[:nr])

    # Maps of correlation, RMSE and trend difference
    fig1, ax1 = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()}, figsize=(10,10))
    plot_map(ax1, centers, cell_stats, cell_stats_long, 'R', x_vec, y_vec, alt_gdf, vmin=0, vmax=1, cmap='YlGn', title='Correlation', label='R')
    fig2, ax2 = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()}, figsize=(10,10))
    plot_map(ax2, centers, cell_stats, cell_stats_long, 'RMSE', x_vec, y_vec, alt_gdf, vmin=0.1, vmax=0.3, cmap='YlOrBr', title='RMSE', label='RMSE [m]')
    fig3, ax3 = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()}, figsize=(10,10))
    plot_map(ax3, centers, cell_stats, cell_stats_long, 'trend_diff', x_vec, y_vec, alt_gdf, vmin=-5.0, vmax=5.0, cmap='bwr', title='Trend difference', label='Trend difference [mm/year]')

    # Get user input which cells to extract
    cell_ex = list(map(int, input("Enter cell numbers to extract, separated by space (e.g. 96 95 83 82 70 69): ").split()))

    # Get cells from user input
    idx_ex = np.empty(0)
    for cell in cell_ex:
        idx_temp, = np.where(alt_gdf['cell'] == cell)
        idx_ex = np.hstack([idx_ex,idx_temp])
    
    alt_gdf_ex = alt_gdf.iloc[idx_ex]

    # Extract timeseries of temporal averages
    # Average over all cells if there is no other quality criterion (no tide gauge comparison):
    df_red_ex = alt_gdf_ex.copy()
    df_red_ex['sla_red'] = CD_statistics.three_sigma_outlier_rejection(df_red_ex['sla'])
    df_red_ex['ssh_red'] = CD_statistics.three_sigma_outlier_rejection(df_red_ex['ssh'])
    
    # Temporal averages weighted with inverse distance to cell
    alt_ex_temp_av = pd.DataFrame()
    for time, df_month in df_red_ex.groupby(pd.Grouper(freq=temp_average)):
        weights = 1/df_month['dist2center']
        sla_temp_av = (df_month['sla_red'] * weights).sum()/weights.sum()
        ssh_temp_av = (df_month['ssh_red'] * weights).sum()/weights.sum()
        mss_temp_av = (df_month['mssh'] * weights).sum()/weights.sum()
        alt_ex_temp_av_temp = pd.DataFrame({'time': time,
                                           'sla_temp_av':sla_temp_av,
                                           'ssh_temp_av':ssh_temp_av,
                                           'msss_temp_av':mss_temp_av
                                              }, index=[time]).set_index('time')
        alt_ex_temp_av = pd.concat([alt_ex_temp_av, alt_ex_temp_av_temp])
    
    return alt_ex_temp_av

# def remove_season_and_trend(ts):
#     '''
#     Remove trend and seasonal signal
#     '''
#     f = 1 # frequency [years]
#     l = ts.dropna()
#     t = l.index.year + (l.index.dayofyear - 1)/365.25
#     amplitude, phase, trend, offset = CD_statistics.compute_periodic_signal_and_trend(t, l, f)
#     model = amplitude * np.sin(2*np.pi*f * t + phase) + trend*t + offset
#     l_corr = l - model
#     return l_corr

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
        
    return alt_gdf, centers, x_vec, y_vec

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

def compute_RMSE(sla, tg):
    sla_red = sla - np.nanmean(sla)
    tg_red = tg - np.nanmean(tg)
    rmse = np.sqrt(((sla_red - tg_red) **2).sum()/(len(sla_red)-1))    
    return rmse

def fill_cell_stats(cell_stats, cell, tg_data, df_temp_av, df_red, temp_average):
    # bring tide gauge data to the same time stamps
    tg_at_alt_time = interpolate_tg_to_alt(df_red.index, tg_data.index, tg_data)
    tg_mon = tg_at_alt_time.groupby(pd.Grouper(freq=temp_average)).mean()
    # remove values in tg_mon where month/year combination is not in df_temp_av
    tg_mon = tg_mon.loc[tg_mon.index.isin(df_temp_av.index)]
    
    # remove rows that contain NaN in tg_mon
    idx = np.where(np.isnan(tg_mon))[0]    
    df_temp_av = df_temp_av.drop(tg_mon.index[idx])
    tg_mon = tg_mon.drop(tg_mon.index[idx])
    
    # correlation
    R_temp, p_temp = stats.pearsonr(df_temp_av['temp_av'].values, tg_mon.values)
    cell_stats.loc[cell, ('R')] = R_temp
    cell_stats.loc[cell, ('p')] = p_temp
    
    # RMSE
    cell_stats.loc[cell, ('RMSE')] = compute_RMSE(df_temp_av['temp_av'], tg_mon/100)

def reformat_chess_data(cell_stats, param, x_vec, y_vec):
    '''
    cell_stats
    param, e.g. 'R'
         
    This function does 2 things:
        1. Adding nans for the cells which don't contain data
        2. Reshapes the values in a way that plotting with pcolormesh colors the right fields on the map.
    '''
    cells_full = np.arange((len(y_vec)-1) * (len(x_vec)-1)) + 1
    empty_cells = [_ for _ in cells_full if _ not in cell_stats.index]
    empty_cells = np.asarray(empty_cells)
    empty_cells_idx = empty_cells -1
    
    chess_stats_full = cell_stats[param].values
    for idx in empty_cells_idx:
        chess_stats_full = np.insert(chess_stats_full, idx, np.nan)

    chess_stats_full_re = np.reshape(chess_stats_full, ((len(x_vec)-1,len(y_vec)-1)))
    chess_stats_full_re = np.transpose(chess_stats_full_re)
    
    return chess_stats_full_re

def plot_map(ax, centers, cell_stats, cell_stats_long, param, x_vec, y_vec, alt_gdf, vmin=None, vmax=None, cmap='Blues', title='', label=''):
    '''
    first create fig and ax like:
        fig = plt.figure(figsize=(12,12))
        ax = plt.axes(projection=ccrs.PlateCarree())
    '''
    plt.ion()
    manager = plt.get_current_fig_manager()
    manager.window.showMaximized()   
    
    data_re = reformat_chess_data(cell_stats, param, x_vec, y_vec)
    crs_32631 = ccrs.epsg('32631')    
    ax.gridlines(draw_labels=True, crs=ccrs.PlateCarree())
    ax.add_feature(LAND, edgecolor = 'darkgray', facecolor = "lightgray", zorder=2)
    plot = ax.pcolormesh(x_vec, y_vec, data_re, transform=crs_32631, cmap=cmap, vmin = vmin, vmax = vmax)
    ax.scatter(alt_gdf.geometry.x, alt_gdf.geometry.y, marker='+', s=1, c='darkgrey', transform=crs_32631)
    
    # plot cell numbers
    for cell_nr in centers.index:
        if cell_nr in cell_stats_long.index:
            text = ax.text(centers.loc[cell_nr, 'center'].x-5000,centers.loc[cell_nr, 'center'].y-5000,str(cell_nr), fontsize=18,
                    transform=crs_32631, bbox=dict(boxstyle="square", fill=False))
        else:
            text = ax.text(centers.loc[cell_nr, 'center'].x-5000,centers.loc[cell_nr, 'center'].y-5000,str(cell_nr), fontsize=18,
                    transform=crs_32631)

    ax.set_title(title)
    ax.text(0, -0.1, 'Press any button to continue.', transform=ax.transAxes)

    plt.colorbar(plot, shrink=1, pad=0.1, label=label, ax=ax)
    plt.draw()
    plt.pause(0.01)
    plt.waitforbuttonpress()
    plt.ioff()




