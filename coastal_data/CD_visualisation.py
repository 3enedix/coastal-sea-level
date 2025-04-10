import pandas as pd
import numpy as np
from coastal_data import CD_statistics

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt

import tabulate

def movie(df, x_poly, y_poly, col_list, first_col, savepath, fn):
    '''
    Create and save animation of Kalman filter or RTS smoother results.
    '''    
    # plt.rcParams["animation.html"] = "jshtml"
    plt.rcParams['figure.dpi'] = 150
    plt.rcParams['animation.embed_limit'] = 100 # MB
    plt.rcParams['animation.ffmpeg_path'] = '/snap/bin/ffmpeg'

    # Set up the usual map
    projection = ccrs.PlateCarree()
    request = cimgt.GoogleTiles(style="satellite")
    fig, ax = plt.subplots(figsize=(15,8), subplot_kw=dict(projection=request.crs));
    ax.set_extent([5.1, 5.6, 53.32, 53.5], crs=ccrs.PlateCarree())
    ax.gridlines(draw_labels=True, zorder=0, color='lightgrey')
    zoom = 10
    ax.add_image(request, zoom, alpha=0.7, zorder=0)
    
    # Initial state
    scat = ax.scatter(x_poly, y_poly, marker='.', s=200, c=df[first_col],
                      transform=ccrs.PlateCarree(), cmap='BrBG_r', vmin=-5, vmax=5)
    title = ax.set_title(first_col)
    fig.colorbar(scat, shrink=1, pad=.12, label='[m]')
    
    # Update function
    def update(frame):
        scat.set_array(df.loc[:,frame])
        title.set_text(frame)
    
    # Animate
    anim = animation.FuncAnimation(fig=fig, func=update, frames=col_list, interval=500)
    anim.save(filename=savepath+fn, writer="pillow")

    return anim

def show_diff_table(rts_data, jarkus_years, fn=None, path=None):
    print_result = False
    stats_list = []
    all_diffs = {}
    for year in year_list:
        idx_year, = np.where(jarkus_years == year)
        diff = rts_data[str(year)] - jarkus_elev[idx_year, :][0]
        # all_diffs[year] = diff # for aggregate histogram
        
        me_kf = CD_statistics.compute_mean_error(diff, print_result)
        mae_kf = CD_statistics.compute_mean_absolute_error(diff, print_result)
        rmse_kf = CD_statistics.compute_rmse(diff, print_result)
        mad_mean_kf = CD_statistics.compute_mad_mean(diff, print_result)
        mad_med_kf = CD_statistics.compute_mad_med(diff, print_result)
    
        stats_list.append([year, me_kf, mae_kf, rmse_kf, mad_mean_kf, mad_med_kf])
    print(tabulate.tabulate(stats_list, headers=['Year', 'ME [m]', 'MAE [m]', 'RMSE [m]', 'MAD (mean) [m]', 'MAD (median) [m]'], tablefmt="fancy_grid"))
    # tablefmt="latex", documentation at https://pypi.org/project/tabulate/

    if path:
        # Save the print output from tabulate to a text file
        with open(f'{savepath_partun}{fn}.txt', 'w') as f:
            f.write(tabulate.tabulate(stats_list, headers=['Year', 'ME [m]', 'MAE [m]', 'RMSE [m]', 'MAD (mean) [m]', 'MAD (median) [m]'], tablefmt="fancy_grid"))

def plot_histogram(ax, all_diffs, title):
    all_values = np.concatenate([_ for _ in all_diffs.values()])
    ax.hist(all_values, bins=50, range=[-25,25])
    ax.set_title(title)
    ax.set_xlabel('Diff [m]')
    ax.set_ylabel('#')
    ax.grid() 

def plot_map(epsg, x, y, data, vminmax=None, cmap='BrBG_r', title=''):
    if epsg == 28992:
        # EPSG:28992 Amersfoort / RD New
        crs = ccrs.epsg('28992')
    elif epsg == 4326:
        # EPSG:4326 WGS 84
        crs = ccrs.PlateCarree()
    else:
        raise ValueError("Use EPSG code 28992 or 4326.")
    
    # request = cimgt.OSM()
    request = cimgt.GoogleTiles(style="satellite")
    fig, ax = plt.subplots(figsize=(15,8), subplot_kw=dict(projection=request.crs))
    ax.set_extent([5.1, 5.6, 53.32, 53.5], crs=ccrs.PlateCarree())
    ax.gridlines(draw_labels=True, zorder=0, color='lightgrey')
    zoom = 10
    ax.add_image(request, zoom, alpha=0.7, zorder=0)

    if vminmax is None:
        vminmax = [np.nanmin(data), np.nanmax(data)]
    
    plot = ax.scatter(x, y, c=data, marker='.', s=2, transform=crs, cmap=cmap, vmin=vminmax[0], vmax=vminmax[1])
    
    plt.title(title)
    plt.colorbar(plot, shrink=.6, label='[m]', pad=.15)









        