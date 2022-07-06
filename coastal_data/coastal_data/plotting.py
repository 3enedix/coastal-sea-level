import matplotlib.pyplot as plt
from cartopy.feature import LAND
import cartopy.crs as ccrs

def plot_altimetry_tracks(lon, lat, save=False, value=None, title='', label='', extent=None):
    """
    lon: list or array
    lat: list or array
    save: bool
        True: saves the plot as .png
    value : tuple of arrays
        Vector with same length as lon and lat, adds colour-coding
    title: string
    label: string
    extent: dictionary in the form of {'lon_min': x.x, 'lon_max': x.x, 'lat_min': x.x, 'lat_max': x.x}
    """
    
    fig = plt.figure(7495,figsize=(20,10))
    #plt.ticklabel_format(useOffset=False) # want to see the real numbers on the y-axes, not something *8e5
    
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.add_feature(LAND, edgecolor = 'darkgray', facecolor = "lightgray", zorder=1)
    ax.gridlines(draw_labels=True, crs=ccrs.PlateCarree())
    
    if extent != None:
        wnse = list(extent.values())
        ax.set_extent(wnse, crs=ccrs.PlateCarree())
    
    if value is None:
        plot = plt.scatter(lon, lat, marker ='+')
    else:
        plot = plt.scatter(lon, lat, c=value, marker='+')

    if value != None:
        cbar = plt.colorbar(plot, orientation="vertical")
        cbar.ax.tick_params(labelsize=20)
        cbar.set_label(label=label, size=20)
                
    plt.title(title,fontsize=25)    
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    
    if save == True:
        plt.savefig('altimetry_track', transparent=True)
        
def plot_timeseries(x_data, y_data, title='', ylabel='', save=False, sym_y_bounds=True):
    """
    x_data: tuple of arrays
    y_data: dictionary in the form of {'data':(array1, array2, ...), 'legend':('legend1', 'legend2', ...), 'color':('color1', 'color2', ...)}
    title: string
    ylabel: string
    save: bool
        True: saves the plot as .png
    sym_y_bounds: bool
        True: boundaries of y-axis are symmetrical (from -x to +x)
    """
    
    plt.figure(7496, figsize=(15,10))
    all_bounds = [] # determine range of y-axis
    
    for i in range(0, len(y_data['data'])):
        plt.plot(x_data[i], y_data['data'][i], '.-', color=y_data['color'][i], label=y_data['legend'][i])
        #plt.legend(fontsize=18)
        all_bounds.append(max(y_data['data'][i].max(), abs(y_data['data'][i].min())))
    
    plt.grid()
    #plt.legend(fontsize=20, loc='lower right') # upper, lower, center
    plt.legend(fontsize=20, loc='best', bbox_to_anchor=(0.5, -0.05, 0.5, 0.5))    
    #plt.legend(fontsize=20)
    #plt.figlegend(fontsize=20)
    
    plt.title(title,fontsize=25)    
    plt.xticks(fontsize=20);
    plt.yticks(fontsize=20);
    
    plt.xlabel('Time', fontsize=20)
    plt.ylabel(ylabel, fontsize=20)
    
    if sym_y_bounds == True:
        bounds = max(all_bounds) + 0.1*max(all_bounds)
        plt.ylim(top=bounds, bottom=-bounds)
    
    if save == True:
        plt.savefig('timeseries', transparent=True)