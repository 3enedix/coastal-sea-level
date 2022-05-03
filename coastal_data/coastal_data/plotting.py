import matplotlib.pyplot as plt
from cartopy.feature import LAND
import cartopy.crs as ccrs

def plot_altimetry_tracks(lon, lat, save=False, value=None, title='', extent=None):
    """
    lon: array
    lat: array
    save: bool
        True: saves the plot as .png
    value : array
        Vector with same length as lon and lat, adds colour-coding
    title: string
    extent: dictionary in the form of {'lon_min': x.x, 'lon_max': x.x, 'lat_min': x.x, 'lat_max': x.x}
    """
    
    fig = plt.figure(1,figsize=(15,15))
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
        plot = plt.scatter(lon, lat, c = value, marker ='+')
        cbar = plt.colorbar(plot, orientation="vertical")
        cbar.ax.tick_params(labelsize=20)
        cbar.set_label(label='[m]',size=20)
    
    plt.title(title,fontsize=25)    
    plt.xticks(fontsize=20); # produces weird array output without ;
    plt.yticks(fontsize=20);
    
    if save == True:
        plt.savefig('altimetry_track', transparent=True)