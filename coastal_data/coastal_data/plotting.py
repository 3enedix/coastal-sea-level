import matplotlib.pyplot as plt
from cartopy.feature import LAND
import cartopy.crs as ccrs

def plot_altimetry_tracks(lon, lat, save=False, *args):
    # TODO:
    # - fix colobar to show the real values and not something *8e2
    # - include extent
    # - documentation
    # - better optional arguments (probably something like if len(args) == 3)
    # - how to use the argument name in the function call? Like plotting.plot_altimetry_tracks(lon=ds.lon_20_ku, lat=ds.lat_20_ku, save=True, value=ds.range_ales_20_ku, title='test') to increase readability of code
    # (also apply learned things to the prepare_S3_data function)
    
    value = args[0]
    title = args[1]
    
    fig = plt.figure(1,figsize=(15,15))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.add_feature(LAND, edgecolor = 'darkgray', facecolor = "lightgray", zorder=1)
    
    #wnse=[3,7,56,53]
    #wnse=[5,6,54,53]
    #ax.set_extent(wnse, crs=ccrs.PlateCarree())
    
    plot = plt.scatter(lon, lat, c = value/1000, marker ='+') # /1000 to get values in [km]
    
    cbar = plt.colorbar(plot, orientation="vertical")
    cbar.ax.tick_params(labelsize=20)
    cbar.set_label(label='[km]',size=20)
    
    plt.title(title,fontsize=25)
    
    plt.xticks(fontsize=20); # produces weird array output without ;
    plt.yticks(fontsize=20);
    
    if save == True:
        plt.savefig('altimetry_track', transparent=True)