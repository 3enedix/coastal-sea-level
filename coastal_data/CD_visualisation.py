import matplotlib.animation as animation
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt

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
    ax.set_extent([5.31, 5.37, 53.41, 53.43], crs=ccrs.PlateCarree())
    ax.gridlines(draw_labels=True, zorder=0, color='lightgrey')
    zoom = 13
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