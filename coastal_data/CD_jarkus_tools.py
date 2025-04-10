import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
from scipy.interpolate import interpn, griddata

def get_jarkus_data(datapath_jarkus, year_start, year_end, poly_target, init):
    '''
    Load Jarkus data, tailored for Terschelling Kalman filter validation.
    Returns a GeoDataFrame (columns: years , rows: elevation points),
    with the elevation data reduced to the target area, and thinned out
    to have roughly the same number of points as the initial state.
    '''
    # Jarkus data
    jarkus = xr.open_dataset(datapath_jarkus + 'transect_red_with_derivatives.nc')
    
    # Reduce to Terschelling and the time after 1992
    idx_alongshore, = np.where(jarkus.areacode == 4)
    jarkus_years = pd.to_datetime(jarkus.time).year
    idx_time, = np.where((jarkus_years >= year_start) & (jarkus_years <= year_end))
    jarkus_years = jarkus_years[idx_time]
    jarkus = jarkus.isel(alongshore=idx_alongshore, time=idx_time)

    # Remove values coming from more than one source
    jarkus['altitude_red'] = xr.where(jarkus.nsources <= 1, jarkus.altitude, np.nan) \
        .assign_attrs({"description": "Values coming \"from more than one source\" are masked out based on variable nsources."})

    # Jarkus GeoDataFrame
    lon_jarkus = jarkus.lon.values.reshape(-1)
    lat_jarkus = jarkus.lat.values.reshape(-1)

    # x_jarkus = jarkus.x.values.reshape(-1)
    # y_jarkus = jarkus.y.values.reshape(-1)

    # reshape jarkus elevation to 2D, one timestep per row
    jarkus_elev = jarkus['altitude_red'].values.reshape(len(jarkus.time),
                                                        len(jarkus.alongshore) * len(jarkus.cross_shore))

    jarkus_df = pd.DataFrame(jarkus_elev).T
    jarkus_df.columns = jarkus_years
    jarkus_gdf = gpd.GeoDataFrame(jarkus_df, geometry=gpd.points_from_xy(lon_jarkus, lat_jarkus))
    # jarkus_gdf = gpd.GeoDataFrame(jarkus_df, geometry=gpd.points_from_xy(x_jarkus, y_jarkus))
    jarkus_gdf.columns = jarkus_gdf.columns.astype(str)

    # Reduce to target area
    # poly_28992 = CD_geometry.transform_polygon(poly_target_red, 4326, 28992)
    jarkus_gdf = jarkus_gdf[jarkus_gdf.intersects(poly_target)]

    # thin out the grid so that there are approximately len(init) values left, as evenly spaced as possible
    temp = init.band_data.values.reshape(-1)
    num_values = len(temp[~np.isnan(temp)])

    thin_out_factor = int(len(jarkus_gdf) / num_values)
    jarkus_gdf = jarkus_gdf.iloc[::thin_out_factor]

    jarkus_gdf = jarkus_gdf.set_crs('4326')
    jarkus_gdf = jarkus_gdf.reset_index(drop=True)
    
    return jarkus_gdf

def interpolate_regular_grid_onto_JARKUS(x_state, elev_col, lon_jarkus, lat_jarkus):
    mx = np.unique(x_state.geometry.x)
    my = np.unique(x_state.geometry.y)
    values = x_state[elev_col].values.reshape((len(mx), len(my)))
    interp = interpn((mx, my), values, (lon_jarkus,lat_jarkus))
    return interp

def interpolate_irregular_grid_onto_JARKUS(x_state, elev_col, lon_jarkus, lat_jarkus):
    mx = x_state.geometry.x
    my = x_state.geometry.y
    values = x_state[elev_col]
    interp = griddata(list(zip(mx,my)), values, list(zip(lon_jarkus,lat_jarkus)), 'linear')
    return interp