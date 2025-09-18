import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
from scipy.interpolate import interpn, griddata
from scipy.signal import find_peaks
from shapely import LineString
from shapely.ops import split
import pdb

from coastal_data import CD_geometry

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

def get_Ters_section_polygons(poly_target, jarkus, buffer_vol=-250):
    buffer_vol = CD_geometry.dist_meter_to_dist_deg(buffer_vol)
    poly_target_red = poly_target.buffer(buffer_vol)
    print(f'Reduced target polygon by {round(1-(poly_target_red.area / poly_target.area), 3)*100} %')
    
    # West Terschelling erosive section
    idx_west = [25, 57]
    # Middle accretive section
    idx_middle = [58, 111]
    # East Terschelling erosive section
    idx_east = [112, 144]
    
    poly_west, _ = cut_poly_with_transects(jarkus, poly_target_red, idx_west)
    poly_center, _ = cut_poly_with_transects(jarkus, poly_target_red, idx_middle)
    poly_east, _ = cut_poly_with_transects(jarkus, poly_target_red, idx_east)

    # remaining area
#     idx_west_out = [14, 25]
#     _, poly_west_out = cut_poly_with_transects(jarkus, poly_target_red, idx_west_out)

#     idx_east_out = [144, -1]
#     poly_east_out, _ = cut_poly_with_transects(jarkus, poly_target_red, idx_east_out)
    
    return poly_target_red, poly_west, poly_center, poly_east #, poly_west_out, poly_east_out

def cut_poly_with_transects(jarkus, poly_target_red, idx):
    jarkus_transect = jarkus.isel(alongshore=idx[0])
    transect_1 = LineString(zip(jarkus_transect.lon, jarkus_transect.lat))
    
    jarkus_transect = jarkus.isel(alongshore=idx[1])
    transect_2 = LineString(zip(jarkus_transect.lon, jarkus_transect.lat))

    split1 = split(poly_target_red, transect_2).geoms[0]
    cut_area = split(split1, transect_1).geoms[1]

    return cut_area, split1
    
# ===================================================================================
# Get shoreline as intersection between elevation profile and a horizontal plane at sea level
# ===================================================================================

def find_intersections(profile, slh, cross_shore):
    '''
    Code adapted from JAT (Christa van IJzendoorn)
    
    profile : array
        Topography heights along one profile at one point in time
    slh : float
        Sea level height at one point in time
    '''
    profile_heights = pd.Series(profile.values).interpolate().tolist()
    slh_plane = np.resize(slh, (len(profile_heights)))

    # subtract the sea level plane from the profile
    diff_profile_sl = profile_heights - slh_plane

    # where is the difference positive/negative (-> where is the sea-level-shifted profile below/above zero):
    below_above = np.sign(diff_profile_sl)

    # where does the sign change:
    transitions = np.diff(below_above)

    # turn nans into zeros
    transitions = np.nan_to_num(transitions)

    # get the cross-shore coordinates for all found transitions
    # get the indices of the land/ocean transitions
    intersection_idxs = np.nonzero(transitions)
    # intersections = np.array([profile.cross_shore[idx] for idx in intersection_idxs[0]])
    intersections = np.array([cross_shore[idx] for idx in intersection_idxs[0]])
    
    return intersections
    
def find_primary_dunetop(profile, cross_shore):
    '''
    Code adapted from JAT (Christa van IJzendoorn)
    
    profile : array
        Topography heights along one profile at one point in time
    '''
    primary_dune_height = 5 # standard value from JAT .yaml file
    primary_dune_prominence = 2.0

    dune_top_prim = find_peaks(profile.values, height=primary_dune_height, prominence=primary_dune_prominence)

    if len(dune_top_prim[0]) != 0: # If a peak is found in the profile
        # Select the most seaward peak found of the primarypeaks
        dune_top_prim_idx = dune_top_prim[0][-1]
        if isinstance(cross_shore, xr.DataArray):
            DuneTop_prim_x = cross_shore[dune_top_prim_idx].values
        elif instance(cross_shore, pd.DataFrame):
            DuneTop_prim_x = cross_shore[dune_top_prim_idx]
    else:
        DuneTop_prim_x = np.nan
        
    return DuneTop_prim_x

def identify_coastline(intersections, DuneTop_prim_x):
    '''
    Code adapted from JAT (Christa van IJzendoorn)
    
    intersections: Output of function find_intersections(profile, slh)
    DuneTop_prim_x: Output of function find_primary_dunetop(profile)
    '''
    # Case 1: No primary dunetop found -> Get the most seaward intersect
    if len(intersections) != 0 and np.isnan(DuneTop_prim_x):
        coastline = intersections[-1]

    # Case 2: Use primary dunetop as landward border
    elif len(intersections) != 0:
        # all intersections seaward of the dunetop:
        intersection_sw = intersections[intersections > DuneTop_prim_x]
        if len(intersection_sw) != 0:

            # Case 2.1: Distance between all intersections seaward of dune peak is larger than 100m:
            if max(intersection_sw) - min(intersection_sw) > 100: 
                # Get all intersections at least 100m landwards of most offshore intersection 
                intersection_lw = intersection_sw[intersection_sw < (min(intersection_sw) + 100)] 
                # Get the most seaward intersection
                coastline = intersection_lw[-1] 

            # Case 2.2: Distance between all intersections seaward of dune peak is smaller than 100m:
            else: 
                # Get the most seaward intersection
                coastline = intersection_sw[-1]
        else:
            coastline = np.nan
    else:
        coastline = np.nan
        
    return coastline