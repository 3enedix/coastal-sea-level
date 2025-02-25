import numpy as np
from scipy.interpolate import interpn, griddata

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