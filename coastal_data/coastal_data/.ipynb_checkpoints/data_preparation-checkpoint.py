import xarray as xr
import numpy as np
import json
from datetime import datetime


def prepare_s3_data(datapath, filename_reprocessed, filename_input, filename_output, dist2coast):

    # import netCDF files
    nc_re = xr.open_dataset(datapath + filename_reprocessed, engine='netcdf4')
    nc_in = xr.open_dataset(datapath + filename_input, engine='netcdf4')

    # recompute timestamps for reprocessed file
    refdate = np.datetime64('2000-01-01T00:00:00')
    deltas = nc_re.time_20_ku * (1e9 * (np.timedelta64(1, 'ns')))
    time_range = {"time_20_ku": refdate + deltas}
    nc_re = nc_re.assign_coords(time_range)

    # indices of the points closer than dist2coast
    idx_20_ku = np.where(nc_in.dist_coast_20_ku < dist2coast)[0]
    idx_01 = np.where(nc_in.dist_coast_01 < dist2coast)[0]
    idx_20_c = np.where(nc_in.dist_coast_20_c < dist2coast)[0]
    idx_all_dict = {"time_20_ku": idx_20_ku, "time_01": idx_01, "time_20_c": idx_20_c}
    idx_20_ku_dict = {"time_20_ku": idx_20_ku}

    # reduce the tracks
    nc_in = nc_in.isel(idx_all_dict)
    nc_re_red = nc_re.isel(idx_20_ku_dict)

    # put the two reduced files together
    nc_in = nc_in.merge(nc_re_red)

    # give the resulting file a few new attributes
    nc_in = nc_in.assign_attrs({
        'description':'Sentinel-3 altimetric range retracked with ALES in EarthConsole PPRO together with the input file, track reduced to keep only points closer than dist2coast.',
        'dist2coast [m]': dist2coast,
        'creation_date':json.dumps(datetime.now(), indent=4, sort_keys=True, default=str),
        'author':'Susann Aschenneller',
        'email':'s.aschenneller@utwente.nl'    
    })

    nc_in.to_netcdf(filename_output)