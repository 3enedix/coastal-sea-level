import xarray as xr
import numpy as np
import json
from datetime import datetime
import pandas as pd
import os


def prepare_s3_data(datapath_in, datapath_out, filename_reprocessed, filename_input, filename_output, dist2coast):

    # import netCDF files
    nc_re = xr.open_dataset(datapath_in + filename_reprocessed, engine='netcdf4')
    nc_in = xr.open_dataset(datapath_in + filename_input, engine='netcdf4')

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
        'author':'Bene Aschenneller',
        'email':'s.aschenneller@utwente.nl'    
    })

    nc_in.to_netcdf(datapath_out + filename_output)
    
def prepare_TG_rijkswaterstaat_data(datapath_in, datapath_out):
    '''
    Combines all .csv-files from datapath_in with tide gauge data from rijkswaterstaat. Extracts only necessary data fields, converts times to UTC.
    
    BA 11/2022
    
    Arguments
    ----------
    'datapath_in': string
    'datapath_out': string
    
    Returns
    ----------
    Saves the combined .csv-file in datapath_out.
    '''
    data = pd.DataFrame(columns=['datetime[utc]', 'ssh[cm]'])
    data = data.set_index('datetime[utc]')

    for root, dirs, files in os.walk(datapath_in):
        for fname in files:
            if ('.csv' in fname) & ('#' not in fname):
                data_orig = pd.read_csv(datapath_in + fname, sep=';', header=0,\
                            encoding='latin_1', engine='python', usecols=['WAARNEMINGDATUM', \
                            'WAARNEMINGTIJD (MET/CET)', 'NUMERIEKEWAARDE', 'WAARDEBEPALINGSMETHODE_CODE'])
                
                # Extract code F007: "Rekenkundig gemiddelde waarde over vorige 5 en volgende 5 minuten"
                F007 = data_orig[data_orig['WAARDEBEPALINGSMETHODE_CODE'] == 'other:F007']
                # Extract code F001: "Rekenkundig gemiddelde waarde over vorige 10 minuten"
                F001 = data_orig[data_orig['WAARDEBEPALINGSMETHODE_CODE'] == 'other:F001']
                data_orig = pd.concat([F001, F007], axis=0)            
                data_orig = data_orig.drop('WAARDEBEPALINGSMETHODE_CODE', axis=1)

                data_temp = pd.DataFrame()
                data_temp['ssh[cm]'] = data_orig['NUMERIEKEWAARDE']
                date = pd.to_datetime(data_orig['WAARNEMINGDATUM'] + ' ' + data_orig['WAARNEMINGTIJD (MET/CET)'], \
                                                format="%d-%m-%Y %H:%M:%S", utc=True)
                date = date - pd.Timedelta("1 hour")
                data_temp['datetime[utc]'] = date
                data_temp = data_temp.set_index('datetime[utc]')

                data = pd.concat([data, data_temp])
                
    data = data.sort_index()
    data = data[data['ssh[cm]'] != 999999999]
    data.to_csv(datapath_out + "TG_rijkswaterstaat_combined.csv")
