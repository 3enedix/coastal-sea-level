from pyproj import CRS, Transformer
import shapefile
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely import get_coordinates, LineString

def combine_cassie(path, folders, epsg):
    '''
    Get and combine Cassie shorelines.
    -----------------
    Input
    -----------------
    path: string, main datapath
    folders: List of folders that contains 'coastlines.shp'
    epsg: int, epsg code of the output crs
    -----------------
    Output
    -----------------
    sl_cassie, dictionary with keys 'dates' and 'shorelines'
                (similar structure as CoastSat output)
    '''    
    crs_4326 = CRS.from_epsg(4326)
    crs_new = CRS.from_epsg(epsg)
    transformer = Transformer.from_crs(crs_4326, crs_new)

    datelist = []
    sllist = []

    fn = 'coastlines' 
    for folder in folders:
        sf = shapefile.Reader(path + folder + fn)    
        shorelines = sf.shapeRecords().__geo_interface__ # shapefile to geojson
        for sl in shorelines['features']:
            date = pd.to_datetime(sl['properties']['date'], utc=True)
            date = date.to_pydatetime()
            datelist.append(date)
    
            sl_4326 = sl['geometry']['coordinates']
            sl_transformed = [transformer.transform(_[1], _[0]) for _ in sl_4326]
            sllist_temp = [list(_) for _ in sl_transformed]
    
            sllist.append(np.array(sllist_temp))
            
    # put the dates and shoreline coordinates in a dictionary (mimics CoastSat output)
    sl_cassie = {'dates' : datelist, 'shorelines' : sllist}
    return sl_cassie

def waterline_method(rs_shoreline, ssh, startdate, enddate):
    '''
    Combine shoreline coordinates with sea surface heights
    as a basis for an intertidal DEM ("waterline method").
    ! This function expects to find a file 'tidal_correction_10minutes.csv'
    ! under '/media/bene/Seagate/PhD-data/3_ocean_tide_models/'
    -----------------
    Input
    -----------------
    rs_shoreline: Dictionary with keys 'dates' and 'shorelines'
    ssh: Pandas series of sea level (tidally corrected) with timestamps in index
    startdate: string as 'yyyy-mm-dd'
    enddate: string as 'yyyy-mm-dd'
    -----------------
    Output
    -----------------
    combined_gdf: Intertidal point cloud, GeoDataFrame with the shoreline coordinates
                  and the corresponding sea surface heights
    '''

    # Extract shorelines that fall in the period between startdate and enddate
    sdate = pd.to_datetime(startdate, utc=True)
    edate = pd.to_datetime(enddate, utc=True)
    
    idx_cassie, = np.nonzero(np.array([(_ > sdate) & (_ < edate) for _ in rs_shoreline['dates']]))
    print(str(len(idx_cassie)) + ' images between ' + str(sdate) + ' and ' + str(edate))
    
    dates_cassie = np.array(rs_shoreline['dates'])[idx_cassie]
    shorelines = [rs_shoreline['shorelines'][_] for _ in idx_cassie]

    # Initialise geodataframe with shoreline coordinates and corresponding sea level
    combined_gdf = gpd.GeoDataFrame(columns=['dates', 'ssh', 'shoreline'], geometry='shoreline')
    for i, cassie_date in enumerate(dates_cassie):
        # Get the sea level observation from the respective month
        idx_ssh, = np.nonzero((cassie_date.year == ssh.index.year) & (cassie_date.month == ssh.index.month))
        if len(idx_ssh) == 0:
            print("No sea level observation for", str(cassie_date))
            continue
        ssh_height_corr = ssh.loc[ssh.index[idx_ssh]]
        
        # Get the tidal correction from the 10-minute interval closest to the time of image acquisition
        path = '/media/bene/Seagate/PhD-data/3_ocean_tide_models/'
        fn = 'tidal_correction_10minutes.csv'
        tidal_corr = pd.read_csv(path + fn, index_col='date', parse_dates=['date'])
        
        timediff = tidal_corr.index - pd.to_datetime(cassie_date)
        closest_diff = np.min(np.abs(timediff))
        idx_tcorr = np.nonzero((timediff == closest_diff) | (timediff == -closest_diff))
        tidal_corr.index[idx_tcorr]
        eot_corr = tidal_corr.loc[tidal_corr.index[idx_tcorr], ['corr_eot[cm]']]
        
        # De-tidal-correct sea level
        ssh_height_decorr = ssh_height_corr.values + eot_corr.values/100 # [m], for +/- decision see notebook 19.12.2023

        # Put shoreline and corresponding SSH in geodataframe
        shoreline_coords = get_coordinates(LineString(shorelines[i]))
        cassie_date_expanded = np.repeat(cassie_date, len(shoreline_coords))
        ssh_expanded = np.repeat(ssh_height_decorr, len(shoreline_coords))
        gdf_temp = gpd.GeoDataFrame({
            'dates':cassie_date_expanded,
            'ssh':ssh_expanded,
            'shoreline':gpd.points_from_xy(shoreline_coords[:,0], shoreline_coords[:,1])
                            })
        combined_gdf = pd.concat([combined_gdf, gdf_temp])
    
    combined_gdf = combined_gdf.set_index('dates')
    return combined_gdf