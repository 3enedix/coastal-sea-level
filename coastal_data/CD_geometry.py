from scipy.spatial import Delaunay
import numpy as np
import pandas as pd
from shapely.ops import polygonize, unary_union
from shapely import Point, LineString, MultiLineString, MultiPoint, Polygon, get_coordinates, segmentize, distance, intersection, buffer
import geopandas as gpd
import geojson
from pyproj import CRS, Transformer
import matplotlib.pyplot as plt

from coastal_data import CD_statistics

import pdb

def median_shoreline_from_transect_intersections(shorelines, spacing=100, transect_length=5000, smooth_factor=100):
    '''
    From a set of shorelines, get the median shoreline
    as the median of the intersections of the shorelines
    with a set of transects.

    ! Does currently not involve tidal correction !

    Input
    shorelines - List of nx2-arrays with n shoreline coordinates (x,y)
    spacing - spacing of the created transects
    transect_length - length of the created transectss
    smooth_factor - filter length, how strong the base shoreline (basis for the transects) should be smoothed

    Output
    median_sl - LineString of the median shoreline
    '''
    # Get the longest shoreline as basis for the transects
    dist = []    
    for sl in shorelines:
        if sl.size == 0:
            continue
        else:
            dist.append(distance(Point(sl[0]), Point(sl[-1])))
    idx_long, = np.nonzero(dist == np.max(dist))[0]
    long_sl = LineString(shorelines[idx_long])
    smooth_sl = smooth_LineString(long_sl, n=smooth_factor)
    
    # Create transects
    # if np.unique(get_coordinates(smooth_sl)).size < 4:
    transects_gdf = create_transects(smooth_sl, spacing=spacing, transect_length=transect_length)
    
    # Compute intersections between shorelines and transects, get the median of all intersections per transects
    nr_transects = len(transects_gdf)
    transect_median = np.full((nr_transects,2), np.nan)
    for idx_trans, transect in enumerate(transects_gdf.geometry):
        x_temp, y_temp = [], [] # x- and y-coords of the intersections between all shorelines and one transect
        for sl in shorelines:
            shoreline = LineString(sl)
            int = intersection(shoreline, transect)
            nr_ints = get_coordinates(int).shape[0]
            if nr_ints == 1:
                x_temp.append(get_coordinates(int)[0][0])
                y_temp.append(get_coordinates(int)[0][1])
            if nr_ints > 1: # = MultiPoint
                # Select the most seaward point (furthest away from transect origin)
                mp_list = list(int.geoms)
                dist = [distance(_, Point(transect.coords[0])) for _ in mp_list]
                idx, = np.nonzero(dist == np.max(dist))[0]
                x_temp.append(mp_list[idx].x)
                y_temp.append(mp_list[idx].y)
        transect_median[idx_trans, 0] = np.median(x_temp)
        transect_median[idx_trans, 1] = np.median(y_temp)
    
    median_sl = LineString(transect_median)
    return median_sl

def get_DEM_contour(x, y, elev, h=0, tarea=None):
    '''
    Extract the contour at a certain elevation from a DEM.
    
    Input
    -----
    x, y: pandas Series with x an dy coordinates of all grid points
    elev - pandas Series with the corresponding elevation values
    h  - int/float, elevation from which to extract the contour
    tarea - shapely polygon of the target area (only the part of the contour inside the target area is kept)
        Providing the target area is optional.

    Output
    -----
    z_ls - shapely LineString of the contour cut to the target area (if supplied)
    '''   
    # Get contour with matplotlib
    zcontour = plt.tricontour(x, y, elev, levels=[h])
    plt.close()

    # Turn matplotlib collection to shapely LineStrings
    path = zcontour.collections[0].get_paths()[0]

    z_ls = LineString(path.vertices)

    # Get only the part inside the target area
    if tarea != None:
        z_ls = intersection(z_ls, tarea)

    return z_ls

def shoreline_outlier_rejection(shorelines, ref_line, epsg, t=2):
    '''
    Input
    -----
    shorelines - array of nx2 arrays with the shoreline coordinates
    ref_line - shapely LineString approximating the shoreline over the entire coastal stretch
    epsg - float
    t - threshold parameter. The threshold is computed as t * std, (std of distances to reference shoreline),
    The shoreline points inside that distance (< t * std) are kept.

    Output
    -----
    shorelines_red - array of nx2 arrays with the remaining shoreline coordinates without outliers
    '''
    if epsg != 4326:
        ref_line = switch_linestring_xy(ref_line)
        
    dists = {} # to keep the relationship with c_shorelines
    all_dists = [] # to compute the std of all distances
    for i, shoreline in enumerate(shorelines):
        dist_temp = [distance(Point(_), ref_line) for _ in shoreline]
        dists[i] = dist_temp
        all_dists = all_dists + dist_temp

    std = CD_statistics.std(all_dists)
    thresh = t*std
    
    shorelines_red = np.empty_like(shorelines)
    
    for i, shoreline in enumerate(shorelines):
        idx = np.where(dists[i] <= thresh)
        shorelines_red[i] = shoreline[idx]

    # Compute percentage of discarded shoreline points
    n_disc = (all_dists > thresh).sum()
    perc_disc = round(n_disc / len(all_dists),2)*100
    print(f'{perc_disc}% of shoreline points were discarded.')

    return shorelines_red

def concave_hull_alpha_shape(points, alpha=0.01):
    '''
    Code adapted from https://gist.github.com/HTenkanen/49528990d1ab4bcb5562ba01ba6262ef
    Compute the alpha hull (a concave hull with a bit of
    slack that can be influenced with the parameter alpha)
    of a GeoSeries of points.
    '''
    coords = get_coordinates(points)
    tri = Delaunay(coords)
    triangles = coords[tri.simplices]
    a = ((triangles[:,0,0] - triangles[:,1,0]) ** 2 + (triangles[:,0,1] - triangles[:,1,1]) ** 2) ** 0.5
    b = ((triangles[:,1,0] - triangles[:,2,0]) ** 2 + (triangles[:,1,1] - triangles[:,2,1]) ** 2) ** 0.5
    c = ((triangles[:,2,0] - triangles[:,0,0]) ** 2 + (triangles[:,2,1] - triangles[:,0,1]) ** 2) ** 0.5
    s = ( a + b + c ) / 2.0
    areas = (s*(s-a)*(s-b)*(s-c)) ** 0.5
    circums = a * b * c / (4.0 * areas)
    filtered = triangles[circums < (1.0 / alpha)]
    edge1 = filtered[:,(0,1)]
    edge2 = filtered[:,(1,2)]
    edge3 = filtered[:,(2,0)]
    edge_points = np.unique(np.concatenate((edge1,edge2,edge3)), axis = 0).tolist()
    m = MultiLineString(edge_points)
    triangles = list(polygonize(m))
    
    # return gpd.GeoDataFrame({"geometry": [unary_union(triangles)]}, index=[0])
    return unary_union(triangles)

def get_polar_angle(seg):
    '''
    Compute polar angle of a single line segment.
    
    Input
    seg - List (len=2) with (x,y)-tuples
    
    Output
    polar angle in radians
    '''
    x1, y1 = seg[0][0], seg[0][1]
    x2, y2 = seg[1][0], seg[1][1]
    
    delta_x = x2 - x1
    delta_y = y2 - y1
    
    polar_angle = np.arctan2(delta_y, delta_x)
    return(polar_angle)

def dist_angle_to_coords(start_point, dist, polar_angle):
    '''
    Compute coordinate from distance and polar angle based on a starting point.

    Input
    start_point - tuple (x,y) (metric)
    dist - distance (same unit as start_point)
    polar_angle - polar angle (azimuth) in radians

    Output
    new coordinate as tuple (x,y)
    '''
    
    x_new = start_point[0] + dist * np.cos(polar_angle)
    y_new = start_point[1] + dist * np.sin(polar_angle)

    return (x_new, y_new)

def create_transects(shoreline, spacing=100, transect_length=5000, save_path=None, save_fn=None):
    '''
    Create transects perpendicular on a LineString and
    saves them as geojson file.

    Input
    shoreline - LineString
    spacing - minimum spacing of the desired transects
    save_path, save_fn - strings to specify the location of the geojson file

    Output
    transects as LineStrings in a GeoDataFrame
    Transects are saved to geojson if save_path and save_fn are passed
    '''
    sl_seg = segmentize(shoreline, spacing)
    # trans_gdf = gpd.GeoDataFrame(columns=['name', 'transect'], geometry='transect').set_index('name')
    trans_gdf = gpd.GeoDataFrame()
    featureList = []
    nr = 1 # transect number
    for i in range(1, len(sl_seg.coords)-1): # i=index of the point through which the transect should go
        seg1 = sl_seg.coords[i-1:i+1]
        seg2 = sl_seg.coords[i:i+2]
    
        t1 = get_polar_angle(seg1)
        t2 = get_polar_angle(seg2)
        polar_angle_new = np.mean([t1,t2]) + np.pi/2
    
        point_seawards = dist_angle_to_coords(seg1[1], transect_length/2, polar_angle_new)
        point_landwards = dist_angle_to_coords(seg1[1], -transect_length/2, polar_angle_new)
        if (len(point_seawards) == 0) | (len(point_landwards) == 0):
            continue
        
        trans_temp = gpd.GeoDataFrame(
            {'name':nr,
            'transect': LineString([point_seawards, point_landwards])},
            geometry='transect',
            index=[0]).set_index('name')
        trans_gdf = pd.concat([trans_gdf, trans_temp])

        if (save_path != None) and (save_fn != None):
            trans_line_temp = geojson.LineString([point_landwards, point_seawards])
            trans_feature_temp = geojson.Feature(geometry = trans_line_temp, properties = {'name': str(nr)})
            featureList.append(trans_feature_temp)
        nr = nr + 1  
        
    if (save_path != None) and (save_fn != None):
        transects = geojson.FeatureCollection(featureList)
        with open(save_path + save_fn, 'w') as f:
            geojson.dump(transects, f)
            
    return trans_gdf

def define_single_transect(first_point, second_point, transect_length=1000):
    '''
    Create a transect perpendicular to a (single) shoreline segment defined by a first and a second point.
    The transect is created through the first point.
    
    Input
    first_point: Tuple with x- and y-coordinate (metric), defining the starting point of the shoreline segment
    second_point: Tuple with x- and y-coordinate (metric), defining the ending point of the shoreline segment
    transect_length: Total length of the resulting transect in [m]

    Output
    LineString of the new transect through first_point
    '''
    seg = [first_point, second_point]
    polar_angle = get_polar_angle(seg)
    polar_angle_new = polar_angle + np.pi/2
    
    x_landwards = x1 - transect_length/2 * np.cos(polar_angle_new)
    y_landwards = y1 - transect_length/2 * np.sin(polar_angle_new)
    
    x_seawards = x1 + transect_length/2 * np.cos(polar_angle_new)
    y_seawards = y1 + transect_length/2 * np.sin(polar_angle_new)

    return LineString([(x_landwards, y_landwards), (x_seawards, y_seawards)])

def smooth_LineString(linestring, n=50):
    '''
    Input
    linestring
    n = filter length
    '''
    df = np.ones(n)/n

    x = get_coordinates(linestring)[:,0]
    y = get_coordinates(linestring)[:,1]
    
    x_smooth = np.convolve(x, df, 'valid')
    y_smooth = np.convolve(y, df, 'valid')
    
    return(LineString(np.vstack([x_smooth, y_smooth]).transpose()))

def get_rotated_boundary_box(array):
    '''
    Code from https://stackoverflow.com/a/45276634
    Input: Array of points
    Output: Corners of the rotated boundary box as shapely Polygon
    '''
    ca = np.cov(array,y = None,rowvar = 0,bias = 1)
    
    v, vect = np.linalg.eig(ca)
    tvect = np.transpose(vect)
    
    ar = np.dot(array,np.linalg.inv(tvect))
    
    # get the minimum and maximum x and y 
    mina = np.min(ar,axis=0)
    maxa = np.max(ar,axis=0)
    diff = (maxa - mina)*0.5
    
    # the center is just half way between the min and max xy
    center = mina + diff
    
    #get the 4 corners by subtracting and adding half the bounding boxes height and width to the center
    corners = np.array([center+[-diff[0],-diff[1]],center+[diff[0],-diff[1]],center+[diff[0],diff[1]],center+[-diff[0],diff[1]],center+[-diff[0],-diff[1]]])
    
    #use the the eigenvectors as a rotation matrix and
    #rotate the corners and the centerback
    corners = np.dot(corners,tvect)
    center = np.dot(center,tvect)

    return(corners)

def get_area_covered_by_shorelines(shorelines, alpha, buffer_size=250):
    '''
    Get the outer boundary of an area defined by a set of shorelines,
    expanded by a buffer zone
    Used to define the target grid for the desired DEM.
    
    Input
    -----
    shorelines - List of nx2-arrays with n shoreline coordinates (x,y)
    alpha - Parameter for alpha shape (how much slack around the concave hull)
    buffer_size - Buffer in [m/degree] (depends on reference system) to expand

    Output
    -----
    poly_buffered - shapely Polygon
    
    '''
    lines = [LineString(_) for _ in shorelines]
    poly = concave_hull_alpha_shape(lines, alpha=alpha)
    # !!! might end up in a MultiPolygon
    poly_buffered = buffer(poly, buffer_size)
    return poly_buffered

def create_target_grid(poly, resolution=100):
    '''
    Create a grid inside a pre-defined polygon.

    Input
    -----
    poly - Shapely polygon defining the grid area.
    resolution - grid size in [m/degree] (depends on reference system) (same in x-/y-direction)
    
    Ouput
    -----
    x, y - List of grid coordinates inside the polygon
    x_full, y_full - List of grid coordinates inside the boundary box
    '''

    lonmin, latmin, lonmax, latmax = poly.bounds
    
    lon_vec = np.arange(lonmin, lonmax, resolution)
    lat_vec = np.arange(latmin, latmax, resolution)
    
    x_grid, y_grid = np.meshgrid(lon_vec, lat_vec)
    # x_grid, y_grid = np.round(x_grid, 4), np.round(y_grid, 4)
    
    points = MultiPoint(list(zip(x_grid.flatten(),y_grid.flatten())))
    x_full = [get_coordinates(_)[0][0] for _ in points.geoms]
    y_full = [get_coordinates(_)[0][1] for _ in points.geoms]

    valid_points = points.intersection(poly)
    
    x = [get_coordinates(_)[0][0] for _ in valid_points.geoms]
    y = [get_coordinates(_)[0][1] for _ in valid_points.geoms]

    return x, y, x_full, y_full

def switch_polygon_xy(poly):
    poly_lat = get_coordinates(poly)[:,0]
    poly_lon = get_coordinates(poly)[:,1]
    return Polygon(list(zip(poly_lon, poly_lat)))

def switch_linestring_xy(line):
    line_lon = get_coordinates(line)[:,0]
    line_lat = get_coordinates(line)[:,1]
    return LineString(list(zip(line_lat, line_lon)))

def transform_polygon(poly, epsg_old, epsg_new):    
    crs_old = CRS.from_epsg(epsg_old)
    crs_new = CRS.from_epsg(epsg_new)
    transformer = Transformer.from_crs(crs_old, crs_new)
    poly_coords = get_coordinates(poly)
    poly_coords_t = [transformer.transform(_[0], _[1]) for _ in poly_coords]
    
    return(Polygon(poly_coords_t))

def cut_DEM_to_target_area(dem, varname, target_poly, source):
    '''
    Extract area within a polygon from a global DEM.
    ! DEM and polygon have to be in the same CRS.

    Input
    dem: Global DEM as xarray Dataset
    varname: The variable name that contains the elevation data
    target_poly: Shapely polygon, where to extract the data
    source: String, e.g. 'gebco'

    Output
    dem_gdf: GeoDataFrame, one Point per row, including elevation and source
    '''
    dem_df = dem.to_dataframe().reset_index()
    dem_df = dem_df.rename(columns={'lat':'y', 'lon':'x'})
    dem_gdf = gpd.GeoDataFrame({
                'elevation':dem_df[varname],
                }, geometry=gpd.points_from_xy(dem_df.x, dem_df.y))

    dem_gdf = dem_gdf[dem_gdf.intersects(target_poly)]
    dem_gdf['source'] = source
    
    return dem_gdf

def dist_meter_to_dist_deg(dist_m):
    R = 6371e3 # Earth radius [m]
    return (180 * dist_m) / (R * np.pi)

def equalise_LineString_segment_lengths(line_orig, seg_length):
    '''
    Equalise the segment lengths of a LineString, so that all segments
    have the exact same length. Uses interpolation, therefore the output
    LineString is not necessarily the exact overlay of the input LineString
    (if the input LineString has segments shorter than the desired seg_length
    the interpolation "cuts off" the edge).
    
    Input
    -----
    line_orig - LineString
    seg_length - Desired segment length (float/int)

    Output
    -----
    line_seg - LineString
    '''
    total_length = line_orig.length
    if total_length == 0:
        return None
    # Number of segments needed
    num_segments = int(total_length / seg_length)
    # Generate evenly spaced points along the LineString
    # Include the start (0) and end (total_length)
    distances = np.linspace(0, total_length, num_segments + 1)
    new_points = [line_orig.interpolate(distance) for distance in distances]

    line_seg = LineString(new_points)
    return line_seg
