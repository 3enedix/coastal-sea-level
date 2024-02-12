from scipy.spatial import Delaunay
import numpy as np
from shapely.ops import polygonize, unary_union
from shapely import LineString, MultiLineString, Polygon, get_coordinates
import geopandas as gpd

def concave_hull_alpha_shape(points, alpha=0.01):
    '''
    Code adapted from https://gist.github.com/HTenkanen/49528990d1ab4bcb5562ba01ba6262ef
    Compute the alpha hull (a concave hull with a bit of
    slack that can be influenced with the parameter alpha)
    of a GeoSeries of points.
    '''
    coords = get_coordinates(points)
    tri = Delaunay(coords)
    triangles = coords[tri.vertices]
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

def define_single_transect(first_point, second_point, transect_length=1000):
    '''
    Create a transect perpendicular to a shoreline segment defined by a first and a second point.
    The transect is created through the first point.
    
    Input
    first_point: Tuple with x- and y-coordinate (metric), defining the starting point of the shoreline segment
    second_point: Tuple with x- and y-coordinate (metric), defining the ending point of the shoreline segment
    transect_length: Total length of the resulting transect in [m]

    Output
    LineString of the new transect through first_point
    '''
    x1, y1 = first_point[0], first_point[1]
    x2, y2 = second_point[0], second_point[1]

    delta_x = x2 - x1
    delta_y = y2 - y1

    polar_angle = np.arctan2(delta_y, delta_x)
    polar_angle_new = polar_angle + np.pi/2
    
    x_landwards = x1 - transect_length/2 * np.cos(polar_angle_new)
    y_landwards = y1 - transect_length/2 * np.sin(polar_angle_new)
    
    x_seawards = x1 + transect_length/2 * np.cos(polar_angle_new)
    y_seawards = y1 + transect_length/2 * np.sin(polar_angle_new)

    return LineString([(x_landwards, y_landwards), (x_seawards, y_seawards)])

def smooth_LineString(linestring, n=50):
    '''
    Input:
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





