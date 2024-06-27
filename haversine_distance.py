#Purpose:  This is a haversine/great-circle distance function
#Author: Jason Apke

from math import *
import netCDF4
import matplotlib
import numpy as np

def hav_dist(lat1,lon1,lat2,lon2):
    lat1=np.asarray(lat1) 
    lon1 = np.asarray(lon1)
    lat2 = np.asarray(lat2)
    lon2 = np.asarray(lon2)
    #convert degrees to radians
    dt = np.dtype('d')
    lat1r = np.radians(lat1,dtype=dt)
    lon1r = np.radians(lon1,dtype=dt)
    lat2r = np.radians(lat2,dtype=dt)
    lon2r = np.radians(lon2,dtype=dt)
    pi = 3.14159
    earthrad=6371000.
    dlon = lon2r-lon1r
    dlat = lat2r-lat1r
    a = (np.sin(dlat/2.,dtype=dt))**2. +np.cos(lat1r,dtype=dt)*np.cos(lat2r,dtype=dt)*(np.sin(dlon/2.,dtype=dt))**2.
    c = 2.*np.arctan2((a)**0.5,(1-a)**0.5,dtype=dt)
    distance=earthrad*c
    return distance


