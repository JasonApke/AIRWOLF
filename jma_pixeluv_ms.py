#Purpose:  This code is designed to read and navigate optical flow output, 
# where u- and v- are displacements in line/element space
#Author: Jason Apke
#Date: 7/1/2022

from math import *
import netCDF4
import numpy as np
from haversine_distance import *
from jma_uv2sd import *
def jma_geonav(geo,x,y):
    dt = np.dtype('d') 
    a = (np.sin(x,dtype=dt))**2.+((np.cos(x,dtype=dt))**2.)*((np.cos(y,dtype=dt))**2.+(geo.req**2.)/(geo.rpol**2.)*(np.sin(y,dtype=dt))**2.)
    b = -2.* geo.H*np.cos(x,dtype=dt)*np.cos(y,dtype=dt)
    c = geo.H**2. - geo.req**2.
    rs = (-b - (b**2. - 4.*a*c)**(0.5))/(2.*a)
    sx = rs*np.cos(x,dtype=dt)*np.cos(y,dtype=dt)
    sy = -rs*np.sin(x,dtype=dt)
    sz = rs*np.cos(x,dtype=dt)*np.sin(y,dtype=dt)
    lat = np.arctan((geo.req**2.)/(geo.rpol**2.)*(sz/((geo.H-sx)**2. +sy**2.)**0.5),dtype=dt)
    lon = geo.lam0 - np.arctan(sy/(geo.H-sx),dtype=dt)
    lat = np.degrees(lat,dtype=dt)
    lon = np.degrees(lon,dtype=dt)
    return lat, lon

def jma_pixeluv_uv(geo,u_raw,v_raw):

    x4 = (np.copy(geo.x) + u_raw)*geo.xscale + geo.xoffset
    y4 = (np.copy(geo.y) + v_raw)*geo.yscale + geo.yoffset
    x1 = (np.copy(geo.x))*geo.xscale + geo.xoffset
    y1 = (np.copy(geo.y))*geo.yscale + geo.yoffset
    lat1 = geo.lat
    lon1 = geo.lon
    lat4, lon4 = jma_geonav(geo,x4,y4)
    lat2, lon2 = jma_geonav(geo,x1,y1)

    upixel = hav_dist(lat1,lon1,lat2,lon4) / geo.dt_seconds
    vpixel = hav_dist(lat1,lon1,lat4,lon2) / geo.dt_seconds
    cond1 = lon1 > lon4
    upixel[cond1] *= -1
    cond1 = lat1 > lat4
    vpixel[cond1] *= -1


    return upixel, vpixel
def jma_pixeluv_ms(geo,u_raw,v_raw):
    x4 = (np.copy(geo.x) + u_raw)*geo.xscale + geo.xoffset
    y4 = (np.copy(geo.y) + v_raw)*geo.yscale + geo.yoffset
    lat1 = geo.lat
    lon1 = geo.lon
    lat4, lon4 = jma_geonav(geo,x4,y4)
    spd = hav_dist(lat1,lon1,lat4,lon4) / np.abs(geo.dt_seconds)
    _,dire = jma_uv2sd(-1*u_raw,v_raw)
    upixel,vpixel = jma_sd2uv(spd,dire)
    return upixel, vpixel

