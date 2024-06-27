#Purpose: A function to convert u and v to spd and direction
import numpy as np

def jma_uv2sd(u,v):
    u = np.asarray(u)
    v = np.asarray(v)
    pi = 3.141539625
    rad = pi/180.
    sp = np.sqrt(u*u + v*v)
    dire = 270. - (np.arctan2(v,u)/rad)
    if (np.isscalar(dire)):
        if(dire >= 360.):
            dire = dire-360.
    else:
        conder1 = (dire >= 360.)
        dire[conder1] = dire[conder1]-360.
    return sp, dire
def jma_sd2uv(spd,dire):

    u = -spd*np.sin(np.radians(dire))
    v = -spd*np.cos(np.radians(dire))
    return u, v

