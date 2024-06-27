###
#Purpose:  This code is designed to read and navigate octane files
#Author: Jason Apke
#Date: 6/27/2024
from math import *
import netCDF4
import numpy as np
import datetime
import os



class jma_goesread_uv(object):
    def __init__(self,path,cal="RAW",navit=1,irctp = 0, normraw = 0):
        #check capitals
        if cal == "Raw" or cal=="raw":
            cal = "RAW"
        if cal == "Brit" or cal=="brit":
            cal = "BRIT"
        if cal == "Temp" or cal=="temp":
            cal = "TEMP"
        if cal == "Ref" or cal=="ref":
            cal = "REF"
        if cal != "RAW" and cal != "BRIT" and cal != "TEMP" and cal != "REF":
            print("Cal incorrectly set, use raw, brit, temp or ref, setting to raw")
            cal = "RAW"

        nc = netCDF4.Dataset(path)
        data = np.squeeze(nc.variables['Rad'][:])
        self.radscale = nc.variables['Rad'].scale_factor
        self.radoffset = nc.variables['Rad'].add_offset

        data2 = np.squeeze(nc.variables['U'][:])
        data3 = np.squeeze(nc.variables['V'][:])
        data2_raw = np.squeeze(nc.variables['U_raw'][:])
        data3_raw = np.squeeze(nc.variables['V_raw'][:])
        self.dt_seconds = nc.variables['optical_flow_settings'].dt_seconds
        dataCTP = np.squeeze(nc.variables['CTP'][:])
        x2 = np.squeeze(nc.variables['x'][:])
        y2 = np.squeeze(nc.variables['y'][:])
        xmin = x2[0]
        xmax = x2[len(x2)-1]
        ymin = y2[0]
        ymax = y2[len(y2)-1]
        xscale = nc.variables['x'].scale_factor
        yscale = nc.variables['y'].scale_factor
        xoffset = nc.variables['x'].add_offset
        yoffset = nc.variables['y'].add_offset
        xoffset2= nc.variables['optical_flow_settings'].Image2_xOffset
        yoffset2= nc.variables['optical_flow_settings'].Image2_yOffset
        x22 = np.round((np.squeeze(nc.variables['x'][:])-nc.variables['x'].add_offset)/(nc.variables['x'].scale_factor))
        y22 = np.round((np.squeeze(nc.variables['y'][:])-nc.variables['y'].add_offset)/(nc.variables['y'].scale_factor))
        x,y = np.meshgrid(x2,y2)
        x222,y222 = np.meshgrid(x22,y22)
        self.gip = nc.variables['goes_imager_projection'][:]
        longname= nc.variables['goes_imager_projection'].long_name
        self.longname=longname
        req= nc.variables['goes_imager_projection'].semi_major_axis
        rpol= nc.variables['goes_imager_projection'].semi_minor_axis
        pph= nc.variables['goes_imager_projection'].perspective_point_height
        self.pph = pph
        lam0= nc.variables['goes_imager_projection'].longitude_of_projection_origin
        self.lam0d = lam0

        
        if cal != "RAW":
            fk1= nc.variables['planck_fk1'][:]
            fk2= nc.variables['planck_fk2'][:]
            bc1= nc.variables['planck_bc1'][:]
            bc2= nc.variables['planck_bc2'][:]
            kap1= nc.variables['kappa0'][:]
        time_var = nc.variables['t']
        self.time_var = time_var[:]
        self.time_units = time_var.units
        dtime = netCDF4.num2date(time_var[:],time_var.units)
        tvs = dtime.strftime('%Y%m%d-%H%M%S')
        geodtime = datetime.datetime.strptime(tvs,'%Y%m%d-%H%M%S')
        lam0= radians(lam0)
        nc.close()
        H = pph+req
        if(navit == 1):
            dt = np.dtype('d') #apparently necessary, convert to double
            a = (np.sin(x,dtype=dt))**2.+((np.cos(x,dtype=dt))**2.)*((np.cos(y,dtype=dt))**2.+(req**2.)/(rpol**2.)*(np.sin(y,dtype=dt))**2.)
            b = -2.* H*np.cos(x,dtype=dt)*np.cos(y,dtype=dt)
            c = H**2. - req**2.
            rs = (-b - (b**2. - 4.*a*c)**(0.5))/(2.*a)
            sx = rs*np.cos(x,dtype=dt)*np.cos(y,dtype=dt)
            sy = -rs*np.sin(x,dtype=dt)
            sz = rs*np.cos(x,dtype=dt)*np.sin(y,dtype=dt)
            lat = np.arctan((req**2.)/(rpol**2.)*(sz/((H-sx)**2. +sy**2.)**0.5),dtype=dt)
            lon = lam0 - np.arctan(sy/(H-sx),dtype=dt)
            lat = np.degrees(lat,dtype=dt)
            lon = np.degrees(lon,dtype=dt)
            lat1 = np.amin(lat)
            lon1 = np.amin(lon)
            lat2 = np.amax(lat)
            lon2 = np.amax(lon)
        else:
            lat = 0
            lon = 0
            lat2 = 0
            lon2 = 0
            lat1 = 0
            lon1 = 0
        self.u=data2
        self.v=data3
        self.u_raw = data2_raw
        self.v_raw = data3_raw
        if(normraw == 1):
            self.u_raw /= self.dt_seconds
            self.v_raw /= self.dt_seconds


        self.uPix= 0.
        self.vPix= 0.
        self.lat1 = lat1
        self.lon1 = lon1
        self.lat2 = lat2
        self.lon2 = lon2
        if cal=="RAW":
            self.data=data
        if cal=="REF":
            self.data=kap1*data
        if cal=="TEMP":
            self.data = (fk2/(np.log((fk1/data)+1.))-bc1)/bc2
        if cal=="BRIT":
            self.data = fk1/(exp(fk2/(bc1+(bc2*((fk2/(np.log((fk1/data)+1.))-bc1)/bc2))))-1)

        self.fk1 = fk1
        self.fk2 = fk2
        self.bc1 = bc1
        self.bc2 = bc2
        self.kap1 = kap1

        
        self.lat=lat
        self.lon=lon
        self.tvs=tvs
        self.dtime = geodtime
        if(irctp == 1):
            dataCTP= dataCTP/100.+300.

        self.ctp=dataCTP
        self.x = x222
        self.y = y222
        self.xraw = x2
        self.yraw = y2
        self.x_scaled = x
        self.y_scaled = y
        self.xmin=xmin
        self.xmax=xmax
        self.ymin=ymin
        self.ymax=ymax
        self.rpol = rpol
        self.req = req
        self.H = H
        self.lam0 = lam0
        self.xscale = xscale
        self.yscale = yscale
        self.xoffset = xoffset
        self.yoffset = yoffset
        self.xoffset2 = xoffset2
        self.yoffset2 = yoffset2
        self.fpath=path
    def setsd(self,spd,dire,spd2,dire2):
        self.spd = spd
        self.spdoffset = 0.
        self.spdscale = 0.01
        self.spd2 = spd2
        self.spd2offset = 0.
        self.spd2scale = 0.01
        self.dire = dire
        self.direoffset = 0.
        self.direscale = 0.1
        self.dire2 = dire2
        self.dire2offset = 0.
        self.dire2scale = 0.1
    def setuv(self,u,v):
        self.U = u
        self.Uvaroffset = 0.
        self.Uvarscale = 0.01
        self.V = v
        self.Vvaroffset = 0.
        self.Vvarscale = 0.01

    def setctcd(self,ctc,ctd):
        self.ctc = ctc
        self.ctcoffset = 0.
        self.ctcscale = 0.1
        self.ctd = ctd*1E3
        self.ctdoffset = 0.
        self.ctdscale = 1E-3
    def setOFS(self,im2xoffset,im2yoffset,lambdav=1.,lambdac=0.,alpha=5.,
            filtsigma=3.,scalef=0.5,kiterations=4,literations=3,
            miterations=5,cgiterations=30,normmax=628.8972,
            normmin=-20.28991,dofirstguess=0,dt_seconds=-60.):
        self.Image2_xOffset = im2xoffset
        self.Image2_yOffset = im2yoffset
        self.lambdav = lambdav
        self.lambdac = lambdac
        self.alpha = alpha
        self.filtsigma = filtsigma
        self.ScaleF = scalef
        self.K_Iterations= kiterations
        self.L_Iterations= literations
        self.M_Iterations = miterations
        self.CG_Iterations = cgiterations
        self.NormMax = normmax
        self.NormMin = normmin
        self.dofirstguess = dofirstguess
        self.dt_seconds = dt_seconds

    def filewrite(self, outloc,dogzip=True):
        #a function to write octane like files for AIRWOLF if needed...
        y, x = self.data.shape
        ncfile = netCDF4.Dataset(outloc, 'w')
        ncfile.createDimension('y',y)
        ncfile.createDimension('x',x)
        y = ncfile.createVariable('y', np.dtype('short').char,('y'))
        x = ncfile.createVariable('x', np.dtype('short').char,('x'))
        Uvar= ncfile.createVariable('U',np.dtype('short').char,('y','x'))
        Vvar= ncfile.createVariable('V',np.dtype('short').char,('y','x'))
        CTPvar= ncfile.createVariable('CTP',np.dtype('short').char,('y','x'))
        Radvar= ncfile.createVariable('Rad',np.dtype('short').char,('y','x'))
        OFSvar= ncfile.createVariable('optical_flow_settings',np.dtype('int').char)
        planckfk1var= ncfile.createVariable('planck_fk1',np.dtype('float').char)
        planckfk2var= ncfile.createVariable('planck_fk2',np.dtype('float').char)
        planckbc1var= ncfile.createVariable('planck_bc1',np.dtype('float').char)
        planckbc2var= ncfile.createVariable('planck_bc2',np.dtype('float').char)
        kappa0var= ncfile.createVariable('kappa0',np.dtype('float').char)
        gipstring = 'fixedgrid_projection'
        gip = ncfile.createVariable(gipstring, np.dtype('short').char)
        t = ncfile.createVariable('t',np.dtype('double').char)
        y[:] = (np.round((self.yraw[:]-self.yoffset)/self.yscale)).astype('short')
        y.standard_name = 'projection_y_coordinate'
        y.scale_factor = self.yscale
        y.add_offset = self.yoffset
        y.units = 'rad'
        x[:] = (np.round((self.xraw[:]-self.xoffset)/self.xscale)).astype('short')
        x.standard_name = 'projection_x_coordinate'
        x.scale_factor = self.xscale
        x.add_offset = self.xoffset
        x.units= 'rad'

        Uvar[:]= self.U[:] 
        Uvar.scale_factor = self.Uvarscale
        Uvar.add_offset = self.Uvaroffset
        Uvar.units = 'm/s'
        Uvar.grid_mapping = gipstring
        Vvar[:]= self.V[:] 
        Vvar.scale_factor = self.Vvarscale
        Vvar.add_offset = self.Vvaroffset
        Vvar.units = 'm/s'
        Vvar.grid_mapping = gipstring
        CTPvar.long_name='Rad'
        CTPvar.grid_mapping = gipstring
        CTPvar.units='m'
        Radvar.long_name = "Rad"
        Radvar.grid_mapping = gipstring
        Radvar.scale_factor = self.radscale
        Radvar.add_offset = self.radoffset
        OFSvar.long_name = "Optical Flow Settings"
        OFSvar.key = "1 = Modified Zimmer et al. (2011), 2 = Farneback, 3 = Brox (2004), 4 = Least Squares"
        OFSvar.Image2_xOffset = self.im2xoffset
        OFSvar.Image2_yOffset = self.im2yoffset
        OFSvar.lambdav = self.lambdav
        OFSvar.lambdac = self.lambdac
        OFSvar.alpha = self.alpha
        OFSvar.filtsigma = self.filtsigma
        OFSvar.ScaleF = self.scalef
        OFSvar.K_Iterations= self.kiterations
        OFSvar.L_Iterations= self.literations
        OFSvar.M_Iterations = self.miterations
        OFSvar.CG_Iterations = self.cgiterations
        OFSvar.NormMax = self.normmax
        OFSvar.NormMin = self.normmin
        OFSvar.dofirstguess = self.dofirstguess
        OFSvar.dt_seconds = self.dt_seconds
        planckfk1var[:] = self.fk1
        planckfk2var[:] = self.fk2
        planckbc1var[:] = self.bc1
        planckbc2var[:] = self.bc2
        kappa0var[:] = self.kap1




        


        gip[:] = self.gip
        gip.long_name = self.longname
        gip.grid_mapping_name = 'geostationary'
        gip.semi_major_axis = self.req
        gip.semi_minor_axis = self.rpol
        gip.perspective_point_height= self.pph
        gip.longitude_of_projection_origin = self.lam0d
        gip.sweep_angle_axis = 'x'
        t[:] = self.time_var
        t.units = self.time_units
        ncfile.project = "GOES"
        ncfile.production_site = "NSOF"
        ncfile.orbital_slot = "GOES-East"
        ncfile.platform_ID = "G16"
        ncfile.scene_id = "Mesoscale"
        ncfile.satellite_latitude= 0.
        ncfile.satellite_longitude = self.lam0d
        ncfile.dataset_name=os.path.basename(outloc)

        ncfile.close()
        if(dogzip):
            os.system('chmod a+rwx '+outloc)
            os.system('gzip '+outloc)



