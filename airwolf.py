#File name: airwolf.py
#Purpose: This code follows the design of SPYNET, using pytorch api, training on OCTANE motion files
# The design is to compute 1-min motions from satellite imagery similar to variational optical flow
#Author: Jason Apke
#Requirements: Pytorch, torchvision, os, glob, netCDF4, matplotlib, cartopy
#
#Library Imports
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torchvision.transforms as transforms
import os, glob, netCDF4,sys, random
import numpy as np
from warnings import filterwarnings
import matplotlib
import matplotlib.pyplot as plt
import cartopy
from cartopy import config
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from jma_pixeluv_ms import *
from jma_goesread_uv2 import *
from jma_uv2sd import *

filterwarnings(action='ignore', category=DeprecationWarning, message='`np.bool` is a deprecated alias')
def impad(im,x1,x2,y1,y2):
    dopad = False
    x2n = x2
    y2n = y2
    x1n = x1
    y1n = y1
    sx, sy = im.shape
    result = im
    xp2 = 0
    xp1 = 0
    yp2 = 0
    yp1 = 0
    if(x2 > sx):
        xp2 = x2-sx
        x2n = sx+(x2-sx) #+(0-x1)
        dopad = True
    if(y2 > sy):
        yp2 = y2-sy
        y2n = sy+(y2-sy) #+(0-y1)
        dopad = True
    if(x1 < 0):
        xp1 = -1*x1
        x1n = 0
        x2n = x2n + (0-x1)
        dopad = True
    if(y1 < 0):
        yp1 = -1*y1
        y1n = 0
        y2n = y2n + (0-y1)
        dopad = True
    if(dopad):
        result = np.pad(im,((xp1,xp2),(yp1,yp2)),constant_values=((0,0),(0,0)))

    return result[x1n:x2n,y1n:y2n]
        
        
    
def solar_zenith_angle(lon,lat,dtvs):
    xj = float(datetime.datetime.strftime(dtvs,"%j"))
    tvs = datetime.datetime.strftime(dtvs,"%Y%m%d-%H%M%S")
    yyyy = float(tvs[0:4])
    mm = float(tvs[4:6])
    dd = float(tvs[6:8])
    hh = float(tvs[9:11])
    mi = float(tvs[11:13])
    ss = float(tvs[13:15])
    timesat = hh+mi/60.
    tsm = timesat+lon/15.0 
    xlo = np.radians(lon)
    xla = np.radians(lat)

    a1 = np.radians((1.00554*xj - 6.28306))
    a2 = np.radians((1.93946*xj + 23.35089))
    et = -7.67825*np.sin(a1) - 10.09176*np.sin(a2)
    tsv = tsm +et/60.0
    tsv = tsv - 12.0
    #Hour angle
    ah = np.radians(tsv*15.0)
    #Solar Declination
    a3 = np.radians(0.9683*xj-78.00878)
    delta = np.radians(23.4856*np.sin(a3))

    #Solar Elevation
    amuzero = np.sin(xla)*np.sin(delta) + np.cos(xla)*np.cos(delta)*np.cos(ah)
    sun_elev = np.arcsin(amuzero)
    #Conversion to degrees
    sun_elev = np.degrees(sun_elev)
    sun_zen = 90.0 - sun_elev # by convention
    return sun_zen

#When training vs testing, you need to separate the loss functions
def EPE(x, y,mean=True):
    loss = torch.linalg.norm(x-y,2,1)
    batch_size = loss.size(0)
    if(mean):
        return loss.mean()
    else:
        return loss.sum()/batch_size
def realEPE (output, target):
    return EPE(output, target, mean=True)
#When co compose is initialized, it transforms the target and input using the
#same transformations, see in main below
class CoCompose(object):

    def __init__(self, co_transforms):
        self.co_transforms = co_transforms

    def __call__(self, input, target):
        for t in self.co_transforms:
            input,target = t(input,target)
        return input,target

#A scaling transformation using the interpolate function
class ScaleItOne(object):

    def __init__(self, scale_factor):
        self.scale_factor= scale_factor

    def __call__(self, inputs):
        _,h, w= inputs[0].shape
        h = int(h / self.scale_factor)
        w = int(w / self.scale_factor)
        if(inputs.dim() == 3):

            inputs = F.interpolate(inputs[None,:], (h,w), mode='area')
        else:

            inputs = F.interpolate(inputs, (h,w), mode='area')

        return torch.squeeze(inputs)

class ScaleIt(object):
    def __init__(self, scale_factor):
        self.scale_factor= scale_factor

    def __call__(self, inputs,target):
        h, w= inputs[0].shape
        h = int(h / self.scale_factor)
        w = int(w / self.scale_factor)
        if(target.dim() == 3):
            target = F.interpolate(target[None,:], (h,w), mode='area')
            inputs = F.interpolate(inputs[None,:], (h,w), mode='area')
        else:
            target = F.interpolate(target, (h,w), mode='area')
            inputs = F.interpolate(inputs, (h,w), mode='area')

        return torch.squeeze(inputs), torch.squeeze(target)
#A simple cropping transformation
class CropIt(object):

    def __init__(self, scale_factor):
        self.scale_factor = scale_factor
    def __call__(self, target):
        _,h,w = target.shape
        h2 = int(60/self.scale_factor)
        w2 = int(60/self.scale_factor)
        target = target[:,h2:h-h2,w2:w-w2]

        return target
class CropIt4D(object):

    def __init__(self, scale_factor):
        self.scale_factor = scale_factor
    def __call__(self, target):
        _,h,w = target[0].shape
        h2 = int(60/self.scale_factor)
        w2 = int(60/self.scale_factor)
        target = target[:,:,h2:h-h2,w2:w-w2]

        return target
class CropIt4D_Set(object):

    def __init__(self, parm):
        self.scale_factor = parm
    def __call__(self, target):
        _,h,w = target[0].shape
        h2 = int(self.scale_factor)
        w2 = int(self.scale_factor)
        target = target[:,:,h2:h-h2,w2:w-w2]

        return target
class WarpIt(object):
    def __init__(self,model,norm,inscale,outscale,image_transform,device):
        self.model = model
        self.norm = norm 
        self.image_transform = image_transform
        self.inscale = inscale
        self.outscale = outscale
        self.device = device

        
    def __call__(self, image,output):
        def warpfunc(images2,xout,norm,inscale):
            tHor = torch.linspace(-1.0 + (1.0 / xout.shape[3]), 1.0 - (1.0 / xout.shape[3]), xout.shape[3]).view(1, 1, 1, -1).repeat(1, 1, xout.shape[2], 1)
            tVer = torch.linspace(-1.0 + (1.0 / xout.shape[2]), 1.0 - (1.0 / xout.shape[2]), xout.shape[2]).view(1, 1, -1, 1).repeat(1, 1, 1, xout.shape[3])

            tenGrid = torch.cat([ tHor, tVer], 1).to(self.device)
            
            xout[:,0,:,:] = xout[:,0,:,:]/(xout.shape[2]/((2*norm/(inscale/2.)))) 
            xout[:,1,:,:] = xout[:,1,:,:]/(xout.shape[3]/((2*norm/(inscale/2.))))

            xout = tenGrid+xout


            x2 = (F.grid_sample(images2[:,:,:,:],xout[:,:,:,:].permute(0,2,3,1),padding_mode="reflection"))[:,:,:,:]

            images2[:,1,:,:] = x2[:,1,:,:] #just change the second image, not the first


            return images2
        #use the model to compute the optical flow, then warp the image with it
        output[:,0,:,:] -= image[:,2,:,:]
        output[:,1,:,:] -= image[:,3,:,:]
        scaleimage = ScaleItOne(self.inscale)
        images= scaleimage(image)
        images = self.image_transform(images)
        images = warpfunc(images,images[:,2:4,:,:],self.norm,self.inscale)
        xouti = self.model(images)



        scalexout = ScaleItOne(self.outscale/self.inscale)
        xout = scalexout(xouti)
        ofu = self.norm*(xout[:,0,:,:].clone())
        ofv = self.norm*(xout[:,1,:,:].clone())
        ofu = ofu[:,None,:,:]
        ofv = ofv[:,None,:,:]

        scaleoutput = ScaleItOne(self.outscale)
        output = scaleoutput(output)
        outputcrop = CropIt4D(2)
        outputcrop2 = CropIt4D(4)
        output = outputcrop(output)
        
        output = output - (xout)*self.norm
        output = outputcrop2(output)
        scaleimage2 = ScaleItOne(self.outscale)
        images2 = scaleimage2(image)
        images2 = outputcrop(images2) 
        images2 = warpfunc(images2,(xout+images2[:,2:4,:,:]),self.norm,self.inscale)
        images2[:,2,:,:] = images2[:,2,:,:] + ofu[:,0,:,:] 
        images2[:,3,:,:] = images2[:,3,:,:] + ofv[:,0,:,:] 
        return images2, output

#Will randomly flip an image in the horizontal with a probability of 0.5
class RandomHorizontalFlip(object):

    def __call__(self, inputs, target):
        if random.random() < 0.5:
            #note fliplr -> flip left to right
            inputs[0] = np.copy(np.fliplr(inputs[0]))
            inputs[1] = np.copy(np.fliplr(inputs[1]))
            target = np.copy(np.fliplr(target))
            target[0,:,:] *= -1
        return inputs,target

#Will randomly flip an image in the vertical with a probability of 0.5
class RandomVerticalFlip(object):

    def __call__(self, inputs, target):
        if random.random() < 0.5:
            #note flipud -> flip up to down
            inputs[0] = np.copy(np.flipud(inputs[0]))
            inputs[1] = np.copy(np.flipud(inputs[1]))
            target = np.copy(np.flipud(target))
            target[1,:,:] *= -1
        return inputs,target

#This is the of read function, reads the optical flow files and stores them in a BIG array
def jma_ofread_3D(fnames,docrop=False,doplot=False,sizex=512,dpix = 60,nx=500,ny=350):
    memsize = (sizex*sizex*len(fnames)*4.*2.)*1E-9
    print("Memory size required (GB)", memsize)
    datainit = True
    datalist = np.zeros((len(fnames),2,sizex+dpix+dpix,sizex+dpix+dpix),dtype=np.float32)
    counter = 0
    for files in fnames:
        print("Reading", files)
        if(doplot):
            #quick debug capability in doplot, check just one file...
            nc = netCDF4.Dataset(fnames[0])
        else:
            nc = netCDF4.Dataset(files)
        if(docrop):
            x1 = nx-dpix
            x2 = nx+sizex+dpix
            y1 = ny-dpix
            y2 = ny+sizex+dpix
            #create a padding function
            datalist[counter,0,:,:] = impad(np.squeeze(nc.variables['U_raw'][:]),x1,x2,y1,y2)
            datalist[counter,1,:,:] = impad(np.squeeze(nc.variables['V_raw'][:]),x1,x2,y1,y2)
            
        else:
            data_u = np.squeeze(nc.variables['U_raw'][:]) 
            data_v = np.squeeze(nc.variables['V_raw'][:]) 
        nc.close()
        counter += 1
        if doplot:
            break



    return datalist
def jma_satread_3D(fnames,docrop=False,doplot=False,sizex=512,dpix = 60,nx=500,ny=350):
    datalist = []
    memsize = (sizex*sizex*len(fnames)*4.)*1E-9
    datalist = np.zeros((len(fnames),sizex+dpix+dpix,sizex+dpix+dpix),dtype=np.float32)
    counter = 0
    
    for files in fnames:
        print("Satread",files)
        if(doplot):
            #quick debug capability in doplot
            nc = netCDF4.Dataset(fnames[0])
        else:
            nc = netCDF4.Dataset(files)
        if(docrop):
            x1 = nx-dpix
            x2 = nx+sizex+dpix
            y1 = ny-dpix
            y2 = ny+sizex+dpix
            datalist[counter,:,:] = impad(np.squeeze(nc.variables['Rad'][:]),x1,x2,y1,y2)
        else:
            data = np.squeeze(nc.variables['Rad'][:]) 
        nc.close()
        counter += 1
        if(doplot):
            break

    return datalist
#netcdf of file reader
#Transformation to convert the arrays in our model to pytorch tensors
class ArrayToTensor(object):
    def __init__(self, device):
        self.device = device
    def __call__(self, array, target):
        assert(isinstance(array, np.ndarray))

        tensor = torch.from_numpy(array).to(self.device)
        tensor2 = torch.from_numpy(target).to(self.device)
        return tensor, tensor2
class ArrayToGPU(object):
    def __init__(self, device):
        self.device = device
    def __call__(self, array, target):
        assert(isinstance(array, np.ndarray))
        tensor = torch.from_numpy(array).to(self.device)
        tensor2 = torch.from_numpy(target).to(self.device)
        return tensor, tensor2
#Creating a dataset for pytorch to read
class ImageDataset(Dataset):
    def __init__(self, image1dir, image2dir, ofdir, device, trained_model_dir = './trained_models',transform=None,target_transform=None,co_transform=None,docrop=False,doplot=False,dowarpit=False,dofullres=False,dosubset=False,ss1=0,ss2=1,nx=500,ny=350,sizex=1024):
        dp = 60*2 #pixel padding, hard coded in for now
        
        self.image1dir = image1dir #first image directory
        self.image2dir = image2dir #second image directory
        self.ofdir = ofdir #optical flow output directory
        if(dosubset):
            self.oflist = jma_ofread_3D(sorted(glob.glob(os.path.join(ofdir,'*')))[ss1:ss2],docrop=True,doplot=doplot, dpix = dp,sizex=sizex,nx=nx,ny=ny) 
            self.image1list = jma_satread_3D(sorted(glob.glob(os.path.join(image1dir,'*')))[ss1:ss2],docrop=True,doplot=doplot,dpix = dp,sizex=sizex,nx=nx,ny=ny) 
            self.image2list = jma_satread_3D(sorted(glob.glob(os.path.join(image2dir,'*')))[ss1:ss2],docrop=True,doplot=doplot,dpix = dp,sizex=sizex,nx=nx,ny=ny) 
            self.filenames = sorted(glob.glob(os.path.join(image1dir,'*')))[ss1:ss2]
            self.trainnames = sorted(glob.glob(os.path.join(ofdir,'*')))[ss1:ss2]
        else:
            self.oflist = jma_ofread_3D(sorted(glob.glob(os.path.join(ofdir,'*'))),docrop=True,doplot=doplot, dpix = dp,sizex=sizex,nx=nx,ny=ny) 
            self.image1list = jma_satread_3D(sorted(glob.glob(os.path.join(image1dir,'*'))),docrop=True,doplot=doplot,dpix = dp,sizex=sizex,nx=nx,ny=ny) 
            self.image2list = jma_satread_3D(sorted(glob.glob(os.path.join(image2dir,'*'))),docrop=True,doplot=doplot,dpix = dp,sizex=sizex,nx=nx,ny=ny) 
            self.filenames = sorted(glob.glob(os.path.join(image1dir,'*')))
            self.trainnames = sorted(glob.glob(os.path.join(ofdir,'*')))
        self.transform = transform
        self.target_transform = target_transform
        self.co_transform = co_transform
        self.device = device
        self.docrop = docrop
        self.doplot = doplot
        self.dowarpit = dowarpit
        if(dowarpit):
            upmodel = AIRWOLF_NET().to('cpu')
            checkpoint = torch.load(os.path.join(trained_model_dir,'airwolf_level1.pth'))
            upmodel.load_state_dict(checkpoint['model_state_dict'])
            #hard coded normalization coefficients, set to 0.64 um min/max values, std_dev = 1
            t1 = transforms.Compose([
                transforms.Normalize(mean=[-20.28991,-20.28991,0.,0.], std=[649.27711,649.27711,1.,1.]),
                transforms.Normalize(mean=[0.5,0.5,0.,0.], std=[1.,1.,1.,1.])
            ])
            if(dofullres):
                upmodel2 = AIRWOLF_NET().to('cpu')
                checkpoint = torch.load(os.path.join(trained_model_dir,'airwolf_level2.pth'))
                upmodel2.load_state_dict(checkpoint['model_state_dict'])
                xwarp = WarpIt(upmodel,2,4,2,t1, 'cpu')
                a2t = ArrayToTensor('cpu')
                ofug = np.copy(self.image1list)
                ofvg = np.copy(self.image1list)
                ofug[:,:,:]= 0
                ofvg[:,:,:]= 0
                imtemp = np.transpose(np.asarray([self.image1list,self.image2list,ofug,ofvg]),(1,0,2,3))
                imtemp, oftemp = a2t(imtemp,self.oflist)

                imtemp, oftemp = xwarp(imtemp,oftemp)
                x2warp = WarpIt(upmodel2,1,2,1,t1, 'cpu')
                scof = ScaleItOne(0.5)
                imtemp = scof(imtemp)
                cropob = CropIt4D_Set(15)
                imtemp = cropob(imtemp)
                self.image1list = np.transpose(np.asarray([self.image1list,self.image2list,ofug,ofvg]),(1,0,2,3))
                self.image1list, self.oflist = a2t(self.image1list,self.oflist)
                cropob2 = CropIt4D_Set(75)

                self.image1list = cropob2(self.image1list)
                self.oflist = cropob2(self.oflist)
                self.image1list[:,2,:,:] = imtemp[:,2,:,:]
                self.image1list[:,3,:,:] = imtemp[:,3,:,:] 

                self.image1list, self.oflist = x2warp(self.image1list,self.oflist)
            

            else:
                xwarp = WarpIt(upmodel,2,4,2,t1, 'cpu')
                a2t = ArrayToTensor('cpu')
                ofug = np.copy(self.image1list)
                ofvg = np.copy(self.image1list)
                ofug[:,:,:]= 0
                ofvg[:,:,:]= 0
                self.image1list = np.transpose(np.asarray([self.image1list,self.image2list,ofug,ofvg]),(1,0,2,3))
                self.image1list, self.oflist = a2t(self.image1list,self.oflist)
                self.image1list, self.oflist = xwarp(self.image1list,self.oflist)
            

    def __len__(self):
        #this is a function to manage when the len() function is called on our object
        return len(self.oflist)

    def __getitem__(self,idx):
        #this is a function to manage when the getitem(index) function is called on our object
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image1 = self.image1list[idx] 
        image2 = self.image2list[idx] 

        if(self.dowarpit):
            image = self.image1list[idx].detach().numpy()
            ofvals = self.oflist[idx].detach().numpy()
        else:
            ofguessu = np.copy(image1)
            ofguessv = np.copy(image1)
            ofguessu[:,:] = 0
            ofguessv[:,:] = 0
            image = np.asarray([image1,image2,ofguessu,ofguessv])
            ofvals = self.oflist[idx] 
            
        #here is where the transformations are implemented on our datasets
        if self.co_transform:
            image, ofvals = self.co_transform(image, ofvals)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            ofvals = self.target_transform(ofvals)
        sample = {'image':image,'ofvals':ofvals,'Files':self.filenames[idx],'Truth':self.trainnames[idx]}
        return sample




class AIRWOLF_NET(nn.Module):
    def __init__(self):
        super(AIRWOLF_NET, self).__init__()
        self.netBasic = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=4, out_channels=32, kernel_size=7, stride=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=7, stride=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=7, stride=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=32, out_channels=16, kernel_size=7, stride=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=16, out_channels=2, kernel_size=7, stride=1)
        )
    def forward(self, image):
        return self.netBasic(image)
def train_loop(dataloader, model, optimizer):
    size = len(dataloader.dataset)
    lastloss = 0.
    bf = 0
    bf2 = 0
    model.train() 
    for batch, Imagedat in enumerate(dataloader):
        pred = model(Imagedat['image'])
        loss = realEPE(pred,Imagedat['ofvals'])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss, current = loss.item(), batch * len(Imagedat['image'])
        print(f"loss: {(loss*2):>7f}  [{current:>5d}/{size:>5d}]")
        lastloss += loss
        bf = batch
        bf2 += 1.
    print("Avg image loss",(2.*lastloss/bf2),bf2,bf)
    return (lastloss/bf2)
def test_loop(dataloader, model):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss = 0.
    model.eval() 
    bs = 0.
    with torch.no_grad():
        for X in dataloader:
            pred = model(X['image'])
            tl = realEPE(pred,X['ofvals'])
            print(f"Test loss: {(2*tl):>8f}")
            test_loss += tl 
            bs += 1.
    test_loss /= num_batches
    print(f"Test Avg loss: {test_loss:>8f} \n")
    return test_loss.cpu()
if __name__ == '__main__':
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    doplot = False
    runcase = True #set to true if your parent directory has a case study you wish to test
    learning_rate = 1e-4
    batch_size = 128
    trainlevel = 1 #there are 3 training levels, 1 is the first, 2 is the middle, 3 is the last
    dowarpit = False
    dofullres = False

    if(runcase):
        dosubset=True
        trainlevel = 3 #this will always run the last level
        dpix = 120
        sizex = 2048
        nx = -24
        ny = -24 #note, these are for a 2000x2000 pixel image
    else:
        dpix = 120
        sizex = 1024 #note it was this before...
        nx = 800
        ny = 800 #note, these are for a 2000x2000 pixel image
        dosubset=False

    if(trainlevel >= 2):
        dowarpit = True #use this to train/use Level 2
    if(trainlevel >=3):
        dofullres = True #use this to train/use Level 3 with dowarpit = True

    if(doplot):
        device = "cpu"
    learning_rate = 1e-4
    batch_size = 128
    epochs = 4000 #level 1 needs many epochs to converge
    lvlname = 'level1'
    if(dowarpit):
        epochs = 250
        lvlname = 'level2'
    if(dowarpit & dofullres):
        epochs = 100 #hyperparameter, epoch training options, usually 100 was enough for level 3
        lvlname = 'level3'
    trained_model_name = './trained_models_test/airwolf_'+lvlname+'.pth'
    #the parent directory needs a specific structure to work here:
    #There must be an IMAGE1, IMAGE2, and TRAINING directory
    #IMAGE1 and IMAGE2 contain the first and second images used in the optical flow computation
    #in this case, it is designed to work with 0.64 um L1b netcdf files from the GOES-R ABI
    #TRAINING contains the output netcdfs from OCTANE: github.com/JasonApke/OCTANE
    #Files should be named YYYYMMDD-hhmmss*.nc
    #Make sure to change this to the parent directory with your case study or training data!
    airwolf_dir = '/mnt/data1/japke/AIRWOLF/'
    if(runcase):
        parentdir = os.path.join(airwolf_dir,'Case_Studies/20230216/')
        testparentdir = os.path.join(airwolf_dir,'Case_Studies/20230216/')
    else:
        parentdir = os.path.join(airwolf_dir,'UNET_Train/')
        testparentdir = os.path.join(airwolf_dir,'UNET_Test/')
        statedir = os.path.join(airwolf_dir,'Model_States/')
        savestates=True

    t1 = transforms.Compose([
        transforms.Normalize(mean=[-20.28991,-20.28991,0.,0.], std=[649.27711,649.27711,1.,1.]),
        transforms.Normalize(mean=[0.5,0.5,0.,0.], std=[1.,1.,1.,1.])
    ])
    if(dowarpit):
        t2 = transforms.Compose([
            transforms.Normalize(mean=[0.,0.], std=[1.,1.])
        ])
    else:
        t2 = transforms.Compose([
            transforms.Normalize(mean=[0.,0.], std=[2.,2.]),
            CropIt(4)
        ])
    sf = 2
    if(dowarpit):
        ct = CoCompose([
            ArrayToGPU(device=device)
        ])
        t2 = None #Shut off transforms now, they are done within warpit as needed
    else:   
        ct = CoCompose([
            ArrayToTensor(device=device),
            ScaleIt(4)
        ])
    of_dataset =   ImageDataset(image1dir=parentdir+'IMAGE1',image2dir=parentdir+'IMAGE2',ofdir=parentdir+'TRAINING',transform=t1, target_transform=t2, co_transform=ct, device=device,docrop=True,doplot=doplot,dowarpit=dowarpit,dofullres=dofullres,dosubset=dosubset,ss2=3,nx=nx,ny=ny,sizex=sizex)

    train_dataloader = DataLoader(of_dataset, batch_size=batch_size,shuffle=True)
    if(runcase==False):
        test_dataset = ImageDataset(image1dir=testparentdir+'IMAGE1',image2dir=testparentdir+'IMAGE2',ofdir=testparentdir+'TRAINING',transform=t1, target_transform=t2, co_transform=ct, device=device,docrop=True,doplot=doplot,dowarpit=dowarpit,dofullres=dofullres,dosubset=dosubset,ss2=3,nx=nx,ny=ny,sizex=sizex)
        test_dataloader = DataLoader(test_dataset,batch_size=batch_size,shuffle=False)

    model = AIRWOLF_NET().to(device)
    if(runcase):
        checkpoint = torch.load('./trained_models/airwolf_level3.pth')
        model.eval()
        model.load_state_dict(checkpoint['model_state_dict'])
        for i in range(0,len(of_dataset)):
            x = of_dataset[i]['image']
            y = of_dataset[i]['ofvals']

            imname = os.path.basename(of_dataset[i]['Files'])[:-len('.nc')]+'.png'
            geo = jma_goesread_uv(of_dataset[i]['Truth'],cal='REF')
            dopad = True
            if(dopad==False):
                geo.data = geo.data[(nx):(nx+sizex),(ny):(ny+sizex)]
                geo.x = geo.x[(nx):(nx+sizex),(ny):(ny+sizex)]
                geo.y = geo.y[(nx):(nx+sizex),(ny):(ny+sizex)]
                geo.lat = geo.lat[(nx):(nx+sizex),(ny):(ny+sizex)]
                geo.lon = geo.lon[(nx):(nx+sizex),(ny):(ny+sizex)]
                geo.u = geo.u[(nx):(nx+sizex),(ny):(ny+sizex)]
                geo.v = geo.v[(nx):(nx+sizex),(ny):(ny+sizex)]
            else:
                geo.data = geo.data
                geo.x =       geo.x
                geo.y =       geo.y
                geo.lat =   geo.lat
                geo.lon =   geo.lon
                geo.u =       geo.u
                geo.v =       geo.v
            x = x[None,:]
            y = y[None,:]
            xout = model(x)

            cpo = CropIt(4)
            x = cpo(x[0,:,:,:])
            x = x[None,:]

            xout = (xout.cpu()).detach().numpy()
            #lossnp = loss.detach().numpy()
            ynp = (y.cpu()).detach().numpy()
            xnp = (x.cpu()).detach().numpy()
            ut = xnp[0,2,:,:]+ynp[0,0,:,:] #truth winds
            vt = xnp[0,3,:,:]+ynp[0,1,:,:] #truth winds
            ug = xnp[0,2,:,:]+xout[0,0,:,:] #guess winds
            vg = xnp[0,3,:,:]+xout[0,1,:,:] #guess winds
            #navigation function below, if needed
            uguess, vguess = jma_pixeluv_uv(geo,ug[24:2024,24:2024],vg[24:2024,24:2024])
            dt = datetime.datetime.strptime(imname[0:15],'%Y%m%d-%H%M%S') #assumes the filename is YYYYMMDD-hhmmss
            timestring = datetime.datetime.strftime(dt,'%b %d, %Y %H:%M:%S UTC')
            fig, axs = plt.subplots(2,2)
            fig.suptitle('AIRWOLF Test 0.64-$\mu$m '+timestring,x=0.5,y=0.93,fontsize=8)

            # Below is the True optical flow field
            ax = axs[0,0]
            strm = ax.imshow(-1*(xnp[0,2,:,:]+ynp[0,0,:,:]),vmin=-5.,vmax=5., cmap=plt.cm.coolwarm)
            ax.xaxis.set_tick_params(labelsize=8)
            ax.yaxis.set_tick_params(labelsize=8)
            
            # below is the model estimated optical flow field
            cbar = fig.colorbar(strm,pad=0.03,shrink=0.6,ax=ax)
            cbar.ax.tick_params(labelsize=6)
            cbar.set_label("U Truth (px/min)",fontsize=8) #,labelpad=-1.)
            ax = axs[0,1]
            strm2 = ax.imshow(-1*(xnp[0,2,:,:]+xout[0,0,:,:]),vmin=-5.,vmax=5., cmap=plt.cm.coolwarm)
            ax.xaxis.set_tick_params(labelsize=8)
            ax.yaxis.set_tick_params(labelsize=8)
            cbar2 = fig.colorbar(strm2,pad=0.03,shrink=0.6,ax=ax)
            cbar2.ax.tick_params(labelsize=6)
            cbar2.set_label("U Model (px/min)",fontsize=8) #,labelpad=-1.)
            ax = axs[1,0]
            wp = 200
            cond1 = (geo.x % wp == 0) & (geo.y % wp == 0)
            xarr = geo.x-np.amin(geo.x)
            yarr = geo.y-np.amin(geo.y)
            gx = xarr[cond1]
            gy = yarr[cond1]
            utr = geo.u[cond1]
            vtr = geo.v[cond1]
            uguess = uguess[cond1]
            vguess = vguess[cond1]

            strm3 = ax.imshow(geo.data,vmin=0,vmax=0.6, cmap=plt.cm.gray)
            ax.xaxis.set_tick_params(labelsize=8)
            ax.yaxis.set_tick_params(labelsize=8)
            ax.barbs(gx,gy,uguess*1.94384,vguess*1.94384,length=5,color='yellow')
            cbar3 = fig.colorbar(strm3,pad=0.03,shrink=0.6,ax=ax)
            cbar3.ax.tick_params(labelsize=6)
            cbar3.set_label("Imagery",fontsize=8) #,labelpad=-1.)

            diff = np.sqrt(((ut-ug)**2.)+((vt-vg)**2.))
            ax = axs[1,1]
            strm4 = ax.imshow(diff,vmin=0,vmax=5., cmap=plt.cm.gist_rainbow_r)
            ax.xaxis.set_tick_params(labelsize=8)
            ax.yaxis.set_tick_params(labelsize=8)
            ax.barbs(gx,gy,(utr-uguess)*1.94384,(vtr-vguess)*1.94384,length=5,color='black')
            cbar4 = fig.colorbar(strm4,pad=0.03,shrink=0.6,ax=ax)
            cbar4.ax.tick_params(labelsize=6)
            cbar4.set_label("VDM (px/min)",fontsize=8) #,labelpad=-1.)
            plt.savefig(parentdir+'FIGS/four_panel/'+imname,format='png',bbox_inches='tight', dpi=150)
            plt.close()
            print("Plotted "+parentdir+'FIGS/four_panel/'+imname)

    if(doplot):
        dohist=False
        checkpoint = torch.load('./trained_models/airwolf_level3.pth')
        model.eval()
        model.load_state_dict(checkpoint['model_state_dict'])
        x = test_dataset[0]['image']
        y = test_dataset[0]['ofvals']
        x = x[None,:]
        y = y[None,:]
        xout = model(x)

        cpo = CropIt(4)
        x = cpo(x[0,:,:,:])
        x = x[None,:]

        tHor = torch.linspace(-1.0 + (1.0 / xout.shape[3]), 1.0 - (1.0 / xout.shape[3]), xout.shape[3]).view(1, 1, 1, -1).repeat(1, 1, xout.shape[2], 1)
        tVer = torch.linspace(-1.0 + (1.0 / xout.shape[2]), 1.0 - (1.0 / xout.shape[2]), xout.shape[2]).view(1, 1, -1, 1).repeat(1, 1, 1, xout.shape[3])
        tenGrid = torch.cat([ tHor, tVer], 1)
        bins = 30
        xout = xout.detach().numpy()
        ynp = y.detach().numpy()
        xnp = x.detach().numpy()
        plt.imshow(xnp[0,2,:,:]+ynp[0,0,:,:],vmin=-2.,vmax=2., cmap=plt.cm.coolwarm)
        cbar=plt.colorbar(orientation='horizontal',pad=0.075,shrink=0.6)
        cbar.ax.tick_params(labelsize=8)
        cbar.set_label("U Motion",fontsize=8) #,labelpad=-1.)


        plt.savefig('./airwolf_level3_test.png',format='png',bbox_inches='tight', dpi=150)
    #if neither doplot or docase are true, then we are in train mode...
    if(runcase == False):
        optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
        lossval = []
        epochval = []
        testlossval = []
        for t in range(epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            loss = train_loop(train_dataloader, model, optimizer)
            lossval.append(loss)
            testloss = test_loop(test_dataloader, model).detach().numpy()
            testlossval.append(testloss)
            epochval.append(t+1)
            plt.plot(epochval,lossval,label='Training')
            plt.plot(epochval,testlossval,label='Testing')
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.legend()
            plt.savefig('./lossfig_airwolf_level3.png',format='png')
            plt.close()
            #save each model at a specific state every 10 epochs
            if(savestates):
                if(t % 10 == 0):
                    torch.save({
                        'epoch': t+1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': lossval,
                        'testloss': testlossval,
                        'epochval': epochval
                        }, os.path.join(statedir,'airwolf'+lvlname+'_epoch'+str(t).zfill(4)+'.pth'))



        print("Done, saving the model to "+trained_model_name)
        torch.save({
            'epoch': epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': lossval,
            'testloss': testlossval,
            'epochval': epochval
            }, trained_model_name)
