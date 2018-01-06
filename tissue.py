"""
For processing large tissue subsection from Frank
"""
# SPecific to a single case, so should be moved elsewhere
import matchedFilter as mF 
import numpy as np
import matplotlib.pylab as plt
class empty:pass


## Based on image characteristics 
#params = empty()
#params.dim = np.shape(gray)
#params.fov = np.array([3916.62,4093.31]) # um (from image caption in imagej. NOTE: need to reverse spatial dimensions to correspond to the way cv2 loads image)
#params.px_per_um = params.dim/params.fov

class Params:pass
params = Params()
params.imgName = "/home/AD/pmke226/DataLocker/cardiac/Sachse/171127_tissue/tissue.tif"
params.fov = np.array([3916.62,4093.31]) # um (from image caption in imagej. NOTE: need to reverse spatial dimensions to correspond to the way cv2 loads image)


import cv2
import util

def Setup():
  img = cv2.imread(params.imgName)
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  params.dim = np.shape(gray)
  params.px_per_um = params.dim/params.fov

  return gray

def SetupTest():
  case=empty()
#case.loc_um = [2366,1086] 
  case.loc_um = [2577,279] 
  case.extent_um = [150,150]
  SetupCase(case)
  return case
  
def SetupCase(case):
  gray = Setup()
  case.subregion = get_fiji(gray,case.loc_um,case.extent_um)

def SetupFilters(rot=20.):
  mfr = CreateFilter(params,rot=rot)
  lobemfr = CreateLobeFilter(params,rot=rot)

  params.mfr=mfr
  params.lobemfr=lobemfr




#print dim
#print px_per_um


# Functions for rescaling 'fiji' coordinates for image to those used by cv2
def conv_fiji(x_um,y_um): # in um
    y_px = int(x_um * params.px_per_um[1])
    x_px = int(y_um * params.px_per_um[0])
    return x_px,y_px

def get_fiji(gray, loc_um,d_um):
    loc= conv_fiji(loc_um[0],loc_um[1])
    d = conv_fiji(d_um[0],d_um[1])
    subregion = gray[loc[0]:(loc[0]+d[0]),loc[1]:(loc[1]+d[1])]
    return subregion

import display_util as du
def dbgimg(case,results,
           orig=True,
           thrsh=True,
           corr=True,
           corrLobe=True,
           snr=True,
           merged=True,
           mergedSmooth=True,
           ul=[0,0],
           lr=[100,100]
           ):
    l,r=ul
    ll,rr=lr
    
    if orig:
        plt.figure()
        plt.imshow(case.subregion[l:r,ll:rr],cmap="gray")
        plt.colorbar()

    if thrsh:    
        plt.figure()
        plt.imshow(results.img[l:r,ll:rr],cmap="gray")
        plt.colorbar()

    if corr:
        plt.figure()
        plt.imshow(results.corr[l:r,ll:rr],cmap="gray")
        plt.colorbar()

    if corrLobe:    
        plt.figure()
        plt.imshow(results.corrlobe[l:r,ll:rr],cmap="gray")
        plt.colorbar()

    if snr:
        plt.figure()
        plt.imshow(results.snr[l:r,ll:rr],cmap="gray")
        plt.colorbar()

    if merged:
        plt.figure()
        #stackem(results.corr[550:750,550:750],results.corrlobe[550:750,550:750])
        du.StackGrayRedAlpha(case.subregion[l:r,ll:rr],results.threshed[l:r,ll:rr])

    if mergedSmooth:
        plt.figure()
        #sadf = sadf>50
        DisplayHits(case.subregion[l:r,ll:rr],results.threshed[l:r,ll:rr])
        plt.colorbar()     

def DisplayHits(img,threshed):
        daround=np.ones([40,40])
        sadf=mF.matchedFilter(threshed,daround,parsevals=False,demean=False)
        du.StackGrayRedAlpha(img,sadf)


  

import imutils
    
def CreateFilter(
    params,
#    rot = 30. # origin
    rot = 22. # hard
):  
    vert = int(3.5 * params.px_per_um[0])
    horz = int(2.1 * params.px_per_um[1])
    w = 2/2 # px (since tiling)
    mf = np.zeros([vert,horz])
    mf[:,0:w]=1.
    mf[:,(horz-w):horz]=1.

    # pad both sides (makes no diff)
    #pad = 2
    #mfp = np.zeros([vert,horz+2*pad])
    #mfp[:,pad:(horz+pad)] = mf
    #mf = mfp

    # multiple tiles
    mf = np.tile(mf, (1, 3))
    #imshow(mf,cmap='gray')


    #plt.figure()
    mfr = imutils.rotate_bound(mf,-rot)
    #imshow(mfr,cmap='gray')

    return mfr

def CreateLobeFilter(params,rot=22.):
  # test with embedded signal
  vert = int(3.5 * params.px_per_um[0])
  sigW = 4  # px
  lobeW = 4 # px
  lobemf = np.ones([vert,lobeW + sigW + lobeW])
  lobemf[:,lobeW:(lobeW+sigW)]=0.

#imshow(mf,cmap='gray')
  lobemfr = imutils.rotate_bound(lobemf,-rot)
  # imshow(lobemfr,cmap='gray')
  return lobemfr



def docalc(imgOrig,
           mf,
           rawFloor = 1., # minimum value to image (usually don't chg)
           eps = 4., # max intensity for TTs
           smoothScale=3, # number of pixels over which real TT should respond
           snrThresh = 70000,
           lossScale = 10, # " over which loss region should be considered
           lossRegionCutoff = 40,
           lobemf=None,
           corrThresh=0.,
           s=1.,name="corr.png"):
    results = empty()

    # correct image so TT features are brightest
    # set noise floor 
    img    = np.copy(imgOrig)              
    img[imgOrig<rawFloor]=rawFloor        
    img[ np.where(imgOrig> eps)] =eps # enhance so TT visible


    

    ## get correlation plane w filter 
    corr = mF.matchedFilter(img,mf,parsevals=False,demean=False)
    plt.subplot(2,2,1)
    plt.imshow(img,cmap='gray')
    #plt.colorbar()

    plt.subplot(2,2,3)
    plt.imshow(corr)
    #plt.colorbar()
    #plt.gcf().savefig("x.png",dpi=300)

    ##corrThreshed = np.zeros_like(img)
    ##corrThreshed[np.where(img>corrThresh)] = 1.

    ## integrate correlation plane over XxX interval
    smoother = np.ones([smoothScale,smoothScale])
    integrated =mF.matchedFilter(corr,smoother,parsevals=False,demean=False)

    ## get side lobe penalty
    if isinstance(lobemf,np.ndarray):
        #corrlobe = np.ones_like(corr)
        #out = mF.matchedFilter(corr,lobemf,parsevals=True,demean=False)
        #out -= np.min(out)
        #out /= np.max(out)
        #corrlobe += s*out
        corrlobe = mF.matchedFilter(corr,lobemf,parsevals=True,demean=False)
        
    else:    
        corrlobe = np.ones_like(corr)
        
    # zero out subthresh components
    plt.subplot(2,2,4)
    plt.imshow(corrlobe)
    
    snr = integrated/corrlobe ##* corrThreshed
    #snr = corrThreshed

    #plt.colorbar()
    #plt.gcf().savefig(name,dpi=300)

    ## thresh
    threshed = snr > snrThresh

    ## make  loss mask, needs a threshold for defining maximum value a 
    ## region can take before its no longer a considered a loss region 
    lossFilter = np.ones([lossScale,lossScale])
    losscorr = mF.matchedFilter(img,lossFilter,parsevals=False,demean=False)
    results.asdf = losscorr       
    lossRegion = losscorr < lossRegionCutoff
    mask = 1-lossRegion
    results.lossFilter = mask
   

    plt.subplot(2,2,2)
    #plt.imshow(snr*mask)
    DisplayHits(imgOrig,threshed)                 


    
    results.img = img
    results.corr = corr
    results.corrlobe = corrlobe
    results.snr = snr
    results.threshed = threshed
    
    return results

def Test1():
  import cv2
  import util
  class empty:pass
  cases = dict()
  
  case=empty()
  #case.loc_um = [2366,1086] 
  case.loc_um = [2577,279] 
  case.extent_um = [150,150]
  
  cases['hard'] = case
  
  case=SetupTest()
  
  SetupFilters()
  
  results=docalc(case.subregion,
                        params.mfr,lobemf=params.lobemfr,
                        snrThresh=42000)

  plt.figure()
  DisplayHits(case.subregion,results.threshed)                 
  plt.gcf().savefig("output.png",dpi=300)
  
  
  
  
  
#Test1()
  
