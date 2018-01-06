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

    plt.subplot(2,2,2)
    plt.imshow(snr*mask)
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
   


    
    results.img = img
    results.corr = corr
    results.corrlobe = corrlobe
    results.snr = snr
    results.threshed = threshed
    
    return results

