"""
Packages routines used to determine if correlation response
constitutes a detection
"""

import numpy as np 
import matchedFilter as mF
class empty:pass

# This script determines detections by integrating the correlation response
# over a small area, then dividing that by the response of a 'lobe' filter 
# need to write this in paper, if it works 
def lobeDetect(
  inputs,    # data sets, filters etc 
  paramDict  # dictionary of parameters needed for detection
  ):
    # get data 
    img = inputs.img # raw (preprocessed image) 
    mf  = inputs.mf  # raw (preprocessed image) 
    lobemf  = inputs.lobemf  # raw (preprocessed image) 
    results = empty()

    ## get correlation plane w filter 
    corr = mF.matchedFilter(img,mf,parsevals=False,demean=False)

    ## integrate correlation plane over XxX interval
    smoothScale = paramDict['smoothScale']
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
        
    snr = integrated/corrlobe ##* corrThreshed
    #snr = corrThreshed

    #plt.colorbar()
    #plt.gcf().savefig(name,dpi=300)

    ## thresh
    threshed = snr > paramDict['snrThresh']

    ## make  loss mask, needs a threshold for defining maximum value a 
    ## region can take before its no longer a considered a loss region 
    lossScale = paramDict['lossScale']
    lossFilter = np.ones([lossScale,lossScale])
    losscorr = mF.matchedFilter(img,lossFilter,parsevals=False,demean=False)
    lossRegion = losscorr < paramDict['lossRegionCutoff']
    mask = 1-lossRegion
    results.lossFilter = mask


    
    ## 
    ## Storing 
    ## 
    results.img = img
    results.corr = corr
    results.corrlobe = corrlobe
    results.snr = snr
    results.threshed = threshed
 
    return results

