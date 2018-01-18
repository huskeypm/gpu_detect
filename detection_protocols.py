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
        
    ## Determine SNR by comparing integrated area with corrlobe response 
    snr = integrated/corrlobe ##* corrThreshed
    #snr = corrThreshed

    #plt.colorbar()
    #plt.gcf().savefig(name,dpi=300)

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

    return results





# TODO phase this out 
import util
def CalcInvFilter(inputs,paramDict,corr):
    
      penaltyscale = paramDict['penaltyscale'] 
      sigma_n  = paramDict['sigma_n']
      angle  = paramDict['angle']
      tN = inputs.img
      filterRef = inputs.mfOrig
      yP = corr
    
      s=1.  
      fInv = np.max(filterRef)- s*filterRef
      rFi = util.PadRotate(fInv,angle)
      rFiN = util.renorm(np.array(rFi,dtype=float),scale=1.)
      yInv  = mF.matchedFilter(tN,rFiN,demean=False,parsevals=True)   
      
      # spot check results
      #hit = np.max(yP) 
      #hitLoc = np.argmax(yP) 
      #hitLoc =np.unravel_index(hitLoc,np.shape(yP))

      ## rescale by penalty 
      # part of the problem earlier was that the 'weak' responses of the 
      # inverse filter would amplify the response, since they were < 1.0. 
      yPN =  util.renorm(yP,scale=1.)
      yInvN =  util.renorm(yInv,scale=1.)

      yPN = np.exp(yPN)
      yInvS = sigma_n*penaltyscale*np.exp(yInvN)
      scaled = np.log(yPN/(yInvS))
    
      return scaled 



# This script determines detections by integrating the correlation response
# over a small area, then dividing that by the response of a 'lobe' filter 
# need to write this in paper, if it works 
def dcDetect(
  inputs,    # data sets, filters etc 
  paramDict  # dictionary of parameters needed for detection
  ):
    # get data 
    img = inputs.img # raw (preprocessed image) 
    mf  = inputs.mf  # raw (preprocessed image) 

    ## get correlation plane w filter 
    corr = mF.matchedFilter(img,mf,parsevals=False,demean=False)

    results = empty()
    return results 

#
# Original detection procedure included with PNP paper
#
def simpleDetect(
  inputs,    # data sets, filters etc 
  paramDict  # dictionary of parameters needed for detection
  ):
  # get data 
  img = inputs.img # raw (preprocessed image) 
  mf  = inputs.mf  # raw (preprocessed image) 

  ## get correlation plane w filter 
  results = empty()
  results.corr = mF.matchedFilter(img,mf,parsevals=False,demean=True) 
  
  if paramDict['useFilterInv']:
      results.snr = CalcInvFilter(inputs,paramDict,results.corr)

  else:
      results.snr = results.corr  /paramDict['sigma_n']



  return results

#
# Calls different modes of selecting best hits 
#
def FilterSingle(
  inputs, # object with specific filters, etc needed for matched filtering
  paramDict = dict(),# pass in parameters through here
  mode = None # ['logMode','dylanmode','lobemode']
  ):
  if mode is not None:
      print "WARNING: replacing with paramDict"
    
  mode = paramDict['filterMode']  
  if mode=="lobemode":
    results = lobeDetect(inputs,paramDict)
  elif mode=="dcmode": 
    results = dcDetect(inputs,paramDict)
  elif mode=="simple":
    results = simpleDetect(inputs,paramDict)
  else: 
    #raise RuntimeError("need to define mode") 
    print "Patiently ignoring you until this is implemented" 
    results = empty()
  return results



