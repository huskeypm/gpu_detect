from copy import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.mlab as mlab
import cv2
from scipy.misc import toimage
from scipy.ndimage.filters import *
import matchedFilter as mF
import imutils
from imtools import *
from matplotlib import cm
import detection_protocols as dps

class empty:pass

##
## Performs matched filtering over desired angles
##
def correlateThresher(
        inputs,
        params,
        iters = [0,30,60,90],  
        printer = True, 
        filterMode=None,
        label="undef",
        ):

    # TODO - this should be done in preprocessing, not here
    #print "PKH: turn into separate proproccessing routine"
    img = inputs.imgOrig
    
    if params['doCLAHE']:
      if img.dtype != 'uint8':
        myImg = np.array((img * 255),dtype='uint8')
      else:
        myImg = img
      clahe99 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16,16))
      img = clahe99.apply(myImg)

    filterRef = util.renorm(np.array(inputs.mfOrig,dtype=float),scale=1.)
    
    ##
    ## Iterate over all filter rotations desired 
    ## TODO - here is the place to stick in the GPU shizzle 
    ## 
    # Store all 'hits' at each angle 
    correlated = []

    inputs.img = util.renorm(np.array(img,dtype=float),scale=1.)
    inputs.demeanedImg = np.abs(np.subtract(inputs.img, np.mean(inputs.img)))
    #print np.max(inputs.demeanedImg)
    #print np.min(inputs.demeanedImg)
    for i, angle in enumerate(iters):
      result = empty()
      # copy of original image 

      # pad/rotate 
      params['angle'] = angle
      rFN = util.PadRotate(filterRef,angle)  
      inputs.mf = rFN  
      
      # matched filtering 
      result = dps.FilterSingle(inputs,params)      

      # store. Results contain both correlation plane and snr
      result.rFN = np.copy(rFN)
      correlated.append(result) 

    ##
    ## write
    ##
    #if label!=None and printer:
    if printer: 
       if label==None:
         label="undef"
       if filterMode==None:
         filterMode="undef"

       for i, angle in enumerate(iters):
        tag = filterMode 
        daTitle = "rot %4.1f "%angle # + "hit %4.1f "%hit 

        result = correlated[i]   
        plt.figure()
        plt.subplot(1,2,1)
        plt.title("Rotated filter") 
        plt.imshow(result.rFN,cmap='gray')
        plt.subplot(1,2,2)
        plt.title("Correlation plane") 
        plt.imshow(result.corr)                
        plt.colorbar()
        plt.tight_layout()
        fileName = label+"_"+tag+'_{}.png'.format(angle)
        plt.gcf().savefig(fileName,dpi=300)
        plt.close()

    return correlated  # a list of objects that contain SNR, etc 

def CalcSNR(signalResponse,sigma_n=1):
  print "PHASE ME OUT" 
  return signalResponse/sigma_n

import util 
import util2
##
## Collects all hits above some criterion for a given angle and 'stacks' them
## into a single image
##
def StackHits(correlated,  # an array of 'correlation planes'
              paramDict, # threshold,
              iters,
              display=False,
              rescaleCorr=False,
              doKMeans=False, #True,
              returnAngles=False):
     # TODO
    if rescaleCorr:
      raise RuntimeError("Why is this needed? IGNORING")
    if display:
      print "Call me debug" 
    # Function that iterates through correlations at varying rotations of a single filter,
    # constructs a mask consisting of 'NaNs' and returns a list of these masked correlations


    ##
    ## select hits based on those entries about the snrThresh 
    ##
    maskList = []
    for i, iteration in enumerate(iters):
        # routine for identifying 'unique' hits
        try:
          daMask = util2.makeMask(paramDict['snrThresh'],img = correlated[i].snr,
                                  doKMeans=doKMeans, inverseThresh=paramDict['inverseSNR'])
        except:
          print "DC: Using workaround for tissue param dictionary. Fix me."
          daMask = util2.makeMask(paramDict['snrThresh'], img=correlated[i].snr,
                                  doKMeans=doKMeans)
                                  
        #  print debugging info                                    
        if display:
            plt.figure()
            plt.subplot(2,1,1)
            plt.imshow(correlated[i].snr)
            plt.colorbar()
            plt.subplot(2,1,2)
            plt.imshow(daMask,cmap="gray")
            #plt.axis("off")
            plt.gcf().savefig("stack_%d.png"%iteration)    

        # i don't think this should be rotated 
        #maskList.append((util2.rotater(daMask,iteration)))
        maskList.append(daMask)

    ### Dylan - I'm changing this. I believe that summing the hits adds some noise when thresholds
    ###         are low relative to the hit intensity.
    # sum all hits; Basically looking for one or more angles that are above threshold
    #stacked  = np.sum(maskList, axis =0)

    # instead, take maximum hit
    stacked = np.max(maskList,axis=0)
    
    
    # default behavior (just returned stacked images) 
    if not returnAngles:
      return stacked 
    
    # function that returns angle associated with optimal response
    if returnAngles:
      imgDims = np.shape(stacked)
      maskArray = np.asarray(maskList)
      
      # defaulting array to -1 so it is evident what is and is not a hit
      stackedAngles = np.subtract(np.zeros_like(stacked,dtype='int'), 1)
      #print np.shape(maskArray), imgDims
      angleCounts = []
      for i in range(imgDims[0]):
        for j in range(imgDims[1]):
          if stacked[i,j] != 0:
            # indicates that this is a hit
            iterIdx = int(np.argmax(maskArray[:,i,j]))
            stackedAngles[i,j] = iterIdx
      return stacked, stackedAngles
  
###
### Function to color the angles previously returned in StackHits
###   
def colorAngles(rawOrig, stackedAngles,iters,leftChannel='red',rightChannel='blue'):
  channelDict = {'blue':0, 'green':1, 'red':2}

  dims = np.shape(stackedAngles)

  if len(np.shape(rawOrig)) > 2:
    coloredImg = rawOrig.copy()
    #plt.figure()
    #plt.imshow(coloredImg)
    #plt.show()
    #quit()
  else:
    # we need to make an RGB version of the image
    coloredImg = np.zeros((dims[0],dims[1],3),dtype='uint8')
    scale = 0.75
    coloredImg[:,:,0] = scale * rawOrig
    coloredImg[:,:,1] = scale * rawOrig
    coloredImg[:,:,2] = scale * rawOrig

  leftmostIdx = 0 # mark as left channel
  rightmostIdx = len(iters) # mark as right channel
  # any idx between these two will be colored a blend of the two channels

  spacing = 255 / rightmostIdx

  for i in range(dims[0]):
    for j in range(dims[1]):
      rotArg = stackedAngles[i,j]
      if rotArg != -1:
        coloredImg[i,j,channelDict[leftChannel]] = int(255 - rotArg*spacing)
        coloredImg[i,j,channelDict[rightChannel]] = int(rotArg*spacing)
  return coloredImg

def paintME(myImg, myFilter1,  threshold = 190, cropper=[24,129,24,129],iters = [0,30,60,90], fused =True):
  correlateThresher(myImg, myFilter1,  threshold, cropper,iters, fused, False)
  for i, val in enumerate(iters):
 
    if fused:
      palette = cm.gray
      palette.set_bad('m', 1.0)
      placer = ReadImg('fusedCorrelated_{}.png'.format(val))
    else:
      palette = cm.gray
      palette.set_bad('b', 1.0)
      placer = ReadImg('bulkCorrelated_{}.png'.format(val))
    plt.figure()

    #print "num maxes", np.shape(np.argwhere(placer>threshold))
    Zm = np.ma.masked_where(placer > threshold, placer)
    fig, ax1 = plt.subplots()
    plt.axis("off")
    im = ax1.pcolormesh(Zm, cmap=palette)
    plt.title('Correlated_Angle_{}'.format(val))
    plt.savefig('falseColor_{}.png'.format(val))
    plt.axis('equal')
    plt.close()
    
                

# Basically just finds a 'unit cell' sized area around each detection 
# for the purpose of interpolating the data 
from scipy import signal
def doLabel(result,dx=10,dy=None,thresh=0):
    if dy == None:
      dy = dx
    img =result.stackedHits > thresh
    kernel = np.ones((dy,dx),np.float32)/(float(dy*dx))
    
    filtered = signal.convolve2d(img, kernel, mode='same') / np.sum(kernel)

    plt.subplot(1,3,1)
    plt.imshow(img)
    plt.subplot(1,3,2)
    plt.imshow(filtered)
    plt.subplot(1,3,3)
    labeled = filtered > 0
    plt.imshow(labeled)
    plt.tight_layout()
    #plt.show()
    
    return labeled

def WT_SNR(Img, WTfilter, WTPunishmentFilter,C,gamma):
  # calculates SNR of WT filter
  
  # get two responses
  h = mF.matchedFilter(Img, WTfilter, demean=False)
  h_star = mF.matchedFilter(Img,WTPunishmentFilter,demean=False)
  
  # calculate SNR
  SNR = h / (C + gamma * h_star)

  return SNR

import matchedFilter as mf
def correlateThresherTT (Img, filterDict, iters=[-10,0,10],
             printer=False, label=None, scale=1.0, sigma_n=1.,
             #WTthreshold=None, LossThreshold=None, LongitudinalThreshold=None,
             thresholdDict=None,
             doCLAHE=True, covarianceM=None, gamma=1.):
  # similar to correlateThresher but I want to do all mFing with all filters at once

  # find max response of each of the filters
  maxResponses = {}
  for name,filt in filterDict.iteritems():
    maxResponses[name] = np.sum(filt)

  # setting parameters for SNR of WT
  if covarianceM == None:
    covarianceM = np.ones_like(Img)

  correlated = []

  if doCLAHE:
    clahe99 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16,16))
    cl1 = clahe99.apply(Img)
    adapt99 = cl1
  else:
    adapt99 = Img
  #print iters
  for i, val in enumerate(iters):
    result = empty()
    # copy of original image
    tN = util.renorm(np.array(adapt99,dtype=float),scale=1.)

    print "DC: lines below should be stored in a separate function, since I think here is where the filtering treatments could diverge. Creating a dummy entry here"
    #inputs = empty() # will revise later
    #results = mf.FilterSingle(inputs,mode="do nothing")
    # DC: consider generalizing the labels here. WT --> label1, etc

    ## WT filter and WT punishment filter
    # pad/rotate
    rotWT = PadRotate(filterDict['WT'].copy(), val)
    rotWTPunish = PadRotate(filterDict['WTPunishFilter'].copy(),val)
    # Calculate SNR of WT Response
    rotWT_SNR = WT_SNR(tN, rotWT, rotWTPunish, covarianceM, gamma)
    rotWT_SNR /= maxResponses['WT']

    ## Longitudinal Filter and Response - still basic here
    # pad/rotate
    rotLong = PadRotate(filterDict['Longitudinal'].copy(), val)
    # matched filtering
    rotLongmF = mF.matchedFilter(tN, rotLong, demean=False)
    rotLongmF /= maxResponses['Longitudinal']

    ## Loss Filter and Response - still basic
    # pad/rotate
    rotLoss = PadRotate(filterDict['Loss'].copy(),val)
    # matched filtering
    rotLossmF = mF.matchedFilter(tN, rotLoss, demean=False)
    #print maxResponses['Loss']
    rotLossmF /= maxResponses['Loss']

    # storing results for each function
    result.WT = rotWT_SNR
    result.Long = rotLongmF
    result.Loss = rotLossmF

    correlated.append(result)

  return correlated
    
    








