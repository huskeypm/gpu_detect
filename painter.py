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

def padWithZeros(array, padwidth, iaxis, kwargs):
    array[:padwidth[0]] = 0
    array[-padwidth[1]:]= 0
    return array



class empty:pass


def PadRotate(myFilter1,val):
  dims = np.shape(myFilter1)
  diff = np.min(dims)
  paddedFilter = np.lib.pad(myFilter1,diff,padWithZeros)
  rotatedFilter = imutils.rotate(paddedFilter,-val)
  rF = np.copy(rotatedFilter)

  return rF

# TODO phase this out 
def CalcInvFilter(filterRef,tN,val,yP,penaltyscale,sigma_n): 
      s=1.  
      fInv = np.max(filterRef)- s*filterRef
      rFi = PadRotate(fInv,val)
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

      # store data 
      #if useFilterInv:
      #  result.corr = scaled    
      #else:
      #  result.corr = yP 


# Need to be careful when cropping image
import detection_protocols as dps

def correlateThresher(
        inputs,
        params,
        iters = [0,30,60,90],  
        printer = True, 
        filterMode=None,
        label=None,
        ):
    # Store all 'hits' at each angle 
    correlated = []

    print "REPALCEME PKH"
    myImg = inputs.imgOrig
    myFilter1 = inputs.mfOrig

    # Ryan ?? equalized image?
    # Dylan - Adding in option to turn off CLAHE
    # TODO - this should be done in preprocessing, not here
    if params['doCLAHE']:
      clahe99 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16,16))
      cl1 = clahe99.apply(myImg)
      adapt99 = cl1
    else:
      adapt99 = myImg

    print "VERIFY WE STILL NEED" 
    filterRef = util.renorm(np.array(myFilter1,dtype=float),scale=1.)

    
    # TODO - here is the place to stick in the GPU shizzle 
    print "HERE"
    for i, val in enumerate(iters):
      print i, val  
      result = empty()
      # copy of original image 
      inputs.img = util.renorm(np.array(adapt99,dtype=float),scale=1.)

      
      ## 'positive' filter 
      # pad/rotate 
      rFN = PadRotate(filterRef,val)  
      inputs.mf = rFN  
   
      # matched filtering 
      result = dps.FilterSingle(inputs,params)      


      ## negative filter 
      if params['useFilterInv']:
        print "NEEDS TO BE MERGED INTO detection protocol"  
        result.corr = CalcInvFilter(filterRef,inputs.img,val,result.corr,
                                    params['penaltyscale'],
                                    params['sigma_n'])

      tag = filterMode 
      #if filterMode=="fused":
      #  tag = tag"fused"
      ###else: 
      #  tag = "bulk"

      #print "MOVE this shit elsewhere" 
      #daTitle = "rot %f "%val + "hit %f "%hit + str(hitLoc)
      daTitle = "rot %4.1f "%val # + "hit %4.1f "%hit 
      #print daTitle
      if printer:   
        plt.figure(figsize=(16,5))
        plt.subplot(1,5,1)
        plt.imshow(adapt99,cmap='gray')          
        plt.subplot(1,5,2)
        plt.title(daTitle)
        testImg = np.zeros_like(adapt99)
        dim = np.shape(rFN)
        testImg[0:dim[0],0:dim[1]] = 255*rFN
        #testImg[0:dim[0],0:dim[1]] = rFiN
        plt.imshow(testImg,cmap="gray")
        #plt.imshow(rFi,cmap="gray")   
        
        plt.subplot(1,5,3)
        #plt.imshow(yPN)   
        #plt.title("corr output") 
        #plt.colorbar()

        #if threshold!=None:
        #  plt.subplot(1,5,4)
        #  #plt.imshow(yP>threshold)   
        #  plt.title("Filter inv")
        #  plt.imshow(yInvN)                  
        #  plt.colorbar()
        #  plt.subplot(1,5,5)
        #  plt.title("filter/inv") 
        #  plt.imshow(scaled)                
        #  plt.colorbar()
        plt.tight_layout()
        fileName = label+"_"+tag+'_{}_full.png'.format(val)
        plt.gcf().savefig(fileName,dpi=300)

      # write
      if label!=None and printer:
        plt.figure()
        plt.subplot(1,2,1)
        plt.title("Rotated filter") 
        plt.imshow(rFN,cmap='gray')
        plt.subplot(1,2,2)
        plt.title("Correlation plane") 
        plt.imshow(result.corr)                
        plt.colorbar()
        plt.tight_layout()
        fileName = label+"_"+tag+'_{}.png'.format(val)
        plt.gcf().savefig(fileName,dpi=300)
        plt.close()
     
      print "NEEDS TO BE MOVED INSIDE PROTOCOL"
      #  
      result.snr = CalcSNR(result.corr,params['sigma_n']) 
      # 
      result.max = np.max(result.corr)

      #result.hit = hit
      #result.hitLoc = hitLoc
      correlated.append(result) 
    print np.shape(correlated)
    print np.shape(correlated[0].corr)
    return correlated  # a list of objects that contain SNR, etc 

def CalcSNR(signalResponse,sigma_n=1):
  return signalResponse/sigma_n

import util 
import util2
def StackHits(correlated,  # an array of 'correlation planes'
              paramDict, # threshold,
              iters,
              display=False,
              rescaleCorr=False,
              doKMeans=False, #True,
              filterType="Pore",
              returnAngles=False):
    # Function that iterates through correlations at varying rotations of a single filter,
    # constructs a mask consisting of 'NaNs' and returns a list of these masked correlations

    print "DC: make into ditionary?" 
    maskList = []
    
    print "REMOVE ME" 
    WTlist = []
    Longlist = []
    Losslist = []

#    daMax = -1e9
    for i, iteration in enumerate(iters):
        #print i, iteration
        if filterType == "Pore":
          corr_i = correlated[i].corr           
 #         daMax = np.max([daMax, np.max(corr_i)]) 
          
          if rescaleCorr:
             img =  util.renorm(corr_i)
             print "Why is this needed? IGNORING"

          # routine for identifying 'unique' hits
          daMask = util2.makeMask(paramDict['snrThresh'],img = corr_i,doKMeans=doKMeans)
          if display:
            plt.figure()
            plt.subplot(1,2,1)
            plt.imshow(img)            
            plt.subplot(1,2,2)
            plt.imshow(daMask)
            plt.close()

          # i don't think this should be rotated 
          #maskList.append((util2.rotater(daMask,iteration)))
          maskList.append(daMask)
      #print maskList

        elif filterType == "TT":
          print "DCL Merge me with oth er filter type" 
          masks = empty()
          corr_i = correlated[i].corr 
          #corr_i_WT = correlated[i].WT
          #corr_i_Long = correlated[i].Long
          #corr_i_Loss = correlated[i].Loss

          mask    = util2.makeMask(threshold, img=corr_i, doKMeans=doKMeans)
          #masks.WT = util2.makeMask(threshold['WT'], img=corr_i_WT, doKMeans=doKMeans)
          #masks.Long = util2.makeMask(threshold['Longitudinal'], img=corr_i_Long, doKMeans=doKMeans)
          #masks.Loss = util2.makeMask(threshold['Loss'], img=corr_i_Loss, doKMeans=doKMeans,inverseThresh=True)
          if display:
            #WT
            plt.figure()
            #plt.subplot(1,1)
            #plt.imshow(img)
            plt.subplot(1,3,1)
            plt.imshow(mask) #s.WT)
            #plt.title('WT')
            # Longituindal
            #plt.subplot(1,3,2)
            #plt.imshow(masks.Long)
            #plt.title('Longitudinal')
            # Loss
            #plt.subplot(1,3,3)
            #plt.imshow(masks.Loss)
            #plt.title('Loss')
            #plt.close()

          # changing values to strings that will later be interpreted by colorHits function
          #colorIndicator = 'rot'+str(i)
          #masks.WT[masks.WT != 0] = colorIndicator
          #masks.Long[masks.Long != 0] = colorIndicator
          # not going to mark Loss in the same way to make code a bit more efficient later

          maskList.append(mask)
          #WTlist.append(masks.WT)
          #Longlist.append(masks.Long)
          #Losslist.append(masks.Loss)

    #
    # Return
    # DC: Need to consolidate 
    #
    print "PKH: Need to conslidate" 
    if filterType == "Pore":
      stacked  = np.sum(maskList, axis =0)
      return stacked 

    elif filterType == "TT":
      #stacked = empty()
      # creating a class that also contains the angle with which the most intense hit was located

      #dims = np.shape(WTlist[0]) # all rotations and filter correlations should be the same dimensions
     
      # creating 'poor mans mask' through use of NaN
      myHolder = np.argmax(maskList,axis=0).astype('float') 
      myHolder[maskList[0] < 1e-5] = np.nan
      return myHolder 
      
      #WTholder = np.argmax(WTlist,axis=0).astype('float')
      #WTholder[WTlist[0] < 0.00001] = np.nan
      #stacked.WT = WTholder
      #Longholder = np.argmax(Longlist,axis=0).astype('float')
      #Longholder[Longlist[0] < 0.00001] = np.nan
      #stacked.Long = Longholder
      #stacked.Loss = np.sum(Losslist,axis=0)

      #if returnAngles: Turning off for code dev
      if 0: 
        stackedAngles = empty()
        WTAngle = WTholder.flatten().astype('float')
        for i,idx in enumerate(np.invert(np.isnan(WTholder.flatten()))):
          if idx:
            WTAngle[i] = iters[int(WTAngle[i])]
        WTAngle = WTAngle.reshape(dims[0],dims[1])
        stackedAngles.WT = WTAngle

        LongAngle = Longholder.flatten().astype('float')
        for i,idx in enumerate(np.invert(np.isnan(Longholder.flatten()))):
          if idx:
            LongAngle[i] = iters[int(LongAngle[i])]
        LongAngle = LongAngle.reshape(dims[0],dims[1])
        stackedAngles.Long = LongAngle
      else:
        stackedAngles = None
      return stacked, stackedAngles
   
    if display and filterType == "Pore": 
      plt.figure()
      plt.imshow(myList)
      plt.close()




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
def doLabel(result,dx=10):
    img =result.stackedHits > 0
    kernel = np.ones((dx,dx),np.float32)/(dx*dx)
    
    filtered = signal.convolve2d(img, kernel, mode='same') / np.sum(kernel)

    plt.subplot(1,3,1)
    plt.imshow(img)
    plt.subplot(1,3,2)
    plt.imshow(filtered)
    plt.subplot(1,3,3)
    labeled = filtered > 0
    plt.imshow(labeled)
    plt.tight_layout()
    
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
    
    








