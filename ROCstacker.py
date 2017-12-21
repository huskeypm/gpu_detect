# runner function that will be used for ROC generation/optimization

import util
import cv2
import numpy as np
import matchedFilter as mF
import os
import bankDetect as bD
import painter


def ReadResizeNormImg(imgName, scale):
    img = cv2.imread(imgName)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    normed = resized.astype('float') / float(np.max(resized))
    
    return normed


# # Pre-Processing
# - Rotate myocyte images in gimp s.t. TTs are orthogonal to the x axis
# - Measure the two sarcomere size of the TTs
# - use util.preprocessPNG to resize to filter two sarcomere size, renorm, CLAHE
# - TENTATIVE: crop out extracellular region/sarcolemma and create mask

# # Pass in Arguments

# example:
#root = "/home/AD/dfco222/Desktop/LouchData/"
#imgName = 'Sham_23'
#fileType = '.png'
#WTthreshold=0.045
#Longitudinalthreshold = 0.38
#gamma = 3.

# distance between the middle of one z-line to the second neighbor of that z-line
filterTwoSarcSize = 25

# designating images used to construct the filters as we won't be testing those
filterDataImages = []

# parameters for all the different analysis options
pad = True
plotRawImages = False
binarizeImgs = False
fixWTFilter = True
fixLongFilter = True
plotFilters = False
plotRawFilterResponse = False # MAKE SURE THIS IS OFF FOR LARGE DATA SIZES

# main function
def giveStackedHits(imgName, WTthreshold, Longitudinalthreshold, gamma,
                     WTFilterName="./images/WTFilter.png",
                     LongitudinalFilterName="./images/LongFilter.png",
                     LossFilterName="./images/LossFilter.png",
                     WTPunishFilterName="./images/WTPunishmentFilter.png",
                     returnMasks=False):
  # Read Images and Apply Masks

  # using old code structure so this dictionary is just arbitrary
  imgTwoSarcSizesDict = {imgName:'Nonsense'}
  for imgName,imgTwoSarcSize in imgTwoSarcSizesDict.iteritems():
      if imgName not in filterDataImages:
          scale = 1. # we're resizing prior to this call
          # additionally tis image is already renormed using util so 
	  # this is just reading in the image
          img = ReadResizeNormImg(imgName, scale)
          combined = img
          # routine to pad the image with a border of zeros.
          if pad:
              combined = util.PadWithZeros(combined)
          imgDim = np.shape(combined)
          if plotRawImages:
              plt.figure()
              imshow(combined)
              plt.title(imgName)
              plt.colorbar()
              plt.close()

  # Read in Filters
  maxResponseDict = {}
  WTfilter = util.ReadImg(WTFilterName).astype('float')/float(255)
  Longitudinalfilter = util.ReadImg(LongitudinalFilterName).astype('float')/float(255)
  Lossfilter = util.ReadImg(LossFilterName).astype('float')/float(255)
  WTPunishFilter = util.ReadImg(WTPunishFilterName)/255

  filterDict = {'WT':WTfilter, 'Longitudinal':Longitudinalfilter, 'Loss':Lossfilter, 'WTPunishFilter':WTPunishFilter}

  # finding maximum response of each filter by integrating intensity
  for filterName, myFilter in filterDict.iteritems():
      maxResponseDict[filterName] = np.sum(myFilter)

  if plotFilters:
        for name,Filter in filterDict.iteritems():
              plt.figure()
              imshow(Filter)
              plt.title(name)
              plt.colorbar()


  # # Convolve Each Image with Each Filter

  rotDegrees = [-20, -15, -10, -5, 0, 5, 10, 15, 20]
  display = False
  thresholdDict = {'WT':WTthreshold, 'Longitudinal':Longitudinalthreshold, 'Loss':0.08}
  Result = bD.TestFilters(img,None,None,filterType="TT",
                              display=display,iters=rotDegrees,filterDict=filterDict,
			      thresholdDict=thresholdDict,doCLAHE=False,
                              colorHitsOutName=imgName,
                              label=imgName,
                              saveColoredFig=False,
                              gamma=gamma)
  #print Result.coloredImg
  #cv2.imwrite("test.png", Result.coloredImg)
  #quit()
  if not returnMasks:
    return Result
  else:
    return Result.coloredImg[:,:,0],Result.coloredImg[:,:,1],Result.coloredImg[:,:,2]

# Message printed when program run without arguments 
def helpmsg():
  scriptName= sys.argv[0]
  msg="""
Purpose: A script to aid in the repetitive correlation used for ROC optimization.
 
Usage: Call script from command line with args being: 1. image name and path 2. wild type threshold 3. longitudinal threshold 4. gamma parameter in SNR
"""
  msg+="""
  
 
Notes:

"""
  return msg


if __name__=="__main__":
  import sys
  msg=helpmsg()
  remap = "none"

  if len(sys.argv) < 5:
      raise RuntimeError(msg)


  imgName = str(sys.argv[1])
  if sys.argv[2] != "-masks":
    WTthresh = float(sys.argv[2])
    Longitudinalthresh = float(sys.argv[3])
    gamma = float(sys.argv[4])
    result = gimmeStackedHits(imgName, WTthresh, Longitudinalthresh, gamma)
    import matplotlib.pylab as plt
    corr = imgName.split('/')[-1]
    name,filetype = corr.split('.')
    myName = name+'_'+str(WTthresh)+'_'+str(Longitudinalthresh)+'_'+str(gamma)+'.'+filetype
    plt.imshow(result)
    plt.gcf().savefig(myName) 

  else:
    WTthresh = float(sys.argv[3])
    Longitudinalthresh = float(sys.argv[4])
    gamma = float(sys.argv[5])
    # this returns the masks individually instead of stacked together
    origMask,LongMask,WTMask = gimmeStackedHits(imgName, WTthresh, Longitudinalthresh,gamma,returnMasks=True)
    corr = imgName.split('/')[-1]
    name,filetype = corr.split('.')
    myName = name+'_'+str(WTthresh)+'_'+str(Longitudinalthresh)+'_'+str(gamma)
    import matplotlib.pylab as plt
    plt.imshow(WTMask)
    plt.gcf().savefig(myName+'_WTMask'+'.'+filetype)
    plt.figure()
    plt.imshow(LongMask)
    plt.gcf().savefig(myName+'_LongMask'+'.'+filetype)
