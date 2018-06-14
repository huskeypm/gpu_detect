"""
Wrapper for 'simplist' of MF calls 
Ultimately will be posted on athena
"""

import matplotlib.pylab as plt 
import numpy as np
import display_util as du
import matchedFilter as mf 
import optimizer
import bankDetect as bD
import util
import painter
import sys
import yaml
import cv2

def DisplayHits(img,threshed,
                smooth=8 # px
                ):
        # smooth out image to make it easier to visualize hits 
        daround=np.ones([smooth,smooth])
        sadf=mf.matchedFilter(threshed,daround,parsevals=False,demean=False)

        # merge two fields 
        du.StackGrayRedAlpha(img,sadf)


class empty:pass    
def docalc(img,
           mf,
           lobemf=None,
           #corrThresh=0.,
           #s=1.,
           paramDict = optimizer.ParamDict(),
           debug=False,
           smooth = 8, # smoothing for final display
           iters = [-20,-10,0,10,20], # needs to be put into param dict
           fileName="corr.png"):



    ## Store info 
    inputs=empty()
    inputs.imgOrig = img
    inputs.mfOrig  = mf
    inputs.lobemf = lobemf

    print "WARNING: TOO RESTRICTIVE ANGLES" 



    results = bD.DetectFilter(inputs,paramDict,iters=iters,display=debug)
    result = results.correlated[0]
    #corr = np.asarray(results.correlated[0],dtype=float) # or
    results.threshed = results.stackedHits

    pasteFilter = True
    if pasteFilter:

      MFy,MFx = util.measureFilterDimensions(mf)
      filterChannel = 0
      imgDim = np.shape(img)
      #coloredImageHolder = np.zeros((imgDim[0],imgDim[1],3),dtype=npfloat64)
      #filterChannelHolder = coloredImageHolder[:,:,filterChannel)
      # TODO: Come back and fix thresh
      results.threshed = painter.doLabel(results,dx=MFx,dy=MFy,thresh=254)
      #coloredImageHolder[:,:,filterChannel] = filterChannelHolder
    
    print "Writing file %s"%fileName
    #plt.figure()
    DisplayHits(img,results.threshed)
    plt.gcf().savefig(fileName,dpi=300)


    return inputs,results 



#!/usr/bin/env python
##################################
#
# Revisions
#       10.08.10 inception
#
##################################

#
# Simple performs a matched filtering detection with a single filter, image and threshold
#
def simple(imgName,mfName,thresh,debug=False,smooth=4,outName="hits.png"): 
  img = util.ReadImg(imgName,renorm=True)
  mf  = util.ReadImg( mfName,renorm=True)

  paramDict = optimizer.ParamDict()
  paramDict['snrThresh'] = thresh
  paramDict['penaltyscale'] = 1.
  paramDict['useFilterInv'] = False
  paramDict['doCLAHE'] = False
  paramDict['demeanMF'] = False
  docalc(img,
         mf,
         paramDict = paramDict, 
         debug=debug,
         smooth=smooth, 
         fileName=outName)

#
# Calls 'simple' with yaml file 
# 
def simpleYaml(ymlName):
  with open(ymlName) as fp:
    data=yaml.load(fp)
  #print data    
  print "Rreading %s" % ymlName    
    
  if 'outName' in data:
      outName=data['outName']
  else:
      outName="hits.png"
  #print outName 
  
  simple(
    data['imgName'],
    data['mfName'],
    data['thresh'],
    debug= data['debug'],
    outName=outName)


###
###  3D routine. Will eventually stick into threeDtense most likely
###

def do3DGPUFiltering():
  '''
  In prototype/testing phase right now
  '''

  import threeDtense as tdt

  # setup your structures
  inputs = empty()
  paramDict = optimizer.ParamDict(typeDict="WT")

  # make a test image of 512x512x50 where the height is hardcoded...
  img = tdt.MakeTestImage(dim=200)
  inputs.imgOrig = img

  # making example 3D WT filters
  mf = util.ReadImg('./myoimages/newSimpleWTFilter.png',renorm=True)
  mf /= np.sum(mf)
  mfPunishment = util.ReadImg('./myoimages/newSimpleWTPunishmentFilter.png',renorm=True)
  mfPunishment /= np.sum(mfPunishment)
  mfDims = np.shape(mf)
  mfPunishmentDims = np.shape(mfPunishment)
  depth = 5
  mf3D = np.zeros((mfDims[0],mfDims[1],depth),dtype=np.float64)
  mfPunishment3D = np.zeros((mfPunishmentDims[0],mfPunishmentDims[1],depth),dtype=np.float64)
  for i in range(depth):
    mf3D[:,:,i] = mf
    mfPunishment3D[:,:,i] = mfPunishment
  inputs.mfOrig = mf3D
  # fix this dumb piece of code and store it inputs
  paramDict['mfPunishment'] = mfPunishment3D
  paramDict['covarianceMatrix'] = np.ones_like(img)

  # call 3D gpu filtering code
  results,tElapsed = tdt.doTFloop(inputs,paramDict)

  #print "HOORAY!!!!!!"
  holder = np.zeros_like(results.stackedHits,dtype=np.uint8)
  holder[results.stackedHits] = 255
  print np.shape(holder)
  print type(holder)
  print holder

  name = "3D_WT_StackedHits.tif"
  cv2.imwrite(name,holder)
  print "Wrote", name





  

#
# Validation 
#
def validation(): 
  imgName ="myoimages/Sham_11.png"
  mfName ="myoimages/WTFilter.png"
  thresh = -0.08 
  thresh = 0.01
  print "WHY IS THRESH NEGATIVE?"              
  simple(imgName,mfName,thresh,smooth=20,
         debug=True)

  print "WARNING: should add an assert here of some sort"

#
# Message printed when program run without arguments 
#
def helpmsg():
  scriptName= sys.argv[0]
  msg="""
Purpose: 
 
Usage:
"""
  msg+="  %s -validation" % (scriptName)
  msg+="""
  
 
Notes:

"""
  return msg

#
# MAIN routine executed when launching this script from command line 
#
if __name__ == "__main__":
  msg = helpmsg()
  remap = "none"

  if len(sys.argv) < 2:
      raise RuntimeError(msg)

  #fileIn= sys.argv[1]
  #if(len(sys.argv)==3):
  #  1
  #  #print "arg"

  # Loops over each argument in the command line 
  for i,arg in enumerate(sys.argv):
    # validation
    if(arg=="-validation"):             
      validation() 
      quit()

    # general test
    if(arg=="-simple"):             
      imgName =sys.argv[i+1] 
      mfName =sys.argv[i+2] 
      thresh = float(sys.argv[i+3])
      simple(imgName,mfName,thresh)
      quit()
  
    # general test with yaml
    if(arg=="-simpleYaml"):             
      ymlName=sys.argv[i+1] 
      simpleYaml(ymlName)
      quit()

    if(arg=="-giveCorrelation"):
      imgName = sys.argv[i+1]
      mfName = sys.argv[i+2]
      thresh = float(sys.argv[i+3])
      debug = True
      simple(imgName,mfName,thresh,debug=debug)
      quit()

    if(arg=="-do3D"):
      do3DGPUFiltering()
      quit()





  raise RuntimeError("Arguments not understood")




