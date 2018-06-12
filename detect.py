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





  raise RuntimeError("Arguments not understood")




