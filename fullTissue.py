'''
Script to house figure generation routines for the analysis of the entire tissue-level image
'''

import os
import sys
import matchedFilter as mF
import myocyteFigs as mFigs
import numpy as np
import matplotlib.pyplot as plt
import cv2
import util
import tissue as tis
import cPickle as pkl

class empty:pass

def analyzeTissue(iteration):
  '''
  This function will take the previously preprocessed tissue and run the WT 
  filter across the entire image. From this, we will apply a large smoothing
  function across the entire detection image. This should hopefully provide
  a nice gradient that shows TT-density loss as proximity to the infarct
  increases

  NOTE: This function is meant to be called repeatedly by a bash script.
          Some library utilized herein is leaky and eats memory for 
          breakfast.

  Inputs:
    iteration - rotation at which to perform the detection

  Outputs:
    None

  Written Files:
    "tissueDetections_<iteration>.pkl"
  '''
  ### Load in tissue
  params = tis.params
  grayTissue = tis.Setup().astype(np.float32)
  grayTissue /= np.max(grayTissue)
  print "Size of single tissue image in Gigabytes:",sys.getsizeof(grayTissue) / 1e9

  ttFilterName = "./myoimages/newSimpleWTFilter.png"
  ttPunishFilterName = "./myoimages/newSimpleWTPunishmentFilter.png"

  inputs = empty()
  inputs.imgOrig = grayTissue
  inputs.useGPU = False

  returnAngles = False

  import time
  startTime = time.time()
  ### This is mombo memory intensive
  thisIteration = [iteration]
  tissueResults = mFigs.WT_Filtering(inputs,thisIteration,ttFilterName,ttPunishFilterName,None,None,False)
  #print "before masking"
  resultsImage = tissueResults.stackedHits > 0 
  #print "after masking"


  ### save tissue detection results
  #print "before dumping"
  name = "tissueDetections_"+str(iteration)
  ## consider replacing with numpy save function since it is much quicker
  #pkl.dump(resultsImage,open(name,'w'))
  np.save(name,resultsImage)
  #print "after dumping"
  endTime = time.time()

  print "Time for algorithm to run:",endTime-startTime

def displayTissueResults():
  '''
  This function's purpose is to display the tissue-level results in a manner 
  that is easy to interpret. This is achieved by running a large smoothing
  filter over all of the detections and overlaying this on the tissue-level
  image.
  '''

  ### Read in tissue image
  params = tis.params
  grayTissue = tis.Setup().astype(np.float32)
  tisMin = 0
  tisMax = np.max(grayTissue)
  ## kill the brightness for future display purposes
  grayTissue *= 0.75


  ### find all detection pkl files we've made in the previous routine
  rotations = []
  for fileName in os.listdir('./'):
    if "tissueDetections_" in fileName:
      _,newName = fileName.split('_')
      rotationNumber,fileType = newName.split('.')
      rotations.append(int(rotationNumber))

  ### sort rotations
  rotations.sort()

  ### impose rotation cutoff
  cutoff = -5 
  rotations = rotations[:np.argmax([rot == cutoff for rot in rotations])+1]
  
  ### create initial storage list based on first detection
  detections = np.load("tissueDetections_"+str(rotations[0])+'.'+fileType)
  storage = detections.copy()

  #detections = np.load("tissueDetections_"+str(rotations[-5])+'.'+fileType)
  #storage = detections.copy()
  
  #plt.figure()
  #plt.imshow(storage)
  #plt.show()
  #quit()

  ### get rid of the first entry so we don't double count it
  rotations.pop(0)

  ### loop through and account for all detections
  for rotation in rotations:
    print rotation
    ### Read in detection results
    detections = np.load("tissueDetections_"+str(rotation)+'.'+fileType,'r')
    storage = np.where(detections, 1,storage)

  ### convert to float32 so we can use convolution function to smooth the results
  storage = storage.astype(np.float32)

  displaySample = False 
  if displaySample:
    plt.figure()
    plt.imshow(grayTissue,cmap='gray',vmin=tisMin,vmax=tisMax)
    plt.imshow(storage,cmap='Blues',alpha=0.75)
    plt.axis('off')
    plt.show()
    return

  ### smooth the resulting image
  kernelSize = (2000,2000)
  storage = cv2.blur(storage,kernelSize)

  clipImage = True
  if clipImage:
    clipLimit = 0.2
    storage[storage<clipLimit*np.max(storage)] = clipLimit*np.max(storage)
  #plt.figure()
  #plt.imshow(storage)
  #plt.show()
  #quit()
  writeImage = False
  downsample = False
  if downsample:
    plt.rcParams['figure.figsize'] = [5,5]
    plt.figure()
    plt.imshow(grayTissue,cmap='gray',vmin=tisMin,vmax=tisMax)
    plt.imshow(storage,cmap='RdYlGn',alpha=0.75)
    plt.axis('off')
    if writeImage:
      plt.gcf().savefig('FullTissueAnalysis.png',dpi=300)
    else:
      plt.show()
  else:
    plt.rcParams['figure.figsize'] = [16,16]
    plt.figure()
    plt.imshow(grayTissue,cmap='gray',vmin=tisMin,vmax=tisMax)
    plt.imshow(storage,cmap='RdYlGn',alpha=0.75)
    plt.axis('off')
    if writeImage:
      plt.gcf().savefig('FullTissueAnalysis.png',dpi=1000)
    else:
      plt.show()




#
# MAIN routine executed when launching this script from command line
#
if __name__ == "__main__":

  if len(sys.argv) < 2:
    raise RuntimeError()

  for i,arg in enumerate(sys.argv):
    if(arg=="-fullTissue"):
      rotation = int(sys.argv[i+1])
      analyzeTissue(rotation)
      quit()

    if(arg=="-makeDisplay"):
      displayTissueResults()
      quit()

  raise RuntimeError("Arguments not understood")
