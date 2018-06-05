###
### Group of functions that will walk the user fully through the preprocessing
### routines.
###

import util, cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import pygame, sys
from PIL import Image
pygame.init() # initializes pygame modules
from sklearn.decomposition import PCA
import imutils
import matchedFilter as mF

###############################################################################
###
### Normalization Routines
###
##############################################################################

def normalizeToStriations(img, subsection):
  '''
  function that will go through the subsection and find average smoothed peak 
  and valley intensity of each striation and will normalize the image 
  based on those values.
  '''
  subsectionDims = np.shape(subsection)
 
  ### smooth striations
  smoothingFilter = np.ones((1,4),dtype=np.float64) / 4. 
  smoothed = mF.matchedFilter(subsection,smoothingFilter,demean=False)

  ### calculate slopes using centered difference formula
  slopeFilter = np.zeros((1, 3),dtype=np.float64)
  slopeFilter[0,0] = -1.
  slopeFilter[0,2] = 1.
  slopeFilter /= 2.
  slopes = mF.matchedFilter(smoothed,slopeFilter,demean=False)
  binarySlopes = slopes > 0

  ### find where slope = 0 (either a peak or valley according to 1st deriv rule)
  #zeroSlopes = np.argwhere(np.abs(slopes) < 0.0000005)

  #peakValues = []
  #valleyValues = []
  #for idx in zeroSlopes:
  #  if idx[1] == 0 or idx[1] == subsectionDims[1]-1:
  #    continue
  #  else:
  #    if slopes[idx[0],idx[1]-1] > 0 and slopes[idx[0],idx[1]+1] < 0:
  #      peakValues.append(smoothed[idx[0],idx[1]]) 
  #    elif slopes[idx[0],idx[1]-1] < 0 and slopes[idx[0],idx[1]+1] > 0:
  #      valleyValues.append(smoothed[idx[0],idx[1]])
  for row in range(subsectionDims[0]):
    thisRow = binarySlopes[row,:]
    #print thisRow
    rolled = np.roll(thisRow, -1)
    #print rolled
    #return thisRow, rolled
    peakIdxs = np.argwhere(thisRow & np.invert(rolled))
    valleyIdxs = np.argwhere(np.invert(thisRow) & rolled)
    print peakIdxs,valleyIdxs
    peaks = []
    for idx in peakIdxs:
      peaks.append(smoothed[row,idx])
    print np.mean(peaks)
    valleys = []
    for idx in valleyIdxs:
      valleys.append(smoothed[row,idx])
    print np.mean(valleys)
    quit()
  meanPeaks = np.mean(peakValues)
  meanValleys = np.mean(valleyValues)

  return meanPeaks,meanValleys
###############################################################################
###
### FFT Filtering Routines
###
###############################################################################




###############################################################################
###
###  Reorientation Routines
###
###############################################################################

def setup(array):
    #px = pygame.image.load(path)
    px = pygame.surfarray.make_surface(array)
    screen = pygame.display.set_mode( px.get_rect()[2:] )
    screen.blit(px, px.get_rect())
    pygame.display.flip()
    return screen, px

def displayImageLine(screen, px, topleft, prior):
    # ensure that the rect always has positive width, height
    x, y = topleft
    xNew = pygame.mouse.get_pos()[0]
    yNew = pygame.mouse.get_pos()[1]
    width = xNew - x
    height = yNew - y
    if width < 0:
        x += width
        width = abs(width)
    if height < 0:
        y += height
        height = abs(height)

    # eliminate redundant drawing cycles (when mouse isn't moving)
    current = x, y, width, height
    if not (width and height):
        return current
    if current == prior:
        return current
    
    # draw line on the image
    red = (255, 0, 0)
    startPoint = topleft
    endPoint = (xNew,yNew)
    screen.blit(px,px.get_rect())
    pygame.draw.line(screen,red,startPoint,endPoint)
    pygame.display.flip()

    # return current box extents
    return (x, y, width, height)

def mainLoopLine(screen, px):
    topleft = bottomright = prior = None
    n=0
    while n!=1:
        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONUP:
                if not topleft:
                    topleft = event.pos
                else:
                    bottomright = event.pos
                    n=1
        if topleft:
            prior = displayImageLine(screen, px, topleft, prior)
    return ( topleft + bottomright )

def giveSubsectionLine(array):
    # pygame has weird indexing
    newArray = np.swapaxes(array,0,1)
    screen, px = setup(newArray)
    pygame.display.set_caption("Draw a Line Orthogonal to Transverse Tubules")
    left, upper, right, lower = mainLoopLine(screen, px)
    pygame.display.quit()
    
    directionVector = (right-left,upper-lower)
    return directionVector

def reorient(img):

  '''
  Function to reorient the myocyte based on a user selected line that is
  orthogonal to the TTs 
  '''

  ### get direction vector from line drawn by user
  dVect = giveSubsectionLine(img)

  ### we want rotation < 90 deg so we ensure correct axis
  if dVect[0] >= 0:
    xAx = [1,0]
  else:
    xAx[-1,0]

  ### compute degrees off center from the direction vector
  dOffCenter = (180./np.pi) * np.arccos(np.dot(xAx,dVect)/np.linalg.norm(dVect))

  ### ensure directionality is correct 
  if dVect[1] <= 0:
    dOffCenter *= -1
  print "Image is",dOffCenter,"degrees off center"

  ### rotate image
  rotated = imutils.rotate_bound(img,dOffCenter)

  return rotated, dOffCenter

###############################################################################
###
### Resizing Routines
###
###############################################################################
def displayImage(screen, px, topleft, prior):
    # ensure that the rect always has positive width, height
    x, y = topleft
    width =  pygame.mouse.get_pos()[0] - topleft[0]
    height = pygame.mouse.get_pos()[1] - topleft[1]
    if width < 0:
        x += width
        width = abs(width)
    if height < 0:
        y += height
        height = abs(height)

    # eliminate redundant drawing cycles (when mouse isn't moving)
    current = x, y, width, height
    if not (width and height):
        return current
    if current == prior:
        return current

    # draw transparent box and blit it onto canvas
    screen.blit(px, px.get_rect())
    im = pygame.Surface((width, height))
    im.fill((128, 128, 128))
    pygame.draw.rect(im, (32, 32, 32), im.get_rect(), 1)
    im.set_alpha(128)
    screen.blit(im, (x, y))
    pygame.display.flip()

    # return current box extents
    return (x, y, width, height)

def mainLoop(screen, px):
    topleft = bottomright = prior = None
    n=0
    while n!=1:
        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONUP:
                if not topleft:
                    topleft = event.pos
                else:
                    bottomright = event.pos
                    n=1
        if topleft:
            prior = displayImage(screen, px, topleft, prior)
    return ( topleft + bottomright )

def giveSubsection(array):
    # pygame has weird indexing
    newArray = np.swapaxes(array,0,1)
    screen, px = setup(newArray)
    pygame.display.set_caption("Draw a Rectangle Around Several Conserved Transverse Tubule Striations")
    left, upper, right, lower = mainLoop(screen, px)

    # ensure output rect always has positive width, height
    if right < left:
        left, right = right, left
    if lower < upper:
        lower, upper = upper, lower
    subsection = array.copy()[upper:lower,left:right]
    subsection = np.asarray(subsection, dtype=np.float64)
    pygame.display.quit()
    return subsection

def resizeToFilterSize(img,filterTwoSarcomereSize):
  '''
  Function to semi-automate the resizing of the image based on the filter
  '''

  ### 1. Select subsection of image that exhibits highly conserved network of TTs
  subsection = giveSubsection(img)#,dtype=np.float32)

  ### 1.5 Attempting simple thresholding based on subsection as well
  #print np.max(img)
  #img[img > 0.3 * np.max(subsection)] = 0.3 * np.max(subsection)
  #img[img > 1.5 * np.mean(subsection)] = np.mean(subsection)
  #img[img < 0.05 * np.max(subsection)] = 0.05 * np.max(subsection)
  #print np.max(img)
  #print np.min(img)
  #img -= np.min(img)
  #img = np.float64(img)
  #img /= np.max(img)
  #img *= 255
  #img = np.uint8(img)

  #import matplotlib.pyplot as plt
  #plt.figure()
  #plt.imshow(img,cmap='gray')
  #plt.show()

  # best to normalize the subsection for display purposes
  subsection /= np.max(subsection)

  ### 2. Using this subsection, calculate the periodogram
  fBig, psd_Big = signal.periodogram(subsection)
  # finding sum, will be easier to identify striation length with singular dimensionality
  bigSum = np.sum(psd_Big,axis=0)

  ### 3. Mask out the noise in the subsection periodogram
  # NOTE: These are imposed assumptions on the resizing routine
  maxStriationSize = 50.
  minStriationSize = 5.
  minPeriodogramValue = 1. / maxStriationSize
  maxPeriodogramValue = 1. / minStriationSize
  bigSum[fBig < minPeriodogramValue] = 0.
  bigSum[fBig > maxPeriodogramValue] = 0.

  display = False
  if display:
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(fBig,bigSum)
    plt.title("Collapsed Periodogram of Subsection")
    plt.show()

  ### 3. Find peak value of periodogram and calculate striation size
  striationSize = 1. / fBig[np.argmax(bigSum)]
  imgTwoSarcomereSize = int(round(2 * striationSize))
  print "Two Sarcomere size:", imgTwoSarcomereSize,"Pixels per Two Sarcomeres"
  if imgTwoSarcomereSize > 70 or imgTwoSarcomereSize < 10:
    print "WARNING: Image likely failed to be properly resized. Manual resizing",\
           "may be necessary!!!!!"

  ### 4. Using peak value, resize the image
  scale = float(filterTwoSarcomereSize) / float(imgTwoSarcomereSize)
  resized = cv2.resize(img,None,fx=scale,fy=scale,interpolation=cv2.INTER_CUBIC)

  return resized,scale,subsection

###############################################################################
###
### CLAHE Routines
###
###############################################################################

def applyCLAHE(img,filterTwoSarcomereSize):
  kernel = (filterTwoSarcomereSize, filterTwoSarcomereSize)
  clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=kernel)

  clahedImage = clahe.apply(img)

  return clahedImage
###############################################################################
###
### Main Routine
###
###############################################################################

def preprocess(fileName,filterTwoSarcomereSize):
  img = util.ReadImg(fileName)

  img,degreesOffCenter = reorient(img)
  img,resizeScale,subsection = resizeToFilterSize(img,filterTwoSarcomereSize)
  #img = normalizeToStriations(img,subsection)
  img = applyCLAHE(img,filterTwoSarcomereSize)

  # fix mask based on img orientation and resize scale
  try:
    processMask(fileName,degreesOffCenter,resizeScale)
  except:
    1

  # write file
  name,fileType = fileName[:-4],fileName[-4:]
  newName = name+"_processed"+fileType
  cv2.imwrite(newName,img)

  return img

def processMask(fileName,degreesOffCenter,resizeScale):
  '''
  function to reorient and resize the mask that was generated for the original
  image.
  '''
  maskName = fileName[:-4]+"_mask"+fileName[-4:]
  mask = util.ReadImg(maskName)
  reoriented = imutils.rotate_bound(mask,degreesOffCenter)
  resized = cv2.resize(reoriented,None,fx=resizeScale,fy=resizeScale,interpolation=cv2.INTER_CUBIC)
  cv2.imwrite(fileName[:-4]+"_processed_mask"+fileName[-4:],resized)

def preprocessAll():
  '''
  function meant to preprocess all of the images needed for data reproduction
  '''
  root = './myoimages/'
  imgNames = ["HF_1.png", "MI_D_73.png","MI_M_45.png","MI_P_16.png","Sham_11.png",
              "Sham_M_65.png"]
  for name in imgNames:
    filterTwoSarcomereSize = 25
    # perform preprocessing on image
    preprocess(root+name,filterTwoSarcomereSize)

###############################################################################
###
### Execution of File
###
###############################################################################

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
tag = "default_"
if __name__ == "__main__":
  import sys
  msg = helpmsg()
  remap = "none"

  if len(sys.argv) < 2:
      raise RuntimeError(msg)

  # Loops over each argument in the command line 
  for i,arg in enumerate(sys.argv):

    ### Main routine to run
    if (arg=="-preprocess"):
      fileName = str(sys.argv[i+1])
      try:
        filterTwoSarcomereSize = sys.argv[i+2]
      except:
        filterTwoSarcomereSize = 25
      preprocess(fileName,filterTwoSarcomereSize)
      quit()
    if (arg=="-preprocessAll"):
      preprocessAll()
      quit()
  raise RuntimeError("Arguments not understood")

