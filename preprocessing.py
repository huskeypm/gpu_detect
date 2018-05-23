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

###############################################################################
###
###  Reorientation Routines
###
###############################################################################

def reorient(img):
  return img

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

def setup(path):
    px = pygame.image.load(path)
    screen = pygame.display.set_mode( px.get_rect()[2:] )
    screen.blit(px, px.get_rect())
    pygame.display.flip()
    return screen, px

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

def giveSubsection(inputFile):
    input_loc = inputFile
    output_loc = 'out.png'
    screen, px = setup(input_loc)
    left, upper, right, lower = mainLoop(screen, px)

    # ensure output rect always has positive width, height
    if right < left:
        left, right = right, left
    if lower < upper:
        lower, upper = upper, lower
    im = Image.open(input_loc)
    subsection = im.crop(( left, upper, right, lower))
    pygame.display.quit()
    return subsection

def resizeToFilterSize(img,filterTwoSarcomereSize,fileName):
  '''
  Function to semi-automate the resizing of the image based on the filter
  '''

  ### 1. Select subsection of image that exhibits highly conserved network of TTs
  subsection = giveSubsection(fileName)#,dtype=np.float32)
  subsection = np.array(subsection.convert("F"))

  # best to normalize the subsection for display purposes
  subsection /= np.max(subsection)

  ### 2. Using this subsection, calculate the periodogram
  fBig, psd_Big = signal.periodogram(subsection)
  # finding sum, will be easier to identify striation length with singular dimensionality
  bigSum = np.sum(psd_Big,axis=0)

  ### 3. Find peak value of periodogram and calculate striation size
  striationSize = 1. / fBig[np.argmax(bigSum)]
  imgTwoSarcomereSize = int(round(2 * striationSize))
  print "Two Sarcomere size:", imgTwoSarcomereSize,"Pixels per Two Sarcomeres"
  if imgTwoSarcomereSize > 50:
    print "WARNING: Image likely failed to be properly resized. Manual resizing\
           may be necessary!!!!!"

  ### 4. Using peak value, resize the image
  scale = float(filterTwoSarcomereSize) / float(imgTwoSarcomereSize)
  resized = cv2.resize(img,None,fx=scale,fy=scale,interpolation=cv2.INTER_CUBIC)

  return resized

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

  img = reorient(img)
  img = resizeToFilterSize(img,filterTwoSarcomereSize,fileName)
  img = applyCLAHE(img,filterTwoSarcomereSize)

  # write file
  name,fileType = fileName[:-4],fileName[-4:]
  newName = name+"_processed_new"+fileType
  cv2.imwrite(newName,img)

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

