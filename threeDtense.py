import sys
import os
import cv2
import matplotlib
#matplotlib.use('Agg')
import numpy as np
import matplotlib.pylab as plt
import tensorflow as tf
import time
import scipy.fftpack as fftp
import util
import imutils
import optimizer
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#from PIL import Image
#from skimage.external.tifffile import TiffWriter
#from skimage.io._plugins.tifffile_plugin import imsave
#import SimpleITK as sitk
import tifffile as tiff
height = 50
#ters = [-30,-20,-10,0,10,20]
iters = [0]

class empty:
  pass


#Ripped straight from stackoverflow
from skimage import io

def Arrayer(fileName = '/home/AD/srbl226/GPU/gpu_detect/140722_2_2.tif',picDims = [0,512,0,512,0,111]):
  im = io.imread(fileName)

  i = im.transpose() ### the image is loaded as z,y,x transposing to become x,y,z

  I = i[picDims[0]:picDims[1],picDims[2]:picDims[3],picDims[4]:picDims[5]]
  print "max of array: ", np.max(I)
  I/= np.max(I)
  #print "total pixels",np.shape(I)[0]*np.shape(I)[1]*np.shape(I)[2]
  print "shape of image array",  I.shape
  return I

from scipy.misc import toimage
def PoreFilter(size=40,dim=3):
  img = np.zeros((size,size,dim), np.uint8) ###build's an image array

  cv2.circle(img,(size/2,size/2),size/20,(255,255,255),-1) ### constructs a circle of radius size/20 in the center of the array


  ### This is probably bad practices
  toimage(img).save('pore.png')  ### stores the image to be recast as a numpy array
  pore = util.ReadImg('pore.png').astype(np.float64)
  #fPore = pore.flatten()  ### flattens to find indicies of non-pore points

  #locs = np.argwhere(fPore<255.0)  ### locating nonpore points
  #fPore[locs] = -100.0  ###setting nonpore points as arbitrarily punished locations
  arbitrary = 10  ### arbitrary depth of filter
  #pore = np.reshape(fPore,(size,size))  ###  rebuilding 2D image
  vec = np.ones((arbitrary)) ### generating vector to project 2D image into 3D
  cross = np.outer(pore,vec)  ### doing the cross product
  crossFilter = np.reshape(cross,(size,size,arbitrary))  ### building the cross product into desired dimensions in 3-space, not sure why this step was necessary
  #crossFilter[:,:,:arbitrary/4] = -100.0  ### adding punishment "cap" to filter
  #crossFilter[:,:,(arbitrary*3):] = -100.0
  print "dimensions of pore filter", np.shape(crossFilter)
  crossFilter/=np.max(crossFilter)  ### renorming it to be max of 1
  return crossFilter

def PunishFilter(size = 40,dim=3):
    img = np.zeros((size,size,dim), np.uint8)
    img[:,:] =(255,255,255)
    cv2.circle(img,(size/2,size/2),size/20,(0,0,0),-1)
    toimage(img).save('punishmentFilter.png')  ### stores the image to be recast as a numpy array
    pore = util.ReadImg('punishmentFilter.png').astype(np.float64)
    arbitrary = 10  ### arbitrary depth of filter
    #pore = np.reshape(fPore,(size,size))  ###  rebuilding 2D image
    vec = np.ones((arbitrary)) ### generating vector to project 2D image into 3D
    cross = np.outer(pore,vec)  ### doing the cross product
    crossFilter = np.reshape(cross,(size,size,arbitrary))  ### building the cross product into desired dimensions in 3-space, not sure why this step was necessary
    #crossFilter[:,:,:arbitrary/4] = -100.0  ### adding punishment "cap" to filter
    #crossFilter[:,:,(arbitrary*3):] = -100.0
    print "dimensions of punishment filter", np.shape(crossFilter)
    crossFilter/=np.max(crossFilter)  ### renorming it to be max of 1
    return crossFilter




def SphereFilter(size=32,dim=3):
  img = np.zeros((size,size,size), np.float64)
  #cv2.circle(img,(size/2,size/2),size/2,(1,1,1),-1)  #used for punishment
  for s in range(size):
	for r in range(size):
		for q in range(size):
			if ((s-size/2.)**2.+(r-size/2.)**2.+(q-size/2.)**2.)**0.5<size/2.:
				img[s,r,q]=1.

  print "newImg max", np.max(img)
  #toimage(img).save('sphere.png')
  #sphere = util.ReadImg('sphere.png').astype(np.float64)
  #cube[:,:] = 1
  #arbitrary =  size
  sphere = np.copy(img)
  sphere/= np.sum(img)
  print "sphere max", np.max(sphere)

  #vec = np.ones((arbitrary)) ############This is arbitrary, please make sensical
  #cross = np.outer(cube,vec)
  #crossFilter = np.reshape(cross,(size,size,arbitrary))
  #print "dimensions of filter", np.shape(crossFilter)
  #crossFilter/=np.max(crossFilter)
  #return crossFilter
  return sphere



#import imutils
def LoadImage(
  imgName = "/home/AD/pmke226/DataLocker/cardiac/Sachse/171127_tissue/tissue.tif",
  mid = 10000,
  maxDim = 100,
  angle = -35.
):
    print "WARNING: please use util.ReadImg"
    # read image
    img = np.array(util.ReadImg(imgName),dtype=np.float64)

    # extract subset
    imgTrunc = img[
      (mid-maxDim):(mid+maxDim),
      (mid-maxDim):(mid+maxDim),(mid-maxDim):(mid+maxDim)]

    # rotate to align TTs while we are testing code
    #imgR = imutils.rotate(imgTrunc, angle)
    #cv2.imshow(imgR,cmap="gray")

    return imgR




# In[132]:

def MakeTestImage(dim = 400):
    l = np.zeros(2); l[0:0]=1.
    z = l
    # there are smarter ways of doing this
    dim = int(dim)
    l = np.zeros(dim)
    for i in range(4):
      l[i::10]=1
    #print l

    striped = np.outer(np.ones(dim),l)
    #cv2.imshow(striped)
    img2 = striped
    height3 = np.ones((height))
    cross = np.outer(img2, height3)

    imgR = np.reshape(cross,(dim,dim,height))
    imgR[:,(dim/2):,:(height/2)] = 0
    #print "Made test Image"
    return imgR

# ### Make filter
# - Measured about 35 degree rotation (clockwise) to align TT with y axis
# - z-lines are about 14 px apart
# - z-line about 3-4 pix thick
# - 14-20 px tall is probably reasonable assumption

# In[128]:

def MakeFilter(
    fw = 4,
    fdim = 14,
    height=10
  ):
    dFilter = np.zeros([fdim,fdim,height])
    dFilter[:,0:fw,0:fw]=1.
    # test
    #dFilter[:] = 1 # basically just blur image
    yFilter = np.roll(dFilter,-np.int(fw/2.),axis=1)
    #cv2.imshow(dFilter,cmap="gray")

    return dFilter


# In[128]:




# In[136]:

def Pad(imgR,
        dFilter):

    fdim = np.shape(dFilter)
    imgDim = np.shape(imgR)
    if len(imgDim) == 3:
      assert (fdim[2] <= imgDim[2]), "Your filter is larger in the z direction than your image"
    filterPadded = np.zeros_like( imgR)
    if len(imgDim) == 3:
      filterPadded[0:fdim[0],0:fdim[1],0:fdim[2]] = dFilter
      hfwz = np.int(fdim[2]/2.)
    else:
      filterPadded[0:fdim[0],0:fdim[1]] = dFilter
    hfw = np.int(fdim[0]/2.)
    hfwx = np.int(fdim[1]/2.)

    # we do this rolling here in the same way we shift for FFTs
    yroll = np.roll(filterPadded,-hfw,axis=0)
    xyroll = np.roll(yroll,-hfwx,axis=1)
    if len(imgDim) == 3:
      xyzroll = np.roll(xyroll,-hfwz, axis=2)
    else:
      xyzroll = xyroll

    # I don't think you need to shift this at all
    #xyzroll = filterPadded

    return xyzroll

def tfMF(img,mf,dimensions=3):
  '''
  Generic workhorse routine that sets up the image and matched filter for
    matched filtering using tensorflow. This routine is called by whichever
    detection scheme has been chosen. NOT by the doTFloop explicitly.

  INPUTS:
    img - 'tensorized' image
    mf - rotated and 'tensorized' matched filter same dimension as img
  '''

  if dimensions == 3:
    xF = tf.fft3d(img)
    mFF = tf.fft3d(mf)
    # this is something to do with the shifting done previously
    xFc = tf.conj(xF)
    out = tf.multiply(xF,mFF)
    xR = tf.ifft3d(out)
  else:
    xF = tf.fft2d(img)
    mFF = tf.fft2d(mf)
    # this is something to do with the shifting done previously
    xFc = tf.conj(xF)
    out = tf.multiply(xF,mFF)
    xR = tf.ifft2d(out)

  return xR


##
## Tensor flow part
##
def doTFloop(inputs,
           #img,# test image
           #mFs, # shifted filter
           paramDict,
           xiters=[0],
           yiters=[0],
           ziters=[0]
           ):
  '''
  Function that performs filtering across all rotations of a given filter
    using tensorflow routines and GPU speedup
  '''

  # TODO: Once TF 1.9 is out and stable, change complex data types to 128 instead of 64
  #       I believe there is some small error arising from this


  # We may potentially want to incorporate all 3 filters into this low level
  # loop to really speed things up

  with tf.Session() as sess:
    start = time.time()

    ### Create and initialize variables
    #print " pre step1", time.time()-start
    tfImg = tf.Variable(inputs.imgOrig, dtype=tf.complex64)
    #print " post step1/ prePad", time.time()-start
    #paddedFilter = Pad(inputs.imgOrig,inputs.mfOrig)
    #print " postPad", time.time()-start
    tfFilt = tf.constant(inputs.mfOrig)
    #tfFilt = tf.Variable(paddedFilter, dtype=tf.complex64)
    paddings = tf.constant([[0,inputs.imgOrig.shape[0]-inputs.mfOrig.shape[0]],[0,inputs.imgOrig.shape[1]-inputs.mfOrig.shape[1]],[0,inputs.imgOrig.shape[2]-inputs.mfOrig.shape[2]]])
    padFilt = tf.pad(tfFilt,paddings,"CONSTANT")
    rollFilt = tf.manip.roll(padFilt,shift=[-inputs.mfOrig.shape[0]/2,-inputs.mfOrig.shape[1]/2,-inputs.mfOrig.shape[2]/2],axis=[0,1,2])
    #print "post roll", time.time()-start
    pF = tf.cast(rollFilt, dtype=tf.complex64)
    tfFilt = tf.Variable(pF, dtype=tf.complex64)

    #print "post Pad---->tf/pre stacked hits", time.time()-start
    if paramDict['inverseSNR']:
      stackedHitsDummy = np.ones_like(inputs.imgOrig)
    else:
      stackedHitsDummy = np.zeros_like(inputs.imgOrig)
    stackedHits = tf.Variable(stackedHitsDummy, dtype=tf.float64)
    #print "post stacked hits", time.time()-start
    bestAngles = tf.Variable(np.zeros_like(inputs.imgOrig),dtype=tf.float64)
    snr = tf.Variable(np.zeros_like(tfImg),dtype=tf.complex64)
    #print "post snr", time.time()-start

    # make big angle container
    numXits = np.shape(xiters)[0]
    numYits = np.shape(yiters)[0]
    numZits = np.shape(ziters)[0]
    cnt = tf.Variable(tf.constant(numXits * numYits * numZits-1,dtype=tf.int64))
    #print "post cnt", time.time()-start

    # It's late and I'm getting lazy
    bigIters = []
    #print "pre bigIters", time.time()-start

    if len(np.shape(inputs.imgOrig)) == 3:
      for i in xiters:
        for j in yiters:
          for k in ziters:
            bigIters.append([i,j,k])
    else:
      bigIters = ziters
    #print "post bigIters", time.time()-start
    # convert iterations from degrees into radians
    bigIters = np.asarray(bigIters,dtype=np.float32)
    bigIters = bigIters * np.pi / 180.


    # have to convert to a tensor so that the rotations can be indexed during tf while loop
    bigIters = tf.Variable(tf.convert_to_tensor(bigIters,dtype=tf.float32))
    paramDict['inverseSNR'] = tf.Variable(paramDict['inverseSNR'], dtype=tf.bool)

    # set up filtering variables
    if paramDict['filterMode'] == 'punishmentFilter':
      paramDict['mfPunishment'] = Pad(inputs.imgOrig,paramDict['mfPunishment'])
      paramDict['mfPunishment'] = tf.Variable(paramDict['mfPunishment'],dtype=tf.complex64)
      paramDict['covarianceMatrix'] = tf.Variable(paramDict['covarianceMatrix'],dtype=tf.complex64)
      paramDict['gamma'] = tf.Variable(paramDict['gamma'],dtype=tf.complex64)
      sess.run(tf.variables_initializer([paramDict['mfPunishment'],paramDict['covarianceMatrix'],paramDict['gamma']]))
    #print "post big dict, pre run", time.time()-start
    sess.run(tf.variables_initializer([tfImg,tfFilt,cnt,bigIters,stackedHits,bestAngles,snr,paramDict['inverseSNR']]))

    inputs.tfImg = tfImg
    inputs.tfFilt = tfFilt
    inputs.bigIters = bigIters
    #print "post run", time.time()-start

    # While loop that counts down to zero and computes reverse and forward fft's
    def condition(cnt,stackedHits,bestAngles):
      return cnt >= 0

    def body3D(cnt,stackedHits,bestAngles):
      # pick out rotation to use
      rotations = bigIters[cnt]

      # rotating matched filter to specific angle
      rotatedMF = util.rotateFilterCube3D(inputs.tfFilt,
                                          rotations[0],
                                          rotations[1],
                                          rotations[2])

      # get detection/snr results
      snr = doDetection(inputs,paramDict,dimensions=3)
      stackedHitsNew,bestAnglesNew = doStackingHits(inputs,paramDict,stackedHits,bestAngles,snr,cnt)
      #stackedHitsNew = stackedHits

      cntnew=cnt-1
      return cntnew,stackedHitsNew,bestAnglesNew

    def body2D(cnt,stackedHits,bestAngles):
      rotation = bigIters[cnt]

      rotatedMF = util.rotateFilter2D(inputs.tfFilt,rotation)

      snr = doDetection(inputs,paramDict,dimensions=2)
      #stackedHitsNew,bestAnglesNew = doStackingHits(inputs,paramDict,stackedHits,bestAngles,snr,cnt)
      stackedHitsNew = tf.cast(snr,dtype=tf.float64)
      bestAnglesNew = stackedHitsNew
      cntnew=cnt-1
      return cntnew,stackedHitsNew,bestAnglesNew

    # can we optimize parallel_iterations based on memory allocation?
    '''
    NOTE: when images get huge (~>512x512x50) and we use parallel iterations=10, they eat a lot of memory
          since the intermediate results from the while loop are stored for back propogation.
          This memory consumption can be offloaded if we use swap_memory=True in the tf.while_loop.
          However, this slows down the calculation immensely, so we want to always cleverly pick
          the sweet spot between NOT offloading the memory eating tensors and NOT bricking the computer.
    '''
    # TODO: See if there is a way to grab maximum available memory for the GPU to automatically
    #       and cleverly determin the sweetspot to where we don't have to offload tensors to cpu
    #       but can also efficiently determine parallel_iterations number
    if len(np.shape(inputs.imgOrig)) == 3:
      cnt,stackedHits,bestAngles = tf.while_loop(condition, body3D,
                                      [cnt,stackedHits,bestAngles], parallel_iterations=1)
    else:
      cnt,stackedHits,bestAngles = tf.while_loop(condition, body2D,
                                      [cnt,stackedHits,bestAngles], parallel_iterations=10)

    compStart = time.time()
    cntF,stackedHitsF,bestAnglesF =  sess.run([cnt,stackedHits,bestAngles])
    compFin = time.time()
    print "Time for tensor flow to execute run:{}s".format(compFin-compStart)
    #print "stackedHits type  ", type(stackedHits), "  stackedHits shape  ", np.shape(stackedHits)
    sHits = stackedHits.eval()
    #print "type sHits ", type(sHits)

    ###############THIS IS A STUPID HACK
    coolLoc = np.argmax(sHits,axis=0)
    coolLoc =np.argmax(coolLoc,axis=0)
    coolLoc =np.argmax(coolLoc,axis=0)

    forTiff = np.asarray(sHits*(255/np.max(sHits)),dtype=np.uint8)

    sHits/=np.max(sHits)

    tElapsed = time.time()-start
    print 'tensorflow:{}s'.format(tElapsed)
    print 'Total time for tensorflow to run:{}s'.format(tElapsed)
    return forTiff, tElapsed #results, tElapsed


def MF(
    dImage,
    dFilter,
    useGPU=False,
    dim=2,
    xiters=[0],
    yiters=[0],
    ziters=[0]
    ):
    # NOTE: May need to do this padding within tensorflow loop itself to
    #       ameliorate latency due to loading large matrices into GPU.
    #       Potentially a use for tf.dynamic_stitch?
    lx = len(xiters); ly = len(yiters); lz = len(ziters)
    numRots = lx * ly * lz
    filt = dFilter#Pad(dImage,dFilter)

    inputs = empty()
    inputs.imgOrig = dImage
    inputs.mfOrig = filt
    if useGPU:
       paramDict = optimizer.ParamDict(typeDict='WT')
       paramDict['filterMode'] = 'simple'
       paramDict['snrThresh'] = 0.0
       corr,tElapsed = doTFloop(inputs,paramDict,xiters=xiters,yiters=yiters,ziters=ziters)
       corr = np.real(corr)
       #print "POST FILTER"
    else:
      pass
      start = time.time()
      filtTF = tf.convert_to_tensor(filt)
      for i,x in enumerate(xiters):
        ii = tf.constant(i,dtype=tf.float64)
        for j,y in enumerate(yiters):
          jj = tf.constant(j,dtype=tf.float64)
          for k,z in enumerate(ziters):
            print "Filtering Progress:", str((i*ly*lz)+(j*lz)+k)+'/'+str(numRots)
            # rotate filter using tensorflow routine anyway
            # NOTE: I may change this
            kk = tf.constant(k,dtype=tf.float64)
            filtRot = util.rotateFilterCube3D(filtTF,ii,jj,kk)
            filtRot = tf.Session().run(filtRot)
            I = dImage
            T = filtRot
            fI = fftp.fftn(I)
            fT = fftp.fftn(T)
            c = np.conj(fI)*fT
            corr = fftp.ifftn(c)
            corr = np.real(corr)
      tElapsed = time.time()-start
      print 'fftp:{}s'.format(tElapsed)

    return corr,tElapsed

###################################################################################################
###
### Detection Schemes: taken from detection_protocols.py and 'tensorized'
###                    See detection_protocols.py for further documentation of functions
###################################################################################################

def doDetection(inputs,paramDict,dimensions=3):
  '''
  Function to route all tf detection schemes through.

  inputs - a class structure containing all necessary inputs
  paramDict - a dictionary containing your parameters

  These should follow the basic structure for the non TF calculations, but just
    'tensorized'
  '''

  mode = paramDict['filterMode']
  if mode=="lobemode":
    print "Tough luck, this isn't supported yet. Bug the developers to implement this. Quitting program"
    quit()
  elif mode=="punishmentFilter":
    results = punishmentFilterTensor(inputs,paramDict,dimensions=dimensions)
  elif mode=="simple":
    results = simpleDetectTensor(inputs,paramDict,dimensions=dimensions)
  elif mode=="regionalDeviation":
    results =regionalDeviationTensor(inputs,paramDict,dimensions=dimensions)
  elif mode=="filterRatio":
    print "Ignoring this for now. Bug the developers to implement if you want this detection scheme."
    quit()
  else:
    print "That mode isn't understood. Returning image instead."
    results = inputs.img

  return results

def punishmentFilterTensor(inputs,paramDict,dimensions=3):
  # call generalized tensorflow matched filter routine
  corr = tfMF(inputs.tfImg,inputs.tfFilt,dimensions=dimensions)
  corrPunishment = tfMF(inputs.tfImg,paramDict['mfPunishment'],dimensions=dimensions)

  # calculate signal to noise ratio
  snr = tf.divide(corr,tf.add(paramDict['covarianceMatrix'],tf.multiply(paramDict['gamma'], corrPunishment)))
  return snr

def simpleDetectTensor(inputs,paramDict,dimensions=3):
  snr = tfMF(inputs.tfImg,inputs.tfFilt,dimensions=dimensions)
  return snr


def regionalDeviationTensor(inputs,paramDict,dimensions=3):
  ### Perform simple detection
  img = inputs.tfImg
  mf = inputs.tfFilt
  corr = tfMF(img, mf,dimensions=dimensions)

  ### construct new kernel for standard deviation calculation
  # find where filter > 0
  kernelLocs = tf.greater(tf.cast(mf,tf.float64),0.)
  kernel = tf.where(kernelLocs,tf.ones_like(mf),tf.zeros_like(mf))
  kernel = tf.cast(kernel,tf.complex64)

  ### construct array that contains number of elements in each window
  n = tfMF(tf.ones_like(img),kernel,dimensions=dimensions)

  ### square image for standard deviation calculation
  imgSquared = tf.square(img)

  ### Calculate standard deviation
  s = tfMF(img,kernel,dimensions=dimensions)
  q = tfMF(imgSquared,kernel,dimensions=dimensions)
  stdDev = tf.sqrt( tf.divide( tf.subtract(q, tf.divide(tf.square(s),n)), tf.subtract(n,1)))

  ### since this is dually conditional, I'll have to mask out the super threshold std Dev hits
  stdDevHits = tf.less(tf.cast(stdDev,tf.float64),paramDict['stdDevThresh'])
  maskedOutHits = tf.where(stdDevHits,corr,tf.zeros_like(corr))

  return maskedOutHits


def doStackingHits(inputs,paramDict,stackedHits,bestAngles,snr,cnt):
  '''
  Function to threshold the calculated snr and apply to the stackedHits container
  '''
  snr = tf.cast(tf.real(snr),dtype=tf.float64)

  # check inverse snr toggle, if true, find snr < stackedHits, if false, find snr > stackedHits
  snrHits = tf.cond(paramDict['inverseSNR'],
                    lambda: tf.less(snr,stackedHits),
                    lambda: tf.greater(snr,stackedHits))
  stackedHits = tf.where(snrHits,snr,stackedHits)
  cntHolder = tf.multiply(tf.ones_like(stackedHits), tf.cast(cnt,tf.float64))
  bestAngles = tf.where(snrHits,cntHolder,bestAngles)
  return stackedHits,bestAngles

###################################################################################################
###
###  End detection schemes
###
###################################################################################################

def writer(testImage,name="out.tif"):
    # rescale
    img = testImage
    imgR = img - np.min(img)
    imgN = imgR/np.max(imgR)*(2**8  - 1 ) # for 8bit channel


    # convert to unsigned image
    out = np.array(imgN,dtype=np.uint8)
    cv2.imwrite(name,out)






#!/usr/bin/env python
import sys
##################################
#
# Revisions
#       10.08.10 inception
#
##################################

#
# ROUTINE
#
def test0(useGPU=True):
  testImage = MakeTestImage()
  dFilter = MakeFilter()
  corr = MF(testImage,dFilter,useGPU=useGPU,dim=3,xiters=iters,yiters=iters,ziters=iters)
  return testImage,corr

def runner(dims):
  times =[]
  f = open("GPU_Benchmark.txt","w")
  f.write('Dims:{}'.format(dims))
  for i,d in enumerate(dims):
    continue
    print "dim", d
    testImage = MakeTestImage(d)
    dFilter = MakeFilter()
    corr,time = MF(testImage,dFilter,useGPU=False,dim=3,xiters=iters,yiters=iters,ziters=iters)
    times.append(time)

  timesGPU =[]
  f.write('\nGPU:')
  for j,d in enumerate(dims):
    print "dim", d
    testImage = MakeTestImage(d)
    dFilter = MakeFilter()
    corr,time = MF(testImage,dFilter,useGPU=True,dim=3,xiters=iters,yiters=iters,ziters=iters)
    timesGPU.append(time)
  f.write('{}'.format(timesGPU))
  return times, timesGPU

def test1(maxDim=100):
  testImage = LoadImage(maxDim=maxDim)
  dFilter = MakeFilter()
  corr = MF(testImage,dFilter,useGPU=True,dim=3,xiters=iters,yiters=iters,ziters=iters)
  writer(corr)
  return testImage,corr, dFilter

def testReal(dims=[1]):
  times =[]
  f = open("GPU_Benchmark.txt","w")
  f.write('Dims:{}'.format(dims))
  f.write('\nCPU:')
  for i,d in enumerate(dims):
    continue
    print "dim", d
    testImage = Arrayer()
    dFilter = PoreFilter()
    corr,time = MF(testImage,dFilter,useGPU=False,dim=3,xiters=iters,yiters=iters,ziters=iters)
    times.append(time)
  f.write('{}'.format(times))

  timesGPU =[]
  f.write('\nGPU:')
  for j,d in enumerate(dims):
    print "dim", d
    testImage = Arrayer()
    dFilter = PoreFilter()

    print " we testin"
    corr,time = MF(testImage,dFilter,useGPU=True,dim=3,xiters=iters,yiters=iters,ziters=iters)
    timesGPU.append(time)
  f.write('{}'.format(timesGPU))
  return times, timesGPU

def pictureMaker(dims=[1]): ### This fxn makes a convolved pic to compare to OG image
  dFilter = PoreFilter()
  for i  in [0,1]:
      for j in [0,1]:
          for k in [0,1]:
            testImage = Arrayer('/home/AD/srbl226/gpu3D/gpu_detect/floored.tiff',[i*512,(i+1)*512,j*512,(j+1)*512,111*k,(k+1)*111])
            im,time = MF(testImage,dFilter,useGPU=True,dim=3,xiters=iters,yiters=iters,ziters=iters)
            #im = convolved ### grab convolved im here
            if not k:
                imZ = im
            else:
                imZ = np.concatenate((imZ,im),axis=2)
          if not j:
            imY = imZ
          else:
            imY = np.concatenate((imY,imZ),axis=1)
      if not i:
          imX = imY
      else:
          imX = np.concatenate((imX,imY),axis=0)
  print "final Dims", np.shape(imX)
  finalImage = imX.transpose().astype(np.float64)
  finalImage*=(255./np.max(finalImage))
  tiff.imsave('convolved.tiff', finalImage.astype(np.uint8))

def kBrane(dims=[1]): ### This fxn makes a convolved pic to compare to OG image
  dFilter = SphereFilter()
  for i  in [0,1]:
      for j in [0,1]:
          for k in [0,1]:
            testImage = Arrayer('/home/AD/srbl226/Fiji/140722_2_2.tif',[i*512,(i+1)*512,j*512,(j+1)*512,111*k,(k+1)*111])
            im,time = MF(testImage,dFilter,useGPU=True,dim=3,xiters=iters,yiters=iters,ziters=iters)
            if not k:
                imZ = im
            else:
                imZ = np.concatenate((imZ,im),axis=2)
          if not j:
            imY = imZ
          else:
            imY = np.concatenate((imY,imZ),axis=1)
      if not i:
          imX = imY
      else:
          imX = np.concatenate((imX,imY),axis=0)
  finalImage = imX.transpose()
  flatFinal = finalImage.flatten()
  sixtyPercent = np.percentile(flatFinal,60)
  locs = np.argwhere(flatFinal>sixtyPercent)
  newIm = io.imread('/home/AD/srbl226/Fiji/140722_2_2.tif')
  I = np.copy(newIm) #.transpose()
  newLocs = np.unravel_index(locs, np.shape(finalImage))   # If breaks cry to Pete
  I[newLocs] = 0
  tiff.imsave('kMembraned.tiff',I)
def kBrane2(dims=[1]):
  dFilter = SphereFilter(size = 16)
  for i  in [0,1]:
      for j in [0,1]:
          for k in [0,1]:
            testImage = Arrayer('/home/AD/srbl226/gpu3D/gpu_detect/kMembraned.tiff',[i*512,(i+1)*512,j*512,(j+1)*512,111*k,(k+1)*111])
            im,time = MF(testImage,dFilter,useGPU=True,dim=3,xiters=iters,yiters=iters,ziters=iters)
            if not k:
                imZ = im
            else:
                imZ = np.concatenate((imZ,im),axis=2)
          if not j:
            imY = imZ
          else:
            imY = np.concatenate((imY,imZ),axis=1)
      if not i:
          imX = imY
      else:
          imX = np.concatenate((imX,imY),axis=0)
  finalImage = imX.transpose()
  flatFinal = finalImage.flatten()
  sixtyPercent = np.percentile(flatFinal,60)
  locs = np.argwhere(flatFinal>sixtyPercent)
  newIm = io.imread('/home/AD/srbl226/gpu3D/gpu_detect/kMembraned.tiff')
  I = np.copy(newIm) #.transpose()
  newLocs = np.unravel_index(locs, np.shape(finalImage))   # If breaks cry to Pete
  I[newLocs] = 0
  tiff.imsave('kMembraned.tiff',I)


def kBrane3(dims=[1]):
  dFilter = SphereFilter(size = 8)
  for i  in [0,1]:
      for j in [0,1]:
          for k in [0,1]:
            testImage = Arrayer('/home/AD/srbl226/gpu3D/gpu_detect/kMembraned.tiff',[i*512,(i+1)*512,j*512,(j+1)*512,111*k,(k+1)*111])
            im,time = MF(testImage,dFilter,useGPU=True,dim=3,xiters=iters,yiters=iters,ziters=iters)
            if not k:
                imZ = im
            else:
                imZ = np.concatenate((imZ,im),axis=2)
          if not j:
            imY = imZ
          else:
            imY = np.concatenate((imY,imZ),axis=1)
      if not i:
          imX = imY
      else:
          imX = np.concatenate((imX,imY),axis=0)
  finalImage = imX.transpose()
  flatFinal = finalImage.flatten()
  sixtyPercent = np.percentile(flatFinal,60)
  locs = np.argwhere(flatFinal>sixtyPercent)
  newIm = io.imread('/home/AD/srbl226/gpu3D/gpu_detect/kMembraned.tiff')
  I = np.copy(newIm) #.transpose()
  newLocs = np.unravel_index(locs, np.shape(finalImage))   # If breaks cry to Pete
  I[newLocs] = 0
  tiff.imsave('kMembraned.tiff',I)


def subset():
  dFilter = PoreFilter()
  testImage = io.imread('/home/AD/srbl226/gpu3D/gpu_detect/fDownsampled.tiff')
  print "testImage shape", np.shape(testImage)
  test = testImage.transpose()
  test/=np.max(test)
  im,time = MF(test,dFilter,useGPU=True,dim=3,xiters=iters,yiters=iters,ziters=iters)
  finalImage = im.transpose().astype(np.float64)
  finalImage *= (255./np.max(finalImage))
  print "finalImage shape", np.shape(finalImage)
  tiff.imsave('smallConvolved.tiff',finalImage.astype(np.uint8))


def Punish():
   dFilter = PoreFilter()
   testImage = io.imread('/home/AD/srbl226/gpu3D/gpu_detect/fDownsampled.tiff')
   print "testImage shape", np.shape(testImage)
   test = testImage.transpose()
   test/=np.max(test)
   im,time = MF(test,dFilter,useGPU=True,dim=3,xiters=iters,yiters=iters,ziters=iters)
   convolvedImage = im.transpose().astype(np.float64)
   convolvedImage *= (255./np.max(convolvedImage))
   print "shape of convolved", np.shape(convolvedImage)

   dFilter = PunishFilter()
   #testImage = io.imread('/home/AD/srbl226/gpu3D/gpu_detect/fDownsampled.tiff')
   print "testImage shape", np.shape(testImage)
   #test = testImage.transpose()
   #test/=np.max(test)
   im,time = MF(test,dFilter,useGPU=True,dim=3,xiters=iters,yiters=iters,ziters=iters)
   punishImage = im.transpose().astype(np.float64)
   punishImage *= (255./np.max(punishImage))
   print "shape of punished", np.shape(punishImage)
   SNR = convolvedImage/(1+5*punishImage)
   flatSNR = SNR.flatten()
   locs = np.argwhere(flatSNR>1.0)
   print "90th percentile SNR: ", np.percentile(SNR,90)
   newLocs = np.unravel_index(locs, np.shape(SNR))
   forStacking = (testImage*(255/np.max(testImage)))
   coloredImage = np.zeros_like(SNR)
   coloredImage[newLocs] = 255



   newImage = np.stack((coloredImage,forStacking,forStacking),axis=1)

   print "finalImage shape", np.shape(newImage)
   tiff.imsave('falseColored.tiff',newImage.astype(np.uint8))




def membraneKnockOut():
  testImage = io.imread('/home/AD/srbl226/gpu3D/gpu_detect/Downsampled.tiff')
  dFilter = SphereFilter(size =32)
  test=testImage.transpose()
  im,time = MF(test,dFilter,useGPU=True,dim=3,xiters=iters,yiters=iters,ziters=iters)
  kImage = im.transpose()
  flatK = kImage.flatten()
  sixtyPercent = np.percentile(flatK,80)
  locs = np.argwhere(flatK>sixtyPercent)
  newIm = io.imread('/home/AD/srbl226/gpu3D/gpu_detect/Downsampled.tiff')
  I = np.copy(newIm)
  newLocs = np.unravel_index(locs, np.shape(kImage))
  I[newLocs] = 0
  tiff.imsave('deMembraned.tiff',I)
  radii = [16,8]
  for i,radius in enumerate(radii):
    testImage = io.imread('/home/AD/srbl226/gpu3D/gpu_detect/kMembraned.tiff')
    dFilter = SphereFilter(size =radius)
    test=testImage.transpose()
    im,time = MF(test,dFilter,useGPU=True,dim=3,xiters=iters,yiters=iters,ziters=iters)
    kImage = im.transpose()
    flatK = kImage.flatten()
    sixtyPercent = np.percentile(flatK,80)
    locs = np.argwhere(flatK>sixtyPercent)
    newIm = io.imread('/home/AD/srbl226/gpu3D/gpu_detect/Downsampled.tiff')
    I = np.copy(newIm)
    newLocs = np.unravel_index(locs, np.shape(kImage))
    I[newLocs] = 0
    tiff.imsave('deMembraned.tiff',I)
  print "file at deMembraned.tiff"


def smoother(dims=[1]): ### This fxn makes a convolved pic to compare to OG image
  dFilter = SphereFilter()
  for i  in [0,1]:
      for j in [0,1]:
          for k in [0,1]:
            testImage = Arrayer('/home/AD/srbl226/Fiji/140722_2_2.tif',[i*512,(i+1)*512,j*512,(j+1)*512,111*k,(k+1)*111])
            im,time = MF(testImage,dFilter,useGPU=True,dim=3,xiters=iters,yiters=iters,ziters=iters)
            #im = convolved ### grab convolved im here
            print "max of convolved: ", np.max(im)
            print "min of convolved: ", np.min(im)
            if not k:
                imZ = im
            else:
                imZ = np.concatenate((imZ,im),axis=2)
          if not j:
            imY = imZ
          else:
            imY = np.concatenate((imY,imZ),axis=1)
      if not i:
          imX = imY
      else:
          imX = np.concatenate((imX,imY),axis=0)
  print "final Dims", np.shape(imX)
  finalImage = imX.transpose()

  print "final Image shape", np.shape(finalImage)
  flatFinal = finalImage.flatten()
  onePercent = np.percentile(flatFinal,60)
  locs = np.argwhere(flatFinal>onePercent)
  newIm = io.imread('/home/AD/srbl226/Fiji/140722_2_2.tif')
  I = np.copy(newIm) #.transpose()
  #I.flatten()

  newLocs = np.unravel_index(locs, np.shape(finalImage))   # If breaks cry to Pete
  I[newLocs] = 0
  #IM = np.reshape(I,(222,1024,1024))
  tiff.imsave('deMembraned.tiff',I)
  finalImage*=(255/np.max(finalImage))
  #tiff.imsave('smoother_convolved.tiff', finalImage)


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
  import sys
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
    # calls 'test0' with the next argument following the argument '-validation'
    if(arg=="-validation"):
      #dims = [5,6,7,8,9,10]
      dims = [1]#[6,7]
      dims = map(lambda x: 2**x,dims)
      print "bouta testREal"
      times,timesGPU = testReal(dims)
      #print "CPU", times
      print"\n" + "GPU",timesGPU
      quit()

    if(arg=="-pic"):
      pictureMaker()
      print " made pic"
      quit()
    if(arg=="-smooth"):
      smoother()
      quit()
    if(arg=="-knocked"):
      kBrane()
      kBrane2()
      kBrane3()
      quit()
    if(arg=="-punish"):
      Punish()
      quit()
    if(arg=='-k3'):
      kBrane3()
      quit()
    if(arg=="-k"):
      membraneKnockOut()
      quit()
    if(arg=="-sub"):
      subset()
      quit()
    if(arg=="-test1"):
      test1()
      quit()
    if(arg=="-test2"):
      test1(maxDim=5000)
      quit()
    if(arg=="-running"):
      #dims = [5,6,7,8,9,10]
      dims = [1]
      dims = map(lambda x: 2**x,dims)
      times,timesGPU = runner(dims)
      print "CPU", times
      print"\n" + "GPU",timesGPU
      quit()






  raise RuntimeError("Arguments not understood")
