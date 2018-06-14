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
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#import cPickle as Pickle

height = 50
iters = [-30,-15,0,15,30]

class empty:
  pass


#import imutils
def LoadImage(
  imgName = "/home/AD/pmke226/DataLocker/cardiac/Sachse/171127_tissue/tissue.tif",
  mid = 10000,
  maxDim = 100,
  angle = -35.    
):
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
        dFilter,
        height=height):
    fdim = np.shape(dFilter)
    imgDim = np.shape(imgR)
    assert (fdim[2] <= imgDim[2]), "Your filter is larger in the z direction than your image"
    filterPadded = np.zeros_like( imgR)
    #print "dFilter dims", np.shape(dFilter)
    filterPadded[0:fdim[0],0:fdim[1],0:fdim[2]] = dFilter
    hfw = np.int(fdim[0]/2.)
    hfwx = np.int(fdim[1]/2.)
    hfwz = np.int(fdim[2]/2.)

    # we do this rolling here in the same way we shift for FFTs
    yroll = np.roll(filterPadded,-hfw,axis=0)
    xyroll = np.roll(yroll,-hfwx,axis=1)
    xyzroll = np.roll(xyroll,-hfwz, axis=2)
    
    #filt = np.reshape(cross,(np.shape(img2)[0],np.shape(img2)[1],height))
    return xyzroll

def tfMF(img,mf):
  '''
  Generic workhorse routine that sets up the image and matched filter for 
    matched filtering using tensorflow. This routine is called by whichever
    detection scheme has been chosen. NOT by the doTFloop explicitly.

  INPUTS:
    img - 'tensorized' image
    mf - rotated and 'tensorized' matched filter same dimension as img
  '''

  xF = tf.fft3d(img)
  #Why are we taking the complex conjugate here??
  xFc = tf.conj(xF)
  mFF = tf.fft3d(mf)
  out = tf.multiply(xFc,mFF)
  xR = tf.ifft3d(out)

  return xR


##
## Tensor flow part 
##
def doTFloop(inputs,
           #img,# test image
           #mFs, # shifted filter
           paramDict,
           xiters=iters,
           yiters=iters,
           ziters=iters
           ):

  # We may potentially want to incorporate all 3 filters into this low level
  # loop to really speed things up
  
  with tf.Session() as sess:
    start = time.time()
    # Create and initialize variables
    #NOTE: double check imgOrig and mfOrig
    tfImg = tf.Variable(inputs.imgOrig, dtype=tf.complex64)
    dims = np.shape(inputs.imgOrig)
    paddedFilter = Pad(inputs.imgOrig,inputs.mfOrig,height=dims[2])
    tfFilt = tf.Variable(paddedFilter, dtype=tf.complex64)
    stackedHitsDummy = np.zeros_like(inputs.imgOrig)
    stackedHits = tf.Variable(stackedHitsDummy, dtype=tf.bool)

    # make big angle container
    numXits = np.shape(xiters)[0]
    numYits = np.shape(yiters)[0]
    numZits = np.shape(ziters)[0]
    cnt = tf.Variable(tf.constant(numXits * numYits * numZits-1))

    #sess.run(tf.variables_initializer([specVar,filtVar,cnt]))

    # It's late and I'm getting lazy
    bigIters = []
    for i in xiters:
      for j in yiters:
        for k in ziters:
          bigIters.append([i,j,k])
    # have to convert to a tensor so that the rotations can be indexed during tf while loop
    bigIters = tf.Variable(tf.convert_to_tensor(bigIters,dtype=tf.float64))

    # set up filtering variables
    if paramDict['filterMode'] == 'punishmentFilter':
      paramDict['mfPunishment'] = Pad(inputs.imgOrig,paramDict['mfPunishment'])
      paramDict['mfPunishment'] = tf.Variable(paramDict['mfPunishment'],dtype=tf.complex64)
      paramDict['covarianceMatrix'] = tf.Variable(paramDict['covarianceMatrix'],dtype=tf.complex64)
      paramDict['gamma'] = tf.Variable(paramDict['gamma'],dtype=tf.complex64)
      sess.run(tf.variables_initializer([paramDict['mfPunishment'],paramDict['covarianceMatrix'],paramDict['gamma']]))

    sess.run(tf.variables_initializer([tfImg,tfFilt,cnt,bigIters,stackedHits]))

    inputs.tfImg = tfImg
    inputs.tfFilt = tfFilt
    inputs.bigIters = bigIters


    # While loop that counts down to zero and computes reverse and forward fft's
    def condition(#inputs,paramDict,
                  cnt,stackedHits):
      return cnt > 0

    def body(#inputs,paramDict,
             cnt,stackedHits):
            #img,mf,results,cnt,bigIters,paramDict):
      # pick out rotation to use
      rotations = bigIters[cnt]

      # rotating matched filter to specific angle
      rotatedMF = util.rotateFilterCube3D(inputs.mfOrig,
                                          rotations[0],
                                          rotations[1],
                                          rotations[2])

      # get detection/snr results
      snr = doDetection(inputs,paramDict)
      stackedHitsNew = doStackingHits(inputs,paramDict,stackedHits,snr)
      #stackedHitsNew = stackedHits

      cntnew=cnt-1
      return cntnew,stackedHitsNew

    # can we optimize parallel_iterations based on memory allocation?
    '''
    NOTE: when images get huge (~>512x512x50) and we use parallel iterations=10, they eat a lot of memory
          since the intermediate results from the while loop are stored for back propogation.
          This memory consumption can be offloaded if we use swap_memory=True in the tf.while_loop.
          However, this slows down the calculation immensely, so we want to always cleverly pick
          the sweet spot between NOT offloading the memory eating tensors and NOT bricking the computer.
    '''
    # TODO: See if there is a way to grab maximum available memory for the GPU to automatically
    #       and cleverly determine the sweetspot to where we don't have to offload tensors to cpu
    #       but can also efficiently determine parallel_iterations number
    cnt,stackedHits = tf.while_loop(condition, body,
                                                 [#inputs,paramDict,
                                                  cnt,stackedHits],
                                                 parallel_iterations=10)
    compStart = time.time()
    cnt,stackedHits =  sess.run([#inputs,paramDict,
                                 cnt,stackedHits])
    compFin = time.time()
    print "Time for tensor flow to execute run:{}s".format(compFin-compStart)

    results = empty()
    results.stackedHits = np.real(stackedHits) 

    #start = time.time()
    tElapsed = time.time()-start
    print 'Total time for tensorflow to run:{}s'.format(tElapsed)


    return results, tElapsed


def MF(
    dImage,
    dFilter,
    useGPU=False,
    dim=2,
    xiters=iters,
    yiters=iters,
    ziters=iters
    ):
    # NOTE: May need to do this padding within tensorflow loop itself to 
    #       ameliorate latency due to loading large matrices into GPU.
    #       Potentially a use for tf.dynamic_stitch?
    lx = len(xiters); ly = len(yiters); lz = len(ziters)
    numRots = lx * ly * lz
    filt = Pad(dImage,dFilter)
    if useGPU:
       corr,tElapsed = doTFloop(dImage,filt,xiters=xiters,yiters=yiters,ziters=ziters)
       corr = np.real(corr)
    else:        
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
       
    # for some reason output seems to be rotated by 180 deg?
    if dim==2:
      corr = imutils.rotate(corr,180.)
    else:
      #corr = tf.flip_left_right(corr)
      #print "Stubbornly refusing to do anything since 3D" 
      this = np.flip(corr,1)
      return this, tElapsed
    return corr,tElapsed    

###################################################################################################
###
### Detection Schemes: taken from detection_protocols.py and 'tensorized'
###                    See detection_protocols.py for further documentation of functions
###################################################################################################

def doDetection(inputs,paramDict):
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
    results = punishmentFilterTensor(inputs,paramDict)
  elif mode=="simple":
    results = simpleDetectTensor(inputs,paramDict)
  elif mode=="regionalDeviation":
    results =regionalDeviationTensor(inputs,paramDict)
  elif mode=="filterRatio":
    print "Ignoring this for now. Bug the developers to implement if you want this detection scheme."
    quit()
  else:
    print "That mode isn't understood. Returning image instead."
    results = inputs.img

  return results

def punishmentFilterTensor(inputs,paramDict):
  # grab already tensorized image and mf
  #img = inputs.tfImg
  #mf = inputs.tfFilt
  #mfPunishment = paramDict['mfPunishment']

  # all of this setup and such should be handled at a higher level
  #try:
  #  mfPunishment = tf.convert_to_tensor(paramDict['mfPunishment'],dtype=tf.complex64)
  #except:
  #  raise RuntimeError("No punishment filter was found in paramDict['mfPunishment']")
  #try:
  #  cM = tf.convert_to_tensor(paramDict['covarianceMatrix'],dtype=tf.complex64)
  #except:
  #  raise RuntimeError("Covariance Matrix Undefined. Replace this with a matrix of all ones the same size as the image")
  #try:
  #  gamma = tf.constant(paramDict['gamma'],dtype=tf.complex64)
  #except:
  #  raise RuntimeError("Punishment filter weighting term (gamma) not found\
  #                        within paramDict")


  # call generalized tensorflow matched filter routine
  corr = tfMF(inputs.imgOrig,inputs.tfFilt)
  corrPunishment = tfMF(inputs.imgOrig,paramDict['mfPunishment'])
  # calculate signal to noise ratio
  snr = corr / (paramDict['covarianceMatrix'] + paramDict['gamma'] * corrPunishment)
  return snr

def simpleDetectTensor(inputs,paramDict):
  1

def regionalDeviationTensor(inputs,paramDict):
  2

def doStackingHits(inputs,paramDict,stackedHits,snr):
  '''
  Function to threshold the calculated snr and apply to the stackedHits container
  '''
  snr = tf.cast(snr,dtype=tf.float64)
  #stackedHits[tf.where(snr > paramDict['snrThresh'])] = 255 

  stackedHits = tf.logical_or(stackedHits,snr>paramDict['snrThresh'])
  return stackedHits

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
    #plt.pcolormesh(out)
    #plt.gcf().savefig("x.png")

    
    



#!/usr/bin/env python
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
  corr = MF(testImage,dFilter,useGPU=useGPU,dim=3) 
  #writer(corr)
  return testImage,corr

def runner(dims):
  times =[]
  f = open("GPU_Benchmark.txt","w")
  f.write('Dims:{}'.format(dims))
  f.write('\nCPU:')
  for i,d in enumerate(dims):
    print "dim", d
    testImage = MakeTestImage(d)
    dFilter = MakeFilter()
    corr,time = MF(testImage,dFilter,useGPU=False,dim=3)
    times.append(time)
    #if dim == dims[-1]:
    #f.write('\ndim:{},time:{};'.format(dim,time))
    #f.write('dim:{},time:{};'.format(dim,time))
  f.write('{}'.format(times))

  timesGPU =[]
  f.write('\nGPU:') 
  for j,d in enumerate(dims):
    print "dim", d
    testImage = MakeTestImage(d)
    dFilter = MakeFilter()
    corr,time = MF(testImage,dFilter,useGPU=True,dim=3)
    timesGPU.append(time)
    #f.write('\ndim:{},time:{};'.format(dim,time))
  f.write('{}'.format(timesGPU))

  #Results = { "CPU":"%s"%(str(times)),"GPU":"%s"%str(timesGPU)}
  #pickle.dump(Results, open("%Benchmark.p"%(str(R),str(length/nm),str(cKCl)),"wb"))
  
  
  return times, timesGPU
def test1(maxDim=100):
  testImage = LoadImage(maxDim=maxDim)
  dFilter = MakeFilter()
  corr = MF(testImage,dFilter,useGPU=True,dim=3) 
  writer(corr)
  return testImage,corr, dFilter




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
    # calls 'test0' with the next argument following the argument '-validation'
    if(arg=="-validation"):
      test0()      
      test0(useGPU = False)
      quit()


    if(arg=="-test1"):
      test1()
      quit()
    if(arg=="-test2"):
      test1(maxDim=5000)
      quit()
    if(arg=="-running"):
      #dims = np.linspace(50,1000,11,endpoint=True)
      #dims = np.linspace(50,1000,11,endpoint=True)
      dims = [5,6,7,8,9,10]
      dims = map(lambda x: 2**x,dims)
      times,timesGPU = runner(dims)
      print "CPU", times
      print"\n" + "GPU",timesGPU
      quit()
  





  raise RuntimeError("Arguments not understood")




