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
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#import cPickle as Pickle

height = 10

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
    fdim = 14
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

def Pad(
    imgR,dFilter):
    fdim = np.shape(dFilter)[0]
    filterPadded = np.zeros_like( imgR[:,:,0])
    #print "dFilter dims", np.shape(dFilter)
    filterPadded[0:fdim,0:fdim] = dFilter[:,:,0]
    hfw = np.int(fdim/2.)
    #print hfw

    xroll = np.roll(filterPadded,-hfw,axis=0)
    xyroll = np.roll(xroll,-hfw,axis=1)
    #xyzroll = np.roll(xyroll,-hfw, axis=2)
    #oners = np.ondes(np.shape(imgR[2]))
    #filt = np.outer(xyroll,ones)
    #cv2.imshow(filt,cmap="gray")
    img2 = xyroll
    height2 = np.ones((height))
    cross = np.outer(img2, height2)

    filt = np.reshape(cross,(np.shape(img2)[0],np.shape(img2)[1],height))
    return filt


##
## Tensor flow part 
##
def doTFloop(img,# test image
           mFs, # shifted filter
           nrots=4   # like filter rotations) 
           ):

  with tf.Session() as sess:
    # Create and initialize variables
    cnt = tf.Variable(tf.constant(nrots))
    specVar = tf.Variable(img, dtype=tf.complex64)
    filtVar = tf.Variable(mFs, dtype=tf.complex64)
    sess.run(tf.variables_initializer([specVar,filtVar,cnt]))

    # While loop that counts down to zero and computes reverse and forward fft's
    def condition(x,mf,cnt):
      return cnt > 0

    def body(x,mf,cnt):
      ## Essentially the matched filtering parts 
      xF  =tf.fft3d(x)
      xFc = tf.conj(xF)
      mFF =tf.fft3d(mf)
      out = tf.multiply(xFc,mFF) # elementwise multiplication for convolutiojn 
      xR  =tf.ifft3d(out)
      # DOESNT LIKE THIS xRr = tf.real(xR)
      ## ------

      cntnew=cnt-1
      return xR, mf,cntnew

    start = time.time()

    final, mfo,cnt= tf.while_loop(condition, body,
                              [specVar,filtVar,cnt], parallel_iterations=1)
    final, mfo,cnt =  sess.run([final,mfo,cnt])
    corr = np.real(final) 

    #start = time.time()
    tElapsed = time.time()-start
    print 'tensorflow:{}s'.format(tElapsed)


    return final, tElapsed


def MF(
    dImage,
    dFilter,
    useGPU=False,
    dim=2
    ):
    filt = Pad(dImage,dFilter)
    #print "PRE FILTER"
    if useGPU:
        # NOTE: I pass in an 'nrots' argument, but it doesn't actually do anything (e.g. 'some assembly required')
       corr,tElapsed = doTFloop(dImage,filt,nrots=1)
       corr = np.real(corr)
       #print "POST FILTER"
    else:        
       start = time.time()
       I = dImage
       T = filt
       fI = fftp.fftn(I)
       fT = fftp.fftn(T)
       c = np.conj(fI)*fT
       corr = fftp.ifftn(c)
       corr = np.real(corr)
       tElapsed = time.time()-start
       print 'fftp:{}s'.format(tElapsed)
       
    #cv2.imshow(dImage,cmap="gray")
    #plt.figure()
    # I'm doing something funky here, s.t. my output is rotated by 180. wtf
    if dim==2:
      corr = imutils.rotate(corr,180.)
    else:
      #corr = tf.flip_left_right(corr)
      #print "Stubbornly refusing to do anything since 3D" 
      this = np.flip(corr,1)
      return this, tElapsed
    return corr,tElapsed    

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
      dims = np.linspace(50,1000,11,endpoint=True)
      times,timesGPU = runner(dims)
      print "CPU", times
      print"\n" + "GPU",timesGPU
      quit()
  





  raise RuntimeError("Arguments not understood")




