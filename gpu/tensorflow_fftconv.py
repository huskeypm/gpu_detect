# https://github.com/tensorflow/tensorflow/issues/6541
import numpy as np
import tensorflow as tf
import time
import matplotlib.pylab as plt
import scipy.fftpack as fftp
import cv2
import os.path



#wav = np.random.random_sample((N,))
#img = np.fft.fft(wav)[:N/2+1]

def saveimg(imgFloat,outName="out.tif"):
    # convert to unsigned image 
    out = np.array(imgFloat,dtype=np.uint8)
    cv2.imwrite("out.png",out)
    plt.pcolormesh(out)
    plt.gcf().savefig("x.png")

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
      xF  =tf.fft2d(x)
      mFF =tf.fft2d(mf)
      out = tf.multiply(xF,mFF) # elementwise multiplication for convolutiojn 
      xR  =tf.ifft2d(out)
      ## ------
  
      cntnew=cnt-1
      return xR, mf,cntnew
  
    start = time.time()
  
    final, mfo,cnt= tf.while_loop(condition, body, 
                              [specVar,filtVar,cnt], parallel_iterations=1)
    final, mfo,cnt =  sess.run([final,mfo,cnt])
  
    tElapsed = time.time()-start
    print 'tensorflow:{}s'.format(tElapsed)
  
    return final, tElapsed

def doFFTPloop(img,mFs,
               nrots=4   # like filter rotations) 
): 
  ##  
  ## numpy loop for comparison
  ## 
  start = time.time()
  x = img 
  for i in range(nrots):
      xF  =fftp.fft2(x)
      mFF =fftp.fft2(mFs)
      out = np.multiply(xF,mFF) # elementwise multiplication for convolutiojn 
      final  =fftp.ifft2(out)
  
  tElapsed = np.float(time.time()-start)
  print 'numpy:{}s'.format(tElapsed)
  return final, tElapsed

 
def TestImg(
  imgName = "test.png",
  maxDim=-1  # truncate image s.t. max dimensions are less than 0..maxDim
  ):

  assert( os.path.exists(imgName) ),imgName +" does not exist"
  
  img = cv2.imread(imgName)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  img = np.array(img, dtype=np.float)

  if maxDim>0:
    maxDim = np.int(maxDim)
    img = img[0:maxDim,0:maxDim]
    cv2.imwrite("cropped.tif",img)
  
  dimx,dimy = np.shape(img)
  print dimx,dimy
  saveimg(img,outName="out.tif")
  quit()
  
  mF = np.zeros_like(img)
  m = 8
  mF[(-m + dimx/2):(m+dimx/2),(-m + dimy/2):(m+dimy/2)]=1.
  mFs = fftp.fftshift(mF)
  ### NOTE: should technically pad image such that dimensions are a log2 integer....

  doTFloop(img,mFs)
  doFFTPloop(img,mFs)

def TestRandom():
  Ns = 2**np.array([7,8,9,10,11,12])
  tTFs=np.zeros_like(Ns,dtype=np.float)
  tFFTPs=np.zeros_like(Ns,dtype=np.float)

  nrots = 20
  for i, N in enumerate(Ns): 
    print N
    img = np.reshape( np.random.rand(N*N), [N,N])
    mF = np.copy(img) 

    tTFs[i] = doTFloop(img,mF,nrots=nrots)
    tFFTPs[i] =doFFTPloop(img,mF,nrots=nrots)

  # plot 
  fig, ax = plt.subplots()
  ax.set_xscale('log', basex=2)
  ax.set_yscale('log', basey=2)
  ax.set_title("time elapsed vs. NxN img") 
  ax.set_ylabel("time [s]") 
  ax.set_xlabel("N [px/dim]")
  ax.plot(Ns,tTFs,label="GPU") 
  ax.plot(Ns,tFFTPs,label="CPU") 
  ax.legend(loc=0)
  plt.gcf().savefig("scaling2.png") 


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

  # Loops over each argument in the command line 
  for i,arg in enumerate(sys.argv):
    # calls 'doit' with the next argument following the argument '-validation'
    if(arg=="-validation"):
      arg1=sys.argv[i+1] 
      raise RuntimeError("need to reintro") 
    #mode="benchmark"
    #mode="testimg"
    #mode="testimgbig"

    if arg == "-benchmark": 
      TestRandom()
      quit()
    elif arg =="-testimg":
      TestImg()
      quit()
    elif arg =="-testimgbig":
      # too big for the 780 GPU 
      imgName = "/home/AD/pmke226/DataLocker/cardiac/Sachse/171127_tissue/tissue.tif"
      maxDim = 1e4
      # rot 35 degrees 
      # filter 
      # 14 px wide
      # 25 tall 
      # z-line about 3-4 px wide
      TestImg(imgName=imgName,maxDim=maxDim)
      quit()
  







