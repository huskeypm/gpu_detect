import matplotlib.pylab as plt 
import numpy as np 
import cv2

root = "myoimages/"

def myplot(img,fileName=None,clim=None):
  plt.axis('equal')
  plt.pcolormesh(img, cmap='gray')
  plt.colorbar()
  if fileName!=None:
    plt.gcf().savefig(fileName,dpi=300)
  if clim!=None:
    plt.clim(clim)

def ReadImg(fileName,renorm=False,bound=False):
    img = cv2.imread(fileName)
    if img is None:
        raise RuntimeError(fileName+" likely doesn't exist")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if bound!=False:
	img=img[bound[0]:bound[1],bound[0]:bound[1]]
    if renorm:# rescaling
    	img = img / np.float(np.amax(img))
    return img  

import scipy.fftpack as fftp

# Prepare matrix of vectorized of FFT'd images
def CalcX(
  imgs,
  debug=False
  ):
  nImg,d1,d2 = np.shape(imgs)
  dd = d1*d2  
  #print nImg, d2
  X = np.zeros([nImg,dd],np.dtype(np.complex128))
    
  for i,img in enumerate(imgs):
    xi = np.array(img,np.dtype(np.complex128))     
    # FFT (don't think I need to shift here)  
    Xi = fftp.fft2( xi )    
    if debug:
      Xi = xi    
    #myplot(np.real(Xi))
    # flatten
    Xif = np.ndarray.flatten(Xi)
    X[i,:]=Xif
  return X  

def TestFilter(
  H, # MACE filter
  I  # test img
):
    print "DEPREACATE THIS FUNCTION" 
    #R = fftp.ifftshift(fftp.ifft2(I*conj(H)));
    icH = I * np.conj(H)
    R = fftp.ifftshift ( fftp.ifft2(icH) ) 
    #R = fftp.ifft2(icH) 

    daMax = np.max(np.real(R))
    print "Response %e"%( daMax )
    #myplot(R)
    return R,daMax

# renormalizes images to exist from 0-255
# rescale/renomalize image 
def renorm(img,scale=255):
    img = img-np.min(img)
    img/= np.max(img)
    img*=scale 
    return img

def GetAnnulus(region,sidx,innerMargin,outerMargin=None):
  if outerMargin==None: 
      # other function wasn't really an annulus 
      raise RuntimeError("Antiquated. See GetRegion")

  if innerMargin%2==0 or outerMargin%2==0:
      print "WARNING: should use odd values for margin!" 

  # grab entire region
  outerRegion,dummy,dummy = GetRegion(region,sidx,outerMargin)

  # block out interior to create annulus 
  annulus = np.copy(outerRegion) 
  s = np.shape(annulus)
  aM = outerMargin - innerMargin
  xMin,xMax = 0+aM, s[0]-aM
  yMin,yMax = 0+aM, s[1]-aM
  interior = np.copy(annulus[xMin:xMax,yMin:yMax])
  annulus[xMin:xMax,yMin:yMax]=0. 

  return annulus,interior

def GetRegion(region,sidx,margin):
      subregion = region[(sidx[0]-margin):(sidx[0]+margin+1),
                         (sidx[1]-margin):(sidx[1]+margin+1)]        
      area = np.float(np.prod(np.shape(subregion)))
      intVal = np.sum(subregion)  
      return subregion, intVal, area

def MaskRegion(region,sidx,margin,value=0):
      region[(sidx[0]-margin):(sidx[0]+margin+1),
                         (sidx[1]-margin):(sidx[1]+margin+1)]=value  


def ApplyCLAHE(grayImgList, tileGridSize, clipLimit=2.0, plot=False):
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    clahedimages = []
    for i,img in enumerate(grayImgList):
        clahedImage = clahe.apply(img)
        clahedimages.append(clahedImage)
    return clahedimages

# function to take raw myocyte png name, read, resize, renorm, CLAHE, save output
def preprocessPNG(imgName, twoSarcSize, filterTwoSarcSize):
  img = ReadImg(imgName)
  scale = float(filterTwoSarcSize) / float(twoSarcSize)
  rescaledImg = cv2.resize(img,None,fx=scale,fy=scale,interpolation=cv2.INTER_CUBIC)
  normed = renorm(rescaledImg)
  clahed = ApplyCLAHE(normed,filterTwoSarcSize)
  name, filetype = imgName.split('.')
  cv2.imwrite(name+'_processed'+filetype,clahed)
  
# # Generating filters
def GenerateWTFilter(WTFilterRoot=root+"/filterImgs/WT/", filterTwoSarcSize=25):
  WTFilterImgs = []
  import os
  for fileName in os.listdir(WTFilterRoot):
      img = cv2.imread(WTFilterRoot+fileName)
      gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
      # let's try and measure two sarc size based on column px intensity separation
      colSum = np.sum(gray,axis=0)
      colSum = colSum.astype('float')
      # get slopes to automatically determine two sarcolemma size for the filter images
      rolledColSum = np.roll(colSum,-1)
      slopes = rolledColSum - colSum
      slopes = slopes[:-1]
      
      idxs = []
      for i in range(len(slopes)-1):
          if slopes[i] > 0 and slopes[i+1] <= 0:
            idxs.append(i)
      if len(idxs) > 2:
          raise RuntimeError, "You have more than two peaks striations in your filter, think about discarding this image"
    
      twoSarcDist = 2 * (idxs[-1] - idxs[0])
      scale = float(filterTwoSarcSize) / float(twoSarcDist)
      resizedFilterImg = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
      WTFilterImgs.append(resizedFilterImg)

  minHeightImgs = min(map(lambda x: np.shape(x)[0], WTFilterImgs))
  for i,img in enumerate(WTFilterImgs):
      WTFilterImgs[i] = img[:minHeightImgs,:]

  colSums = []
  sortedWTFilterImgs = sorted(WTFilterImgs, key=lambda x: np.shape(x)[1])
  minNumCols = np.shape(sortedWTFilterImgs[0])[1]
  for img in sortedWTFilterImgs:
      colSum = np.sum(img,axis=0)
    
  bestIdxs = []
  for i in range(len(sortedWTFilterImgs)-1):
      # line up each striation with the one directly 'above' it in list
      img = sortedWTFilterImgs[i].copy()
      img = img.astype('float')
      lengthImg = np.shape(img)[1]
      nextImg = sortedWTFilterImgs[i+1].copy()
      nextImg = nextImg.astype('float')
      nextLengthImg = np.shape(nextImg)[1]
      errOld = 10e10
      bestIdx = 0
      for idx in range(nextLengthImg - minNumCols):
          err = np.sum(np.power(np.sum(nextImg[:,idx:(minNumCols+idx)],axis=0) - np.sum(img,axis=0),2))
          if err < errOld:
              bestIdx = idx
              errOld = err
      bestIdxs.append(bestIdx)
      sortedWTFilterImgs[i+1] = nextImg[:,bestIdx:(minNumCols+bestIdx)]

  WTFilter = np.mean(np.asarray(sortedWTFilterImgs),axis=0)
  WTFilter /= np.max(WTFilter)
  return WTFilter

def GenerateLongFilter(filterRoot, twoSarcLengthDict, filterTwoSarcLength=24):
  import os
  import operator
  '''
  Input
 
   - Directory where the filter images are located
   - Dictionary containing two sarcolemma size for the images (measured previously)
   - Desired filter two sarcolemma size
  '''
  # Parameters - Sample Input
  '''
  filterRoot = "./images/Remodeled_TTs/"
  twoSarcLengthDict = {"SongWKY_long1":8,
                       "Xie_RV_Control_long2":7,
                       "Xie_RV_Control_long1":8,
                       "Guo2013Fig1C_long1":11
                       }
  filterTwoSarcLength = 24
  '''
  # Read in images, gray scale, normalize
  colLengths = {}
  rowLengths = {}
  imgDict = {}
  colSums = {}
  rowSums = {}
  #colSlopes = {}
  for fileName in os.listdir(filterRoot):
      if 'long' in fileName:
          img = cv2.imread(filterRoot+fileName)
          goodName = fileName.split(".")[0]
          gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
          gray = gray.astype('float')
          gray /= np.max(gray)
          scale = float(filterTwoSarcLength) / float(twoSarcLengthDict[goodName])
          resized = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
          imgDim = np.shape(resized)
          colLengths[goodName] = imgDim[1]
          rowLengths[goodName] = imgDim[0]
          colSums[goodName] = np.sum(resized,axis=0)
          rowSums[goodName] = np.sum(resized,axis=1)
          imgDict[goodName] = resized

  # ### Line up based on the column sums (lining up the TT striation)
  # First order images in terms of their column lengths
  sortedColLengths = sorted(colLengths, key=operator.itemgetter(1), reverse=True)
  minLength = colLengths[sortedColLengths[0]]
  
  # Now iterate through the list of images, lining up the column lengths
  for i in range(len(sortedColLengths) - 1):
      img = imgDict[sortedColLengths[i]]
      imgLength = colLengths[sortedColLengths[i]]
      nextImg = imgDict[sortedColLengths[i+1]]
      nextImgLength = colLengths[sortedColLengths[i+1]]
      errOld = 10e15
      bestIdx = 0
      for idx in range(nextImgLength - minLength):
          err = np.sum(np.power(np.sum(nextImg[:,idx:(minLength+idx)],axis=0) - np.sum(img,axis=0),2))
          if err < errOld:
              bestIdx = idx
              errOld = err
      imgDict[sortedColLengths[i+1]] = imgDict[sortedColLengths[i+1]][:,bestIdx:(minLength+bestIdx)]
  # ### Line up based on the row sums (lining up the transverse portion)
  # Repeating the same procedure as above
  sortedRowLengths = sorted(rowLengths, key=operator.itemgetter(1), reverse=True)
  minHeight = rowLengths[sortedRowLengths[0]]
  for i in range(len(sortedRowLengths) - 1):
      img = imgDict[sortedRowLengths[i]]
      imgHeight = rowLengths[sortedRowLengths[i]]
      nextImg = imgDict[sortedRowLengths[i+1]]
      nextImgHeight = rowLengths[sortedRowLengths[i+1]]
      errOld = 10e15
      bestIdx = 0
      for idx in range(nextImgHeight - minHeight - 1):
          err = np.sum(np.power(np.sum(nextImg[idx:(minHeight+idx),:],axis=1) - np.sum(img,axis=1),2))
          if err < errOld:
              bestIdx = idx
              errOld = err
      imgDict[sortedRowLengths[i+1]] = nextImg[bestIdx:(minHeight+bestIdx),:]
  # ### Output the image
  avgImg = np.sum(np.asarray([v for (k,v) in imgDict.iteritems()]),axis=0)
  avgImg /= np.max(avgImg)
  return avgImg

def GenerateLossFilter(LossFilterName, scale):
  '''
  routine to generate the loss filter used in the myocyte filtering routines.
  '''
  lossFilter = cv2.imread(LossFilterName)
  gray = cv2.cvtColor(lossFilter, cv2.COLOR_BGR2GRAY)
  resized = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC).astype('float')
  resized /= np.max(resized)
  resized = 1 - resized
  resized[resized < 0.3] = 0
  return resized

def GenerateWTPunishmentFilter(LongitudinalFilterName,
                               rowMin=2,rowMax=-1,colMin=6,colMax=13):
  # generates the punishment filter in WT SNR calculation based upon longitudinal filter
  LongFilter = ReadImg(LongitudinalFilterName)
  punishFilter = LongFilter.copy()
  if rowMax == None:
    punishFilter = punishFilter[rowMin:,:]
  else:
    punishFilter = punishFilter[rowMin:rowMax,:]
  if colMax == None:
    punishFilter = punishFilter[:,colMin:]
  else:
    punishFilter = punishFilter[:,colMin:colMax]
  return punishFilter


# # 'Fixing' filters (improving contrast) via basic thresholding
def fixFilter(Filter,pixelCeiling=0.7,pixelFloor=0.4,rowMin=0, rowMax=None, colMin=0, colMax=None):
  fixedFilter = Filter.copy()
  fixedFilter[fixedFilter > pixelCeiling] = pixelCeiling
  fixedFilter[fixedFilter < pixelFloor] = 0
  fixedFilter /= np.max(fixedFilter)
  if rowMax == None:
    fixedFilter = fixedFilter[rowMin:,:]
  else:
    fixedFilter = fixedFilter[rowMin:rowMax,:]
  if colMax == None:
    fixedFilter = fixedFilter[:,colMin:]
  else:
    fixedFilter = fixedFilter[:,colMin:colMax]
  return fixedFilter

def SaveFixedWTFilter(WTFilterRoot=root+"filterImgs/WT/",filterTwoSarcSize=25,
                      pixelCeiling=0.6,pixelFloor=0.25,
                      rowMin=20, rowMax=None, colMin=1, colMax=None):
  # opting now to save WT filter and load into the workhorse script instead of generating filter every call
  WTFilter = GenerateWTFilter(WTFilterRoot=WTFilterRoot,filterTwoSarcSize=filterTwoSarcSize)
  fixedFilter = fixFilter(WTFilter,pixelCeiling=pixelCeiling,pixelFloor=pixelFloor,
                          rowMin=rowMin,rowMax=rowMax,colMin=colMin,colMax=colMax)
  # convert to png format
  savedFilt = fixedFilter * 255
  savedFilt = savedFilt.astype('uint8')
  # save filter
  cv2.imwrite(root+"WTFilter.png",savedFilt)

def SaveFixedLongFilter(LongFilterRoot=root+"filterImgs/Longitudinal/",
                        filterTwoSarcSizeDict={"SongWKY_long1":16, "Xie_RV_Control_long2":14, "Xie_RV_Control_long1":16, "Guo2013Fig1C_long1":22},
                        filterTwoSarcLength=25,
                        pixelCeiling=0.7,pixelFloor=0.4,
                        rowMin=6, rowMax=13, colMin=0, colMax=-1):
  # opting now to save Long filter and load into the workhorse script instead of generating filter every call
  Filter = GenerateLongFilter(LongFilterRoot,filterTwoSarcSizeDict,filterTwoSarcLength=filterTwoSarcLength)
  fixedFilter = fixFilter(Filter,pixelCeiling=pixelCeiling,pixelFloor=pixelFloor,
                          rowMin=rowMin,rowMax=rowMax,colMin=colMin,colMax=colMax)
  # convert to png format
  savedFilt = fixedFilter * 255
  savedFilt = savedFilt.astype('uint8')
  # save filter
  cv2.imwrite(root+"LongFilter.png",savedFilt)

def SaveFixedLossFilter(LossFilterName=root+"filterImgs/Loss/TT_Idealized_Loss_TruthFilter.png",
                        LossScale=float(25)/float(28)):
  LossFilter = GenerateLossFilter(LossFilterName, LossScale) * 255
  cv2.imwrite(root+"LossFilter.png",LossFilter)

def SaveFixedPunishmentFilter(LongitudinalFilterName=root+"LongFilter.png",
                              rowMin=2,rowMax=-1,colMin=6,colMax=13):
  punishFilter = GenerateWTPunishmentFilter(LongitudinalFilterName,
                                            rowMin=2,rowMax=-1,colMin=6,colMax=13)
  cv2.imwrite(root+"WTPunishmentFilter.png",punishFilter)

def PadWithZeros(img, padding = 15):
  '''
  routine to pad your image with a border of zeros. This reduces the 
  unwanted response from shifting the nyquist.
  '''

  imgType = type(img[0,0])
  imgDim = np.shape(img)
  newImg = np.zeros([imgDim[0]+2*padding, imgDim[1]+2*padding])
  newImg[padding:-padding,padding:-padding] = img
  newImg = newImg.astype(imgType)
  return newImg

def Depad(img, padding=15):
  '''
  routine to return the img passed into 'PadWithZeros' 
  '''
  imgType = type(img[0,0])
  imgDim = np.shape(img)
  newImg = img[padding:-padding,padding:-padding]
  newImg = newImg.astype(imgType)
  return newImg

def PasteFilter(img, filt):
  '''
  function to paste the filter in the upper left region of the img 
  to make sure they are scaled correctly
  '''
    
  myImg = img.copy()
  filtDim = np.shape(filt)
  myImg[:filtDim[0],:filtDim[1]] = filt
  return myImg

def SaveAllMyo():
      SaveFixedWTFilter()
      SaveFixedLongFilter()
      SaveFixedLossFilter()
      SaveFixedPunishmentFilter()

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

  # Loops over each argument in the command line 
  for i,arg in enumerate(sys.argv):
    if(arg=="-genWT"):
      SaveFixedWTFilter()
    elif(arg=="-genLong"):
      SaveFixedLongFilter()
    elif(arg=="-genLoss"):
      SaveFixedLossFilter()
    elif(arg=="-genPunishment"):
      SaveFixedPunishmentFilter()
    elif(arg=="-genAllMyo"): 
      SaveAllMyo()

    elif(i>0):
      raise RuntimeError("Arguments not understood")




