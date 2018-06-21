import sys
import os
import matplotlib.pylab as plt 
import numpy as np 
import cv2
import scipy
import scipy.signal as sig
import scipy.fftpack as fftp
import imutils
import operator
import tensorflow as tf
 

root = "myoimages/"

def myplot(img,fileName=None,clim=None):
  plt.axis('equal')
  plt.pcolormesh(img, cmap='gray')
  plt.colorbar()
  if fileName!=None:
    plt.gcf().savefig(fileName,dpi=300)
  if clim!=None:
    plt.clim(clim)

def ReadImg(fileName,cvtColor=True,renorm=False,bound=False):
    img = cv2.imread(fileName)
    if img is None:
        raise RuntimeError(fileName+" likely doesn't exist")
    if cvtColor:
      img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if bound!=False:
	img=img[bound[0]:bound[1],bound[0]:bound[1]]
    if renorm:# rescaling
    	img = img / np.float(np.amax(img))
    return img  

def LoadFilter(fileName):
  filterImg = ReadImg(fileName,cvtColor=True,renorm=True).astype(np.float64)
  filterImg /=  np.sum(filterImg)
  return filterImg

def measureFilterDimensions(grayFilter):
  '''
  Measures where the filter has any data and returns a minimum bounding
  rectangle around the filter. To be used in conjunction with the 
  pasteFilters flag in DataSet
  '''
  filtery,filterx = np.shape(grayFilter)

  collapsedRows = np.sum(grayFilter,0)
  leftPadding = np.argmax(collapsedRows>0)
  rightPadding = np.argmax(collapsedRows[::-1]>0)
  numRows = filtery - leftPadding - rightPadding

  collapsedCols = np.sum(grayFilter,1)
  topPadding = np.argmax(collapsedCols>0)
  bottomPadding = np.argmax(collapsedCols[::-1]>0)
  numCols = filterx - topPadding - bottomPadding

  print "filter y,x:",numRows,numCols

  return numRows, numCols

def makeCubeFilter(prismFilter):
  '''
  Function to make a filter that is sufficiently padded with zeros such that 
  any rotation performed on the filter will not cause the filter information
  to be clipped by any rotation algorithm.
  '''
  # get shape of old filter
  fy,fx,fz = np.shape(prismFilter)

  # get shape of new cubic filter
  biggestDimension = np.max((fy,fx,fz))
  newDim = int(np.ceil(np.sqrt(2) * biggestDimension))
  if newDim % 2 != 0:
    newDim += 1

  # construct holder for new filter
  cubeFilt = np.zeros((newDim,newDim,newDim),dtype=np.float64)
  center = newDim / 2

  # store old filter in the new filter
  cubeFilt[center - int(np.floor(fy/2.)):center + int(np.ceil(fy/2.)),
           center - int(np.floor(fx/2.)):center + int(np.ceil(fx/2.)),
           center - int(np.floor(fz/2.)):center + int(np.ceil(fz/2.))] = prismFilter

  return cubeFilt

def viewer(tensor):
    def dummy(tensor2):
        return tensor2

    with tf.Session() as sess:
        result = sess.run(dummy(tensor))
        print np.shape(result)
        print result
        return result

def rotateFilterCube3D(image,rot1,rot2,rot3):
  '''
  Function to rotate a 3D matrix according to three angles of rotation
  image - 3D np.array that has previously been ran through util.makeCubeFilter
          as to ensure that no information will be lost due to rotation.
          New image has the same size as the input image.
          NOTE: Convention is that image is referenced as 
                image[x,y,z] 
  rot1 - tensorflow constant rotation angle (in degrees) COUNTERCLOCKWISE about the x axis
  rot2 - tensorflow constant rotation angle (in degrees) COUNTERCLOCKWISE about the y axis
  rot3 - tensorflow constant rotation angle (in degrees) COUNTERCLOCKWISE about the z axis

  HEAVILY modified version of code found on:
  https://stackoverflow.com/questions/34801342/tensorflow-how-to-rotate-an-image-for-data-augmentation
  '''

  #rot1 = tf.constant(-rot1 / 180. * np.pi,dtype=tf.float64)
  #rot2 = tf.constant(-rot2 / 180. * np.pi,dtype=tf.float64)
  #rot3 = tf.constant(-rot3 / 180. * np.pi,dtype=tf.float64)

  rot1 = tf.multiply(-rot1, np.pi/180.)
  rot2 = tf.multiply(-rot2, np.pi/180.)
  rot3 = tf.multiply(-rot3, np.pi/180.)

  rot1 = tf.cast(rot1,dtype=tf.float64)
  rot2 = tf.cast(rot2,dtype=tf.float64)
  rot3 = tf.cast(rot3,dtype=tf.float64)


  #image = tf.convert_to_tensor(image,dtype=tf.float64)
  image = tf.cast(image, dtype=tf.float64)

  s = image.get_shape().as_list()

  assert len(s) == 3, "Input needs to be 3D."
  #assert (mode == 'repeat') or (mode == 'black') or (mode == 'white') or (mode == 'ones'), "Unknown boundary mode."
  image_center = [np.floor(x/2) for x in s]

  # Coordinates of new image
  coord1 = tf.range(s[0])
  coord2 = tf.range(s[1])
  coord3 = tf.range(s[2])

  # Create vectors of those coordinates in order to vectorize the image
  coord1_vec = tf.tile(coord1, [s[1]*s[2]]) # get first index for all points

  coord2_vec_unordered = tf.tile(coord2, [s[0]*s[2]])
  coord2_vec_unordered = tf.reshape(coord2_vec_unordered, [s[1], -1])
  coord2_vec = tf.reshape(tf.transpose(coord2_vec_unordered, [1, 0]), [-1]) # get second index for all points

  coord3_vec_unordered = tf.tile(coord3, [s[0]*s[1]])
  # something really strange is happening with this reshape. Most likely due to dominant axis or something
  coord3_vec_unordered = tf.reshape(coord3_vec_unordered, [s[0]*s[1],s[2]])
  coord3_vec = tf.reshape(tf.transpose(coord3_vec_unordered, [1, 0]), [-1])# get third index for all points

  # center coordinates since rotation center is supposed to be in the image center
  coord1_vec_centered = coord1_vec - image_center[0]
  coord2_vec_centered = coord2_vec - image_center[1]
  coord3_vec_centered = coord3_vec - image_center[2]
  coord_new_centered = tf.cast(tf.stack([coord1_vec_centered, coord2_vec_centered, coord3_vec_centered]), tf.float64)

  # Perform backward transformation of the image coordinates
  thisZero = tf.constant(0,dtype=tf.float64)
  thisOne = tf.constant(1,dtype=tf.float64)
  rot_mat_inv_1 = tf.reshape([tf.cos(rot1), thisZero, tf.sin(rot1),
                              thisZero     , thisOne , thisZero,
                              -tf.sin(rot1),thisZero, tf.cos(rot1)],shape=[3,3])
  rot_mat_inv_2 = tf.reshape([thisOne , thisZero     , thisZero,
                              thisZero, tf.cos(rot2), -tf.sin(rot2),
                              thisZero, tf.sin(rot2), tf.cos(rot2)],shape=[3,3])
  rot_mat_inv_3 = tf.reshape([tf.cos(rot3), -tf.sin(rot3), thisZero,
                              tf.sin(rot3), tf.cos(rot3) , thisZero,
                              thisZero    , thisZero     , thisOne],shape=[3,3])
  rot_mat_inv = tf.matmul(rot_mat_inv_1,rot_mat_inv_2)
  rot_mat_inv = tf.matmul(rot_mat_inv,rot_mat_inv_3)

  coord_old_centered = tf.matmul(rot_mat_inv, coord_new_centered)

  # Find nearest neighbor in old image
  coord1_old_nn = tf.cast(tf.round(coord_old_centered[0, :] + image_center[0]), tf.int32)
  coord2_old_nn = tf.cast(tf.round(coord_old_centered[1, :] + image_center[1]), tf.int32)
  coord3_old_nn = tf.cast(tf.round(coord_old_centered[2, :] + image_center[2]), tf.int32)

  # Clip values to stay inside image coordinates
  outside_ind1 = tf.logical_or(tf.greater(coord1_old_nn, s[0]-1), tf.less(coord1_old_nn, 0))
  outside_ind2 = tf.logical_or(tf.greater(coord2_old_nn, s[1]-1), tf.less(coord2_old_nn, 0))
  outside_ind3 = tf.logical_or(tf.greater(coord3_old_nn, s[2]-1), tf.less(coord3_old_nn, 0))
  outside_ind = tf.logical_or(outside_ind1, outside_ind2)
  outside_ind = tf.logical_or(outside_ind,  outside_ind3)

  coord_old1_clipped = tf.boolean_mask(coord1_old_nn, tf.logical_not(outside_ind))
  coord_old2_clipped = tf.boolean_mask(coord2_old_nn, tf.logical_not(outside_ind))
  coord_old3_clipped = tf.boolean_mask(coord3_old_nn, tf.logical_not(outside_ind))

  coord1_vec = tf.boolean_mask(coord1_vec, tf.logical_not(outside_ind))
  coord2_vec = tf.boolean_mask(coord2_vec, tf.logical_not(outside_ind))
  coord3_vec = tf.boolean_mask(coord3_vec, tf.logical_not(outside_ind))

  coord_old_clipped = tf.cast(tf.transpose(tf.stack([coord_old1_clipped, coord_old2_clipped, coord_old3_clipped]), [1, 0]), tf.int32)

  # Coordinates of the new image
  coord_new = tf.transpose(tf.cast(tf.stack([coord1_vec, coord2_vec, coord3_vec]), tf.int32), [1, 0])
  # new values for the image
  image_new_values = tf.gather_nd(image,coord_old_clipped)
  background_color = 0
  rotated = tf.sparse_to_dense(coord_new,[s[0],s[1],s[2]],image_new_values, background_color,
                               validate_indices=False)
  rotated = tf.cast(rotated,tf.complex64)
  return rotated

def rotateFilter2D(img,rotation):
  '''
  Function to 'tensorize' the rotation of the filter
  '''
  #rotated = tf.cast(img,dtype=tf.float64)
  #rotated = tf.cast(img,dtype=tf.float32)
  rotated = tf.to_float(img)
  rotated = tf.contrib.image.rotate(rotated,rotation,interpolation="BILINEAR")
  rotated = tf.cast(rotated,dtype=tf.complex64)
  return rotated


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
    clahedImage = clahe.apply(grayImgList) # stupid hack
    return clahedImage

### Generating filters
def generateWTFilter(WTFilterRoot=root+"/filterImgs/WT/", filterTwoSarcSize=25):
  WTFilterImgs = []
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

def generateLongFilter(filterRoot, twoSarcLengthDict, filterTwoSarcLength=24):
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

def generateWTPunishmentFilter(LongitudinalFilterName,
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

def saveSingleTTFilter():
  '''
  Generates a bar filter for detection of a single transverse tubule
  '''
  WTfilter = np.zeros((20,12),dtype=np.float64)
  WTfilter[:,3:-3] = 1.
  WTfilter *= 255.
  WTfilter = WTfilter.astype(np.uint8)
  cv2.imwrite("./myoimages/singleTTFilter.png",WTfilter)

def saveSingleTTPunishmentFilter():
  '''
  Generates a punishment corollary to the generateSingleTTFilter() function
  above.
  '''
  punishFilter = np.zeros((20,12),dtype=np.float64)
  punishFilter[:,:3] = 1.; punishFilter[:,-3:] = 1.
  punishFilter *= 255.
  punishFilter =  punishFilter.astype(np.uint8)
  cv2.imwrite("./myoimages/singleTTPunishmentFilter.png",punishFilter)


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

def saveFixedWTFilter(WTFilterRoot=root+"filterImgs/WT/",filterTwoSarcSize=25,
                      pixelCeiling=0.6,pixelFloor=0.25,
                      rowMin=20, rowMax=None, colMin=1, colMax=None):
  # opting now to save WT filter and load into the workhorse script instead of generating filter every call
  WTFilter = generateWTFilter(WTFilterRoot=WTFilterRoot,filterTwoSarcSize=filterTwoSarcSize)
  fixedFilter = fixFilter(WTFilter,pixelCeiling=pixelCeiling,pixelFloor=pixelFloor,
                          rowMin=rowMin,rowMax=rowMax,colMin=colMin,colMax=colMax)
  # convert to png format
  savedFilt = fixedFilter * 255
  savedFilt = savedFilt.astype('uint8')

  # cropping image
  savedFilt = savedFilt[6:,:]

  # save filter
  cv2.imwrite(root+"WTFilter.png",savedFilt)

def saveSimpleWTFilter():
  '''
  function to write the wt filter used as of June 5, 2018
  '''
  filterLength = 10
  TTwidth = 6
  bufferWidth = 1
  punishWidth = 5
  filterWidth = TTwidth + bufferWidth + punishWidth + bufferWidth + TTwidth
  WTfilter = np.zeros((filterLength,filterWidth),dtype=np.uint8)
  punishFilter = np.zeros_like(WTfilter)

  WTfilter[:,:TTwidth] = 255
  WTfilter[:,-TTwidth:] = 255 

  punishFilter[:,(TTwidth+bufferWidth):-(TTwidth+bufferWidth)] = 255

  cv2.imwrite("./myoimages/newSimpleWTFilter.png", WTfilter)
  cv2.imwrite("./myoimages/newSimpleWTPunishmentFilter.png",punishFilter)

def saveFixedLongFilter(LongFilterRoot=root+"filterImgs/Longitudinal/",
                        filterTwoSarcSizeDict={"SongWKY_long1":16, "Xie_RV_Control_long2":14, "Xie_RV_Control_long1":16, "Guo2013Fig1C_long1":22},
                        filterTwoSarcLength=25,
                        pixelCeiling=0.7,pixelFloor=0.4,
                        rowMin=8, rowMax=11, colMin=0, colMax=-1):
  # opting now to save Long filter and load into the workhorse script instead of generating filter every call
  Filter = generateLongFilter(LongFilterRoot,filterTwoSarcSizeDict,filterTwoSarcLength=filterTwoSarcLength)
  fixedFilter = fixFilter(Filter,pixelCeiling=pixelCeiling,pixelFloor=pixelFloor,
                          rowMin=rowMin,rowMax=rowMax,colMin=colMin,colMax=colMax)
  # convert to png format
  savedFilt = fixedFilter * 255
  savedFilt = savedFilt.astype('uint8')
  # save filter
  cv2.imwrite(root+"LongFilter.png",savedFilt)

def saveWeirdLongFilter():
  filt = np.zeros((6,17),dtype='uint8')
  filt[1:-1,6:11] = 255

  punish = np.zeros((6,17),dtype='uint8')
  punish[1:-1,1:5] = 255
  punish[1:-1,12:-1] = 255

  cv2.imwrite("./myoimages/weirdLTfilter.png", filt)
  cv2.imwrite("./myoimages/weirdLTPunishmentfilter.png", punish)

def saveGaussLongFilter():
  ### Take Some Measurements of LT Growths in Real Myocytes
  height = 3 # pixels
  width = 15 # pixels
  
  ### Make Gaussian for Filter
  std = 4
  squish = 1.2
  gauss = sig.gaussian(width,std)
  gauss /= squish
  gauss += (1 - 1./squish)

  ### Generate Filter
  LTfilter = np.zeros((height+1,width+2))
  imgDim = np.shape(LTfilter)
  cY,cX = int(round(imgDim[0]/2.)),int(round(imgDim[1]/2.))
  loc = 1
  while loc < height:
    LTfilter[loc,1:-1] = gauss
    loc += 1

  ### Save in CV2 Friendly Format
  LTfilter *= 255
  LTfilter = np.asarray(LTfilter,np.uint8)
  cv2.imwrite("./myoimages/LongitudinalFilter.png",LTfilter)

def saveFixedLossFilter():
  dim = 16
  img = np.zeros((dim+2,dim+2,),dtype='uint8')
  img[2:-2,2:-2] = 255
  cv2.imwrite(root+"LossFilter.png",img)

def saveFixedPunishmentFilter(LongitudinalFilterName=root+"LongFilter.png",
                              rowMin=2,rowMax=-1,colMin=6,colMax=13):
  #punishFilter = GenerateWTPunishmentFilter(LongitudinalFilterName,
  #                                          rowMin=2,rowMax=-1,colMin=7,colMax=12)
  punishFilter = np.zeros((28,7),dtype='uint8')
  punishFilter[1:-1,2:-2] = 255
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

def CalcPSD(Hs): # fourier xformed data
    psd = np.real(np.conj(Hs)*Hs)
    eps = 1e-5
    psd[ psd < eps ] = eps
    return fftp.fftshift(np.log( psd ))


def dissimilar(
    theFilter,
    theDecoy,
    Cv = 1.,
    Clutter = 1.,
    beta = 0.,
    gamma = 0.0001):

    s = 1.
    s2 = 3.

    # filter FFT
    kernel = np.ones((s2,s2),np.float32)/np.float(s2*s2)
    h = np.array(theFilter,dtype=np.float)
    h = cv2.resize(h, None, fx = s, fy = s, interpolation = cv2.INTER_CUBIC)
    
    plt.figure()
    plt.subplot(2,2,1)
    plt.imshow(h,cmap="gray")
    hs = fftp.fftshift(h)
    Hs = fftp.fftn(hs)
    plt.subplot(2,2,2)
    plt.imshow(CalcPSD(Hs))

    # noise FFT 
    Cn = Cv*np.ones_like(h,dtype=np.float)

    # clutter FFT
    Cc = np.ones_like(h,dtype=np.float)

    # difference image 
    p = cv2.resize(theDecoy, None, fx = s, fy = s, interpolation = cv2.INTER_CUBIC)
    p = cv2.filter2D(p,-1,kernel)
    k = p - h
    k[ k < 0] = 0
    plt.subplot(2,2,3)
    plt.imshow(k,cmap='gray')
    ks = fftp.fftshift(k)
    Ks = fftp.fftn(ks)
    print np.min(k), np.min(Ks)
    #Ks = cv2.filter2D(np.real(Ks),-1,kernel)
    #Ks = cv2.filter2D(np.real(Ks),-1,kernel)
    #Ks = cv2.filter2D(np.real(Ks),-1,kernel)    
    plt.subplot(2,2,4)
    plt.imshow(CalcPSD(Ks))

    ### modified filter
    Fs = Hs / (Cn + beta*Cc + gamma*np.abs(Ks))
    
    fs = fftp.ifftn(Fs)
    f  = fftp.ifftshift(fs)
    f-= np.min(f); f/=np.max(f); f*=255
    f = np.array(np.real(f),dtype=np.uint8)
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(f,cmap="gray")
    plt.subplot(1,2,2)
    plt.imshow(CalcPSD(Fs))
    
    
    f = cv2.resize(f, None, fx = 1/s, fy = 1/s, interpolation = cv2.INTER_CUBIC)
    return f, Hs,Ks

def PasteFilter(img, filt):
  '''
  function to paste the filter in the upper left region of the img 
  to make sure they are scaled correctly
  '''
    
  myImg = img.copy()
  filtDim = np.shape(filt)
  myImg[:filtDim[0],:filtDim[1]] = filt
  return myImg

def saveAllMyo():
      saveFixedWTFilter()
      saveSimpleWTFilter()
      #saveFixedLongFilter()
      saveGaussLongFilter()
      saveFixedLossFilter()
      saveFixedPunishmentFilter()
      saveSingleTTFilter()
      saveSingleTTPunishmentFilter()

# Embegs signal into known iomage for testing 
def embedSignal(img,mf,loc=None,scale=0.5):
    #plt.figure()
    s= np.max(img)*scale
    mfs= np.array(mf*s,dtype=np.uint8)
    imgEmb = np.copy(img)
    dimr = np.shape(mf)
    if isinstance(loc,np.ndarray):
      1
    else: 
      loc = [0,0]
    imgEmb[loc[0]:(loc[0]+dimr[0]),loc[1]:(loc[1]+dimr[1])] += mfs 
    #imshow(imgEmb)
    return imgEmb

def padWithZeros(array, padwidth, iaxis, kwargs):
    array[:padwidth[0]] = 0
    array[-padwidth[1]:]= 0
    return array


def PadRotate(myFilter1,val):
  dims = np.shape(myFilter1)
  diff = np.min(dims)
  paddedFilter = np.lib.pad(myFilter1,diff,padWithZeros)
  rotatedFilter = imutils.rotate(paddedFilter,-val)
  rF = np.copy(rotatedFilter)

  return rF


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

  # Loops over each argument in the command line 
  for i,arg in enumerate(sys.argv):
    if(arg=="-genWT"):
      saveFixedWTFilter()
    elif(arg=="-genSimpleWTFilter"):
      saveSimpleWT()
    elif(arg=="-genLong"):
      print "WARNING: DEPRECATED. Use -genGaussLongFilter"
      saveFixedLongFilter()
    elif(arg=="-genLoss"):
      saveFixedLossFilter()
    elif(arg=="-genPunishment"):
      saveFixedPunishmentFilter()
    elif(arg=="-genAllMyo"): 
      saveAllMyo()
    elif(arg=="-preprocess"):
      print "WARNING THIS IS DEPRECATED. USE PREPROCESSING.PY FILE NOW!!!!!"
      imgName = sys.argv[i+1]
      imgTwoSarcSize = float(sys.argv[i+2])
      try:
        filterTwoSarcSize = int(sys.argv[i+3])
      except:
        filterTwoSarcSize = 25
      preprocessPNG(imgName, imgTwoSarcSize, filterTwoSarcSize)
      quit()
    elif(arg=="-genWeirdLong"):
      print "WARNING: DEPRECATED. Use -genGaussLongFilter"
      saveWeirdLongFilter()
      quit()
    elif(arg=="-genSimpleLong"):
      print "WARNING: DEPRECATED. Use -genGaussLongFilter"
      saveSimpleLongFilter()
      quit()
    elif(arg=="-genGaussLong"):
      saveGaussLongFilter()
      quit()

    elif(i>0):
      raise RuntimeError("Arguments not understood")






