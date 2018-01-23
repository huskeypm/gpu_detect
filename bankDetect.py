"""
Finds all hits for rotated filter 
"""
#import matplotlib
#matplotlib.use('Agg')
#from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

class empty:pass

import cv2
import painter 
import numpy as np 
import matplotlib.pylab as plt 


##
## For a single matched filter, this function iterates over passed-in angles 
## and reports highest correlation output for each iteration 
## 
def DetectFilter(
  inputs,  # basically contains test data and matched filter 
  paramDict,  # parameter dictionary  
  iters,   # rotations over which mf will be tested
  display=False,
  label=None,
  filterMode=None,
  filterType="Pore",
  returnAngles=True,
):

  if inputs is None:
    raise RuntimeError("PLACEHOLDER TO REMIND ONE TO USE INPUT/PARAMDICT OBJECTS")

  # store
  result = empty()

  result.stackedDict = dict()


  # do correlations across all iter
  result.correlated = painter.correlateThresher(
     inputs,
     paramDict,
     iters=iters,
     printer=display,
     filterMode=filterMode,
     label=label,
     )

  # stack hits to form 'total field' of hits
  result.stackedHits= painter.StackHits(
    result.correlated,paramDict,iters, display=False)#,display=display)
    
  return result

def GetHits(aboveThresholdPoints):
  ## idenfity hits (binary)  
  mylocs =  np.zeros_like(aboveThresholdPoints.flatten())
  hits = np.argwhere(aboveThresholdPoints.flatten()>0)
  mylocs[hits] = 1
  
  #locs = mylocs.reshape((100,100,1))
  dims = np.shape(aboveThresholdPoints)  
  #locs = mylocs.reshape((dims[0],dims[1],1))  
  locs = mylocs.reshape(dims)  
  #print np.sum(locs)    
  #print np.shape(locs)  
  #plt.imshow(locs)  
  #print np.shape(please)
  #Not sure why this is needed  
  #zeros = np.zeros_like(locs)
  #please = np.concatenate((locs,zeros,zeros),axis=2)
  #print np.shape(please)
  return locs

# color red channel 
def ColorChannel(Img,stackedHits,chIdx=0):  
    locs = GetHits(stackedHits)   
    chFloat =np.array(Img[:,:,chIdx],dtype=np.float)
    #chFloat[10:20,10:20] += 255 
    chFloat+= 255*locs  
    chFloat[np.where(chFloat>255)]=255
    Img[:,:,chIdx] = np.array(chFloat,dtype=np.uint8)  



# red - entries where hits are to be colored (same size as rawOrig)
# will label in rawOrig detects in the 'red' channel as red, etc 
def colorHits(rawOrig,red=None,green=None,outName=None,label="",plotMe=True):
  dims = np.shape(rawOrig)  
  
  # make RGB version of data   
  Img = np.zeros([dims[0],dims[1],3],dtype=np.uint8)
  scale = 0.5  
  Img[:,:,0] = scale * rawOrig
  Img[:,:,1] = scale * rawOrig
  Img[:,:,2] = scale * rawOrig
    

  
  if isinstance(red, (list, tuple, np.ndarray)): 
    ColorChannel(Img,red,chIdx=0)
  if isinstance(green, (list, tuple, np.ndarray)): 
    ColorChannel(Img,green,chIdx=1)    

  if plotMe:
    plt.figure()  
    plt.subplot(1,2,1)
    plt.title("Raw data (%s)"%label)
    plt.imshow(rawOrig,cmap='gray')
    plt.subplot(1,2,2)
    plt.title("Marked") 
    plt.imshow(Img)  

  if outName!=None:
    plt.tight_layout()
    plt.gcf().savefig(outName,dpi=300)
    plt.close()
  else:
    plt.close()
    return Img  

def colorHitsTT(rawOrig,LongStacked,WTStacked,iters,outName=None,label='',plotMe=True,
                returnAngles=False):
  # need a function for coloring TT differently than PNP since we are using a
  # heatmap of red v blue to denote rotation degree
  dims=np.shape(rawOrig)

  # make RGB version
  Img = np.zeros([dims[0],dims[1],3],dtype=np.uint8)
  scale = 0.75
  Img[:,:,0] = scale * rawOrig
  Img[:,:,1] = scale * rawOrig
  Img[:,:,2] = scale * rawOrig

  # mark longitudinal hits without regard to rotation for now (green)
  # to do this we must modify the LongStacked array
  LongStacked[np.invert(np.isnan(LongStacked))] = 1 
  LongStacked = np.nan_to_num(LongStacked).astype('uint8')
  #plt.figure()
  #plt.imshow(LongStacked)
  #plt.colorbar()
  #plt.show()

  print "DC: refactor as colorAngles"
  ColorChannel(Img,LongStacked,chIdx=1)

  # mark WT wrt rotation degree via heatmap
  #leftMostRotationIdx = 0  # mark as red
  rightMostRotationIdx = len(iters) # mark as blue
  spacing = 255 / rightMostRotationIdx
  #print spacing

  for i in range(dims[0]):
    for j in range(dims[1]):
      # this contains our rotation degree wrt iters
      rotArg = WTStacked[i,j]
      if not np.isnan(rotArg):
        Img[i,j,0] = int(255 - rotArg*spacing)
        Img[i,j,2] = int(rotArg*spacing)
      else: 
        continue
  
  if plotMe:
    plt.figure()
    plt.subplot(1,2,1)
    plt.title("Raw data (%s)"%label)
    plt.imshow(rawOrig,cmap='gray')
    plt.subplot(1,2,2)
    plt.title("Marked")
    plt.imshow(Img)
    plt.close()
  if outName != None:
    plt.tight_layout()
    plt.gcf().savefig(outName,dpi=300)
    plt.close()
  else:
    return Img


# main engine 
# TODO remove scale/pass into filter itself
# tests filters1 and filters2 against a test data set
def TestFilters(
    testData, # data against which filters 1 and 2 are applied
    filter1Data, # matched filter 1
    filter2Data, # matched filter 2
    filter1Thresh=60,filter2Thresh=50,
    iters = [0,10,20,30,40,50,60,70,80,90], 
            
    display=False,
    colorHitsOutName=None,
    label="test",
    filterType="Pore",
    filterDict=None, thresholdDict=None,
    saveColoredFig=True,
    returnAngles=True,
    single = False,
######
    paramDict=None   # PUT ALL PARAMETERS HERE COMMON TO ALL FILTERS
):       

    #raise RuntimeError("Require Dataset object, as done for tissue validation") 
    params=paramDict
    if filterType == "Pore":
      ## perform detection 
      inputs = empty()
      inputs.imgOrig = testData

      
      daColors =dict()
      ### filter 1 
      inputs.mfOrig = filter1Data
      params['snrThresh'] = filter1Thresh
      filter1Result = DetectFilter(inputs,params,iters,
                                       display=display,filterMode="filter1",label=label)
      daColors['green']= filter1Result.stackedHits
      
      ### filter 2 
      if single is False:
        inputs.mfOrig = filter2Data
        params['snrThresh'] = filter2Thresh
        filter2Result = DetectFilter(inputs,params,iters,
                                       display=display,filterMode="filter2",label=label)
        daColors['red']= filter2Result.stackedHits
      else:
        filter2Result = None
        daColors['red'] =None
        
      # colorHits(asdfsdf, red=filter1output, green=filter2output)
      reportAngle = False
      if reportAngle:
        1
        # colorMyANgles(data,filterOutput)

      ## display results 
      if colorHitsOutName!=None: 
        colorHits(testData,
                red=daColors['red'],
                green=daColors['green'],
                label=label,
                outName=colorHitsOutName)                       

      # DC this could be generalized into a dictionary to add more filter; might make ROC difficult though 
      return filter1Result, filter2Result 

    elif filterType == "TT":
      print "filterType=TT is deprecated"

      print "DC: merge up the case-specific coloring for now"

      # utilizing runner functions to produce stacked images
      print "DC: this is wehre you'll want to iterature over WT, Longi and loss" 
      raise RuntimeError("Exiting here, since DetectFilter inputs different now")
      
      paramsDict = {'doCLAHE':False}
      print "Currently not returning angles to reduce computational load. Be sure to change"
      resultContainer = DetectFilter(inputs, paramDict, iters, returnAngles=False)


      if colorHitsOutName != None and saveColoredFig:
        # need to update once I have working code 
        colorImg = testDataName * 255
        colorImg = colorImg.astype('uint8')
        colorHits(colorImg, red=resultContainer.stackedHits.WT, green=resultContainer.stackedHits.Long,
                  #blue=resultContainer.stackedHits.Loss,
                  label=label,outName=colorHitsOutName)
      elif colorHitsOutName != None and not saveColoredFig:
        colorImg = testDataName
        # keep in mind when calling this function that red and green are for the matplotlib convention.
        # CV2 spits out red -> blue

        # changing since we return the angle at which the maximum response is
        resultContainer.coloredImg = colorHitsTT(colorImg.copy(), resultContainer.stackedHits.Long,
                                                 resultContainer.stackedHits.WT, iters,
                                                 label=label,outName=None,plotMe=False)
        if returnAngles:
          resultContainer.coloredAngles = colorHitsTT(colorImg.copy(), resultContainer.stackedHits.Long,
                                                      resultContainer.stackedHits.WT, iters,
                                                      returnAngles=True,
                                                      label=label,outName=None,plotMe=False)
      return resultContainer
    else:
      raise RuntimeError, "Filtering type not understood"


  
  
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



