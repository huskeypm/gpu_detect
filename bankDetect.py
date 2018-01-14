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
def DetectFilter(dataSet,mf,threshold,iters,display=False,sigma_n=1.,
                 label=None,filterMode=None,useFilterInv=False,penaltyscale=1.,
                 doCLAHE=True,filterType="Pore",returnAngles=True):

  # store
  result = empty()
  # difference for TT routines these are now dictionaries
  result.threshold = threshold
  result.mf= mf

  if filterType == "Pore":
    # do correlations across all iter
    result.correlated = painter.correlateThresher(
       dataSet,result.mf, threshold = result.threshold, iters=iters,
       printer=display,
       sigma_n=sigma_n,
       penaltyscale=penaltyscale,
       filterMode=filterMode,
       useFilterInv=useFilterInv,
       label=label,
       doCLAHE=doCLAHE)

    # record snr 
    #snrs = [] 
    #for i, resulti in enumerate(result.correlated):
    #  maxSNR = np.max( resulti.snr) 
    #  snrs.append( maxSNR )              
    #result.snrs = np.array( snrs)
    #result.iters = iters 
  
    # stack hits to form 'total field' of hits
    result.stackedHits= painter.StackHits(
      result.correlated,result.threshold,iters,doKMeans=False, display=False)#,display=display)
    
  elif filterType == "TT":
    print "WARNING: Need to consolidate this with filterType=Pore"
    result.correlated = painter.correlateThresherTT(
       dataSet,result.mf, 
       thresholdDict=result.threshold,iters=iters,doCLAHE=doCLAHE)

    # stack filter hits
    result.stackedHits,result.stackedAngles = painter.StackHits(result.correlated,
                                                                threshold,iters,display=display,
                                                                doKMeans=False,
                                                                filterType="TT",returnAngles=returnAngles)
    #print result.stackedHits.WT
    #quit() 
  
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
def TestFilters(testDataName,
                filter1FilterName,filter2FilterName,
                testData = None, # can pass in (ultimately preferred) or if none, will read in based on dataName 
                filter1Data=None,
                filter2Data=None,
                filter1Thresh=60,filter2Thresh=50,
                subsection=None,
                display=False,
                colorHitsOutName=None,
                sigma_n = 1., 
                iters = [0,10,20,30,40,50,60,70,80,90], 
                penaltyscale=1.0,      
                useFilterInv=False,
                label="test",
                filterType="Pore",
                filterDict=None, thresholdDict=None,
                doCLAHE=True,saveColoredFig=True,
                gamma=3.,
                returnAngles=True):       

    print "DC: do away with the filterType Pore distinction here"
    print "DC: can keep the case-specific coloring for now"
    if filterType == "Pore":
      if testData is None: 
        # load data against which filters are tested
        testData = cv2.imread(testDataName)
        testData = cv2.cvtColor(testData, cv2.COLOR_BGR2GRAY)
    
        if isinstance(subsection, (list, tuple, np.ndarray)): 
          testData = testData[subsection[0]:subsection[1],subsection[2]:subsection[3]]
        print "WARNING: should really use the SetupTests function, as we'll retire this later" 

      ## should offloat elsewhere
      if filter1Data is None:
        print "WARNING: should really use the SetupTests function, as we'll retire this later" 
        # load fused filter
        filter1Filter = cv2.imread(filter1FilterName)
        filter1Data   = cv2.cvtColor(filter1Filter, cv2.COLOR_BGR2GRAY)

      if filter2Data is None:
        print "WARNING: should really use the SetupTests function, as we'll retire this later" 
        # load bulk filter 
        filter2Filter = cv2.imread(filter2FilterName)
        filter2Data   = cv2.cvtColor(filter2Filter, cv2.COLOR_BGR2GRAY)


      ## perform detection 
      print "DC: this part can be replaced with a dictionary of filters" 
      filter1PoreResult = DetectFilter(testData,filter1Data  ,filter1Thresh,
                                     iters,display=display,sigma_n=sigma_n,
                                     filterMode="filter1",label=label,
                                     penaltyscale=penaltyscale,
                                     useFilterInv=useFilterInv)

      filter2PoreResult = DetectFilter( testData,filter2Data  ,filter2Thresh,
                                     iters,display=display,sigma_n=sigma_n,
                                     filterMode="filter2",label=label,
                                     penaltyscale=penaltyscale,
                                     useFilterInv=useFilterInv)

      print "DC: color channels by dictionary results"
      # colorHits(asdfsdf, red=filter1output, green=filter2output)
      reportAngle = False
      if reportAngle:
        1
        # colorMyANgles(data,filterOutput)

      ## display results 
      if colorHitsOutName!=None: 
        colorHits(testData,
                red=filter2PoreResult.stackedHits,
                green=filter1PoreResult.stackedHits,
                label=label,
                outName=colorHitsOutName)                       

      return filter1PoreResult, filter2PoreResult 

    elif filterType == "TT":
      # moving dictionary abstraction up a lvl from DetectFilters to TestFilters
      #resultsDict = {}
      #for filterName,filterArray in filterDict.iteritems():
      #  resultsDict[filterName] = DetectFilter(testDataName,filterArray,thresholdDict[filterName],
      #                                         iters,display=display,sigma_n=sigma_n,filterType="TT",
      #                                         doCLAHE=doCLAHE,returnAngles=returnAngles)



      # utilizing runner functions to produce stacked images
      resultContainer = DetectFilter(testDataName,
                                     filterDict,thresholdDict,
                                     iters,display=display,sigma_n=sigma_n,
                                     filterType="TT",
                                     doCLAHE=doCLAHE,returnAngles=returnAngles)

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


def TestTrueData():
  root 							= "/net/share/shared/papers/nanoporous/images/"
  img1 = '/home/AD/srbl226/spark/sparkdetection/roc/clahe_Best.jpg'
  img2 = root+"full.png"
  dummy = TestFilters(
    img1, # testData
    root+'fusedBase.png',         # fusedfilter Name
    root+'bulkCellTEM.png',        # bulkFilter name
    #subsection=[200,400,200,500],   # subsection of testData
    subsection=[200,400,200,500],   # subsection of testData
    fusedThresh = 6.,  
    bulkThresh = 6., 
    colorHitsoutName = "filters_on_pristine.png",
    display=False   
  )  
  
  
  
  
  
  
  
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
    # calls 'doit' with the next argument following the argument '-validation'
    if(arg=="-validation"):
      #arg1=sys.argv[i+1] 
      TestTrueData() 
      quit()






  raise RuntimeError("Arguments not understood")


