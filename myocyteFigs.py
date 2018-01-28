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
import ROCstacker as Rs
import numpy as np
import cv2
import matplotlib.pylab as plt
import bankDetect as bD
import util
import optimizer
class empty:pass

root = "myoimages/"

## WT 
def fig3(): 
  testImage = root+"Sham_M_65_annotation.png"
  figAnalysis(
    testImage=testImage,
    tag = "WT", 
    writeImage=True) 

  # write angle 
  print "DC: UPDATE THRESHOLDS TO OPTIMIZED PARAMETERS (from ROC) " 
  results = Rs.giveStackedHits(testImage, 0.06, 0.38, 3.)
  print results.stackedAngles.WT
                           


## HF 
def fig4(): 

  filterTwoSarcSize = 25

  imgName = root+"HF_1.png"
  twoSarcSize = 21

  rawImg = util.ReadImg(imgName)

  markedImg = giveMarkedMyocyte(testImage=imgName,ImgTwoSarcSize=twoSarcSize)

  wtContent, ltContent, lossContent = assessContent(markedImg)
  contents = np.asarray([wtContent, ltContent, lossContent])
  dims = np.shape(markedImg[:,:,0])
  area = float(dims[0]) * float(dims[1])
  contents = np.divide(contents, area)
  normedContents = contents / np.max(contents)

  # generating figure
  width = 0.25
  colors = ["blue","green","red"]
  marks = ["WT","LT","Loss"]

  # opting to make a single bar chart
  N = 1
  indices = np.arange(N) + width
  fig,ax = plt.subplots()
  rects1 = ax.bar(indices, normedContents[0], width, color=colors[0])
  rects2 = ax.bar(indices+width, normedContents[1], width, color=colors[1])
  rects3 = ax.bar(indices+2*width, normedContents[2], width, color=colors[2])
  ax.set_ylabel('Normalized Content')
  ax.legend(marks)
  ax.set_xticks([])
  plt.gcf().savefig('fig4_BarChart.png')
 
  # constructing actual figure
  fig, axarr = plt.subplots(2,1)
  axarr[0].imshow(rawImg,cmap='gray')
  axarr[0].set_title("HF Raw")
  axarr[0].axis('off') 

  switchedImg = switchBRChannels(markedImg)
  axarr[1].imshow(switchedImg)
  axarr[1].set_title("HF Marked")
  axarr[1].axis('off')
  plt.gcf().savefig("fig4_RawAndMarked.png")

## MI 
def fig5(): 
  # update if need be
  filterTwoSarcSize = 25

  # Distal, Medial, Proximal
  DImageName = root+"MI_D_78.png"
  DTwoSarcSize = 22
  MImageName = root+"MI_M_45.png"
  MTwoSarcSize = 21
  PImageName = root+"MI_P_16.png"
  PTwoSarcSize = 21

  imgNames = [DImageName, MImageName, PImageName]
  ImgTwoSarcSizes = [DTwoSarcSize,MTwoSarcSize,PTwoSarcSize]

  # Read in images for figure
  DImage = util.ReadImg(DImageName)
  MImage = util.ReadImg(MImageName)
  PImage = util.ReadImg(PImageName)
  images = [DImage, MImage, PImage]

  # BE SURE TO UPDATE TESTMF WITH OPTIMIZED PARAMS
  Dimg = giveMarkedMyocyte(testImage=DImageName,ImgTwoSarcSize=DTwoSarcSize)
  Mimg = giveMarkedMyocyte(testImage=MImageName,ImgTwoSarcSize=MTwoSarcSize)
  Pimg = giveMarkedMyocyte(testImage=PImageName,ImgTwoSarcSize=PTwoSarcSize)

  results = [Dimg, Mimg, Pimg]
  keys = ['D', 'M', 'P']
  areas = {}

  ttResults = []
  ltResults = []
  lossResults = []

  # report responses for each case
  for i,img in enumerate(results):
    print "Replace with assessContent function"
    dimensions = np.shape(img)
    wtChannel = img[:,:,0].copy()
    wtChannel[wtChannel == 255] = 1
    wtChannel[wtChannel != 1] = 0
    ltChannel = img[:,:,1].copy()
    ltChannel[ltChannel == 255] = 1
    ltChannel[ltChannel != 1] = 0
    lossChannel = img[:,:,2].copy()
    lossChannel[lossChannel == 255] = 1
    lossChannel[lossChannel != 1] = 0
    area = float(dimensions[0] * dimensions[1])
    ttContent = np.sum(wtChannel) / area
    ltContent = np.sum(ltChannel) / area
    lossContent = np.sum(lossChannel) / area
    # construct array of areas and norm 
    newAreas = np.array([ttContent, ltContent, lossContent])
    normedAreas = newAreas / np.max(newAreas)
    areas[keys[i]] = normedAreas
    # store in lists
    ttResults.append(normedAreas[0])
    ltResults.append(normedAreas[1])
    lossResults.append(normedAreas[2])

  # generating figure
  #fig, axarr = plt.subplots(3,3)
  width = 0.25
  colors = ["blue","green","red"]
  marks = ["WT","LT","Loss"]

  # opting to make a single bar chart
  N = 3
  indices = np.arange(N) + width
  fig,ax = plt.subplots()
  rects1 = ax.bar(indices, ttResults, width, color=colors[0])
  rects2 = ax.bar(indices+width, ltResults, width, color=colors[1])
  rects3 = ax.bar(indices+2*width, lossResults,width, color=colors[2])
  ax.set_ylabel('Normalized Content')
  ax.set_xticks(indices + width* 3/2)
  ax.set_xticklabels(keys)
  ax.legend(marks)
  plt.gcf().savefig('fig5_BarChart.png')

  # saving individual marked images
  fig, axarr = plt.subplots(3,2)
  for i,img in enumerate(images):
    scale = float(filterTwoSarcSize) / float(ImgTwoSarcSizes[i])
    resizedImg = cv2.resize(img,None,fx=scale,fy=scale,interpolation=cv2.INTER_CUBIC)
    axarr[i,0].imshow(resizedImg,cmap='gray')
    axarr[i,0].axis('off')
    axarr[i,0].set_title(keys[i]+" Raw")

    # switching color channels due to discrepency between cv2 and matplotlib
    newResult = switchBRChannels(results[i])

    axarr[i,1].imshow(newResult)
    axarr[i,1].axis('off')
    axarr[i,1].set_title(keys[i]+" Marked")
  plt.tight_layout()
  plt.gcf().savefig("fig5_RawAndMarked.png")

def figAnalysis(
      ttFilterName=root+"WTFilter.png",
      ltFilterName=root+"LongFilter.png",
      testImage=root+"MI_D_73_annotation.png",
      ttThresh=0.06 ,
      ltThresh=0.38 ,
      gamma=3.,
      ImgTwoSarcSize=None,
      tag = "valid", # tag to prepend to images 
      writeImage = False):
  print "DC: Function is broken. Fix using new giveMarkedMyocyte function"
  quit()
  results = testMF(
      ttFilterName=ttFilterName,#root+"WTFilter.png",
      ltFilterName=ltFilterName,#root+"LongFilter.png",
      testImage=testImage,#root+"MI_D_73_annotation.png",
      ttThresh=ttThresh,#0.06 ,
      ltThresh=ltThresh,#0.38 ,
      gamma=gamma,
      ImgTwoSarcSize=ImgTwoSarcSize,
      writeImage=writeImage)        

  stackedHits = results.stackedHits

  # report responses for each channel   
  dimensions = np.shape(stackedHits.WT)
  area = float(dimensions[0] * dimensions[1])
  results.ttContent = np.sum(stackedHits.WT)/ area
  #print results.ttContent
  results.ltContent = np.sum(stackedHits.Long) / area
  #print results.ltContent
  results.lossContent = 0.


  # write bar plot of feature content  
  fig, ax = plt.subplots()
  ax.set_title("% content") 
  values= np.array([ results.ttContent, results.ltContent, results.lossContent])
  values = values/np.max( values ) 
  ind = np.arange(np.shape ( values )[0])
  width = 1.0
  color = ["blue","green","red"] 
  marks = ["WT","LT","Loss"] 
  rects = ax.bar(ind, values, width,color=color)   
  ax.set_xticks(ind+width)
  ax.set_xticklabels( marks ,rotation=90 )
  if writeImage:
    plt.gcf().savefig(tag+"_content.png") 

def testMFExp():
    dataSet = Myocyte() 
    iters = [-20, -15, -10, -5, 0, 5, 10, 15, 20]
		
    filter1_filter1Test, filter2_filter1Test = bD.TestFilters(
      dataSet.filter1TestName, # testData
      dataSet.filter1Name,                # fusedfilter Name
      dataSet.filter2Name,              # bulkFilter name
      #testData = dataSet.filter1TestData,
      #subsection=dataSet.filter1TestRegion, #[200,400,200,500],   # subsection of testData
      filter1Thresh = dataSet.filter1Thresh,
      filter2Thresh = dataSet.filter2Thresh,
      sigma_n = dataSet.sigma_n,
      #iters = [optimalAngleFused],
      iters=iters,
      useFilterInv=False,
      penaltyscale=0.,
      colorHitsOutName="filter1Marked_%f_%f.png"%(dataSet.filter2Thresh,dataSet.filter1Thresh),
      display=display
    )

def giveMarkedMyocyte(
      ttFilterName=root+"WTFilter.png",
      ltFilterName=root+"LongFilter.png",
      lossFilterName=root+"LossFilter.png",
      wtPunishFilterName=root+"WTPunishmentFilter.png",
      testImage=root+"MI_D_73_annotation.png",
      ImgTwoSarcSize=None,
      tag = "default_",
      writeImage = False,
      iters=[-20,-15,-10,-5,0,5,10,15,20]):
  
  img = util.ReadImg(testImage,renorm=True)

  # WT filtering
  WTparams = optimizer.ParamDict(typeDict='WT')
  WTparams['covarianceMatrix'] = np.ones_like(img)
  ttFilter = util.ReadImg(ttFilterName, renorm=True)
  WTresults, _ = bD.TestFilters(testData = img,
                           filter1Data = ttFilter,
                           filter2Data = None, 
                           filter1Thresh = WTparams['snrThresh'],
                           iters = iters,
                           single = True,
                           paramDict = WTparams)
  WTstackedHits = WTresults.stackedHits
  #plt.figure()
  #plt.imshow(WTstackedHits)
  #plt.colorbar()
  #plt.show()
  #quit()

  # LT filtering
  LTparams = optimizer.ParamDict(typeDict='LT')
  LTFilter = util.ReadImg(ltFilterName, renorm = True)
  LTresults, _ = bD.TestFilters(testData = img,
                           filter1Data = LTFilter,
                           filter2Data = None, 
                           filter1Thresh = LTparams['snrThresh'],
                           iters = iters,
                           single = True,
                           paramDict = LTparams)
  LTstackedHits = LTresults.stackedHits

  # Loss filtering
  Lossparams = optimizer.ParamDict(typeDict='Loss')
  LossFilter = util.ReadImg(lossFilterName, renorm = True)
  Lossiters = [0] # don't need rotations for loss filtering
  Lossresults, _ = bD.TestFilters(testData = img,
                           filter1Data = LossFilter,
                           filter2Data = None, 
                           filter1Thresh = Lossparams['snrThresh'],
                           iters = Lossiters,
                           single = True,
                           paramDict = Lossparams)
  LossstackedHits = Lossresults.stackedHits
 
  # BE SURE TO REMOVE ME!!
  print "WARNING: nan returned from stackedHits, so 'circumventing this'"
  cI = util.ReadImg(testImage,cvtColor=False)

  # Marking superthreshold hits for loss filter
  LossstackedHits[LossstackedHits != 0] = 255
  LossstackedHits = np.asarray(LossstackedHits, dtype='uint8')

  # applying a loss mask to attenuate false positives from WT and Longitudinal filter
  WTstackedHits[LossstackedHits == 255] = 0
  LTstackedHits[LossstackedHits == 255] = 0

  # marking superthreshold hits for longitudinal filter
  LTstackedHits[LTstackedHits != 0] = 255
  LTstackedHits = np.asarray(LTstackedHits, dtype='uint8')

  # masking WT response with LT mask so there is no overlap in the markings
  WTstackedHits[LTstackedHits == 255] = 0

  # marking superthreshold hits for WT filter
  WTstackedHits[WTstackedHits != 0] = 255
  WTstackedHits = np.asarray(WTstackedHits, dtype='uint8')

  # apply preprocessed masks
  wtMasked = ReadResizeApplyMask(WTstackedHits,testImage,ImgTwoSarcSize,
                                 filterTwoSarcSize=ImgTwoSarcSize)
  ltMasked = ReadResizeApplyMask(LTstackedHits,testImage,ImgTwoSarcSize,
                                 filterTwoSarcSize=ImgTwoSarcSize)
  lossMasked = ReadResizeApplyMask(LossstackedHits,testImage,ImgTwoSarcSize,
                                   filterTwoSarcSize=ImgTwoSarcSize)

  # color corrresponding channels
  WTcopy = cI[:,:,0]
  WTcopy[wtMasked == 255] = 255

  LTcopy = cI[:,:,1]
  LTcopy[ltMasked == 255] = 255

  Losscopy = cI[:,:,2]
  Losscopy[lossMasked == 255] = 255
  
  if writeImage:
    # write outputs	  
    cv2.imwrite(tag+"_output.png",cI)       

  return cI 

##
## Defines dataset for myocyte (MI) 
##
def Myocyte():
    # where to look for images
    root = "myoimages/"
    # name of data used for testing algorithm 
    filter1TestName = root + 'MI_D_73_annotation.png'
    # version of filter1TestName marked 'white' where you expect to get hits for filter1
    # or by marking 'positive' channel 
    filter1PositiveTest = root+"MI_D_73_annotation_channels.png"

    dataSet = optimizer.DataSet(
        root = root,
        filter1TestName = filter1TestName,
        filter1TestRegion = None,
        filter1PositiveTest = filter1PositiveTest,
        filter1PositiveChannel= 0,  # blue, WT 
        filter1Label = "TT",
        filter1Name = root+'WTFilter.png',          
        filter1Thresh=0.06, 
        
        filter2TestName = filter1TestName,
        filter2TestRegion = None,
        filter2PositiveTest = filter1PositiveTest,
        filter2PositiveChannel= 1,  # green, longi
        filter2Label = "LT",
        filter2Name = root+'LongFilter.png',        
        filter2Thresh=0.38 
    )

    return dataSet


def rocData(): 
  dataSet = Myocyte() 


  ## Testing TT first 
  dataSet.filter1PositiveChannel= 0
  dataSet.filter1Label = "TT"
  dataSet.filter1Name = root+'WTFilter.png'
  optimizer.SetupTests(dataSet)
  paramDict = optimizer.ParamDict(typeDict='WT')
  paramDict['covarianceMatrix'] = np.ones_like(dataSet.filter1TestData)
  paramDict['mfPunishment'] = util.ReadImg(root+"WTPunishmentFilter.png",renorm=True)
  
  optimizer.GenFigROC_TruePos_FalsePos(
        dataSet,
        paramDict,
        filter1Label = dataSet.filter1Label,
        f1ts = np.linspace(3,15,12),
        #display=True
      )

  ## Testing LT now
  dataSet.filter1PositiveChannel=1
  dataSet.filter1Label = "LT"
  dataSet.filter1Name = root+'LongFilter.png'
  optimizer.SetupTests(dataSet)
  paramDict = optimizer.ParamDict(typeDict='LT')  
  
  optimizer.GenFigROC_TruePos_FalsePos(
        dataSet,
        paramDict,
        filter1Label = dataSet.filter1Label,
        f1ts = np.linspace(15,25,3),
        #display=True
      )

  ## Testing Loss
  print "NOTE: This is using the new loss filter. Rename filter and fix function call."
  dataSet.filter1PositiveChannel = 2
  dataSet.filter1Label = "Loss"
  dataSet.filter1Name = root+"newLossFilter.png"
  optimizer.SetupTests(dataSet)
  paramDict = optimizer.ParamDict(typeDict='Loss')

  optimizer.GenFigROC_TruePos_FalsePos(
         dataSet,
         paramDict,
         filter1Label = dataSet.filter1Label,
         f1ts = np.linspace(4,15,11),
         #display=True
       )

###
### Function to convert from cv2's color channel convention to matplotlib's
###         
def switchBRChannels(img):
  newImg = img.copy()

  # ensuring to copy so that we don't accidentally alter the original image
  newImg[:,:,0] = img[:,:,2].copy()
  newImg[:,:,2] = img[:,:,0].copy()

  return newImg
  


def ReadResizeApplyMask(img,imgName,ImgTwoSarcSize,filterTwoSarcSize=25):
  # function to apply the image mask before outputting results
  maskName = imgName[:-4]; fileType = imgName[-4:]
  fileName = maskName+'_mask'+fileType
  mask = cv2.imread(fileName)                       
  try:
    maskGray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
  except:
    print "No mask named '"+fileName +"' was found. Circumventing masking."
    return img
  scale = float(filterTwoSarcSize) / float(ImgTwoSarcSize)
  maskResized = cv2.resize(maskGray,None,fx=scale,fy=scale,interpolation=cv2.INTER_CUBIC)
  normed = maskResized.astype('float') / float(np.max(maskResized))
  normed[normed < 1.0] = 0
  dimensions = np.shape(img)
  if len(dimensions) < 3:
    combined = img * normed 
  else:
    combined = img
    for i in range(dimensions[2]):
      combined[:,:,i] = combined[:,:,i] * normed
  return combined

def assessContent(markedImg):
  # pull out channels
  wt = markedImg[:,:,0]
  lt = markedImg[:,:,1]
  loss = markedImg[:,:,2]

  # get rid of everything that isn't a hit (hits are marked as 255)
  wt[wt != 255] = 0
  lt[lt != 255] = 0
  loss[loss != 255] = 0

  # normalize
  wtNormed = wt / np.max(wt)
  ltNormed = lt / np.max(lt)
  lossNormed = loss / np.max(loss)

  # calculate content
  wtContent = np.sum(wtNormed)
  ltContent = np.sum(ltNormed)
  lossContent = np.sum(lossNormed)

  return wtContent, ltContent, lossContent

# function to validate that code has not changed since last commit
def validate(testImage=root+"MI_D_78.png",
             ImgTwoSarcSize=22,
             ):
  # run algorithm
  markedImg = giveMarkedMyocyte(testImage=testImage,ImgTwoSarcSize=ImgTwoSarcSize)

  # calculate wt, lt, and loss content  
  wtContent, ltContent, lossContent = assessContent(markedImg)

  print "WT Content:",wtContent
  print "LT Content:", ltContent
  print "Loss Content:", lossContent
  
  assert(abs(wtContent - 12534) < 1), "WT validation failed."
  assert(abs(ltContent - 25687) < 1), "LT validation failed."
  assert(abs(lossContent - 2198) < 1), "Loss validation failed."
  print "PASSED!"

# A minor validation function to serve as small tests between commits
def minorValidate(testImage=root+"MI_D_73_annotation.png",
                  ImgTwoSarcSize=25, #img is already resized to 25 px
                  iters=[-10,0,10]):

  # run algorithm
  markedImg = giveMarkedMyocyte(testImage=testImage, 
                                ImgTwoSarcSize=ImgTwoSarcSize,iters=iters)

  # assess content
  wtContent, ltContent, lossContent = assessContent(markedImg) 
  
  print "WT Content:",wtContent
  print "Longitudinal Content", ltContent
  print "Loss Content", lossContent

  val = 133 
  assert(abs(wtContent - val) < 1),"%f != %f"%(wtContent, val)       
  val = 25286
  assert(abs(ltContent - val) < 1),"%f != %f"%(ltContent, val) 
  val = 0
  assert(abs(lossContent - val) < 1),"%f != %f"%(lossContent, val)
  print "PASSED!"


###
### Function to test that the optimizer routines that assess positive and negative
### filter scores are working correctly.
###
def scoreTest():
  dataSet = Myocyte() 

  ## Testing TT first 
  dataSet.filter1PositiveChannel=0
  dataSet.filter1Label = "TT"
  dataSet.filter1Name = root+'WTFilter.png'
  optimizer.SetupTests(dataSet)
  dataSet.filter1Thresh = 5.5

  paramDict = optimizer.ParamDict(typeDict='WT')
  paramDict['covarianceMatrix'] = np.ones_like(dataSet.filter1TestData)

  filter1PS,filter1NS = optimizer.TestParams_Single(
    dataSet,
    paramDict,
    iters=[-20,-15,-10,-5,0,5,10,15,20],
    display=False)  
    #display=True)  

  print filter1PS, filter1NS

  val = 0.926816518557
  assert((filter1PS - val) < 1e-3), "Filter 1 Positive Score failed"
  val = 0.342082872458
  assert((filter1NS - val) < 1e-3), "Filter 1 Negative Score failed"
  print "PASSED"


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
    if(arg=="-validate"):
      print "Consider developing a more robust behavior test"
      validate()
      quit()

    # this function will generate input data for the current fig #3 in the paper 
    if(arg=="-fig3"):               
      1
      # DC: what is my WT test image (top panel)  
      # DC: call to generate middle panel 
      # DC: call to generate bottom panel 

    if(arg=="-fig4"):               
      fig4()
      quit()

    if(arg=="-fig5"):               
      fig5()
      quit()

    if(arg=="-fig6"):               
      # RB: generate detected version of Fig 6
      # PKH: add in scaling plot 
      1
    # generates all figs
    if(arg=="-allFigs"):
      fig3()     
      fig4()     
      fig5()
      fig6()     
      quit()

    if(arg=="-tag"):
      tag = sys.argv[i+1]
   
    if(arg=="-roc"): 
      rocData()
      quit()
	   
    if(arg=="-testexp"):
      testMFExp()
      quit()
    if(arg=="-test"):
      giveMarkedMyocyte(      
        ttFilterName=sys.argv[i+1],
        ltFilterName=sys.argv[i+2],
        testImage=sys.argv[i+3],           
        ttThresh=np.float(sys.argv[i+4]),           
        ltThresh=np.float(sys.argv[i+5]),
        gamma=np.float(sys.argv[i+6]),
	tag = tag,
	writeImage = True)            
      quit()
    if(arg=="-scoretest"):
      scoreTest()             
      quit()
    if(arg=="-minorValidate"):
      minorValidate()
      quit()

  raise RuntimeError("Arguments not understood")
