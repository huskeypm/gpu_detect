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

  figAnalysis(
    testImage=root+"HF_1_annotation.png",
    tag = "HF", 
    writeImage=True) 

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
  DResults = testMF(testImage=DImageName,ImgTwoSarcSize=DTwoSarcSize)
  MResults = testMF(testImage=MImageName,ImgTwoSarcSize=MTwoSarcSize)
  PResults = testMF(testImage=PImageName,ImgTwoSarcSize=PTwoSarcSize)

  results = [DResults, MResults, PResults]
  keys = ['D', 'M', 'P']
  areas = {}

  ttResults = []
  ltResults = []
  lossResults = []

  # report responses for each case
  for i,result in enumerate(results):
    sH = result.stackedHits
    dimensions = np.shape(sH.WT)
    area = float(dimensions[0] * dimensions[1])
    ttContent = np.sum(sH.WT) / area
    ltContent = np.sum(sH.Long) / area
    lossContent = np.sum(sH.loss) / area
    # construct array of areas and norm 
    newAreas = np.array([ttContent, ltContent, lossContent])
    normedAreas = newAreas / np.max(newAreas)
    areas[keys[i]] = normedAreas
    # append to lists
    ttResults.append(normedAreas[0])
    ltResults.append(normedAreas[1])
    lossResults.append(normedAreas[2])

  # generating figure
  #fig, axarr = plt.subplots(3,3)
  width = 0.25
  colors = ["blue","green","red"]
  marks = ["WT","LT","Loss"]
  #plt.rcParams['font.size'] = 6.0
  #plt.rcParams['figure.figsize'] = [16,8]
  #plt.rcParams['figure.dpi'] = 300
  #for i,img in enumerate(images):
  #  axarr[i,0].imshow(img,cmap='gray')
  #  axarr[i,0].set_title(keys[i]+" Raw")
  #  axarr[i,1].imshow(results[i].coloredImg)
  #  axarr[i,1].set_title(keys[i]+" Marked")
    # construct bar plot
  #  axarr[i,2].set_title(keys[i]+" Content")
  #  indices = np.arange(np.shape(areas[keys[i]])[0])
  #  rectangles = axarr[i,2].bar(indices,areas[keys[i]],width,color=colors)
  #  axarr[i,2].set_xticks(indices+width)
  #  axarr[i,2].set_xticklabels(marks, rotation=90)
  #plt.gcf().savefig("fig5.png")

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
    #axarr[i,0].set_title(keys[i]+" Raw")
    # applying masks to each
    masked = ReadResizeApplyMask(results[i].coloredImg,imgNames[i],ImgTwoSarcSizes[i])
    axarr[i,1].imshow(masked)
    axarr[i,1].axis('off')
    #axarr[i,1].set_title(keys[i]+" Marked")
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

def testMF(
      ttFilterName=root+"WTFilter.png",
      ltFilterName=root+"LongFilter.png",
      lossFilterName=root+"LossFilter.png",
      wtPunishFilterName=root+"WTPunishmentFilter.png",
      testImage=root+"MI_D_73_annotation.png",
      ttThresh=0.06 ,
      ltThresh=0.38 ,
      gamma=3.,
      ImgTwoSarcSize=None,
      tag = "default_",
      writeImage = False):

  #results = empty()
  results = Rs.giveStackedHits(testImage, 
		   ttThresh, ltThresh, gamma, ImgTwoSarcSize=ImgTwoSarcSize,
                   WTFilterName=ttFilterName,
                   LongitudinalFilterName=ltFilterName,
                   LossFilterName = lossFilterName,
                   WTPunishFilterName=wtPunishFilterName
                   )
  stackedHits = results.stackedHits

  # BE SURE TO REMOVE ME!!
  print "WARNING: nan returned from stackedHits, so 'circumventing this'"
  cI = results.coloredImg

  wt = np.zeros_like( cI[:,:,0]) 
  wt[ np.where(cI[:,:,2] > 100) ] = 1
  lt = np.zeros_like( wt )
  lt[ np.where(cI[:,:,1] > 100) ] = 1
  loss = np.zeros_like( wt )
  loss[ np.where(cI[:,:,0] > 100) ] = 1

  # apply masks
  wtMasked = ReadResizeApplyMask(wt,testImage,ImgTwoSarcSize)
  ltMasked = ReadResizeApplyMask(lt,testImage,ImgTwoSarcSize)
  lossMasked = ReadResizeApplyMask(loss,testImage,ImgTwoSarcSize)

  # add to containers
  stackedHits.WT = wtMasked 
  stackedHits.Long = ltMasked 
  stackedHits.loss = lossMasked 

  if writeImage:
    # write outputs	  
    cv2.imwrite(tag+"_output.png",results.coloredImg)       


  return results 

##
## Defines dataset for myocyte (MI) 
##
import optimizer
def Myocyte():
  root = "myoimages/"
  filter1TestName = root + 'MI_D_73_annotation.png'
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
  optimizer.SetupTests(dataSet)

  #pass in data like you are doing in your other tests 
  #threshold? 
  optimizer.GenFigROC(
        dataSet,
        filter1Label = dataSet.filter1Label,
        filter2Label = dataSet.filter2Label,
        f1ts = np.linspace(0.05,0.50,9),
        f2ts = np.linspace(0.05,0.30,9),
        penaltyscales = [1.2],
        useFilterInv=True,
      )

def ReadResizeApplyMask(img,imgName,ImgTwoSarcSize,filterTwoSarcSize=25):
  # function to apply the image mask before outputting results
  maskName, fileType = imgName.split('.')
  mask = cv2.imread(maskName+'_mask.'+fileType)
  maskGray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
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

# function to validate that code has not changed since last commit
def validate(testImage=root+"MI_D_78.png",
             ImgTwoSarcSize=22,
             ):
  # run algorithm
  results = testMF(testImage=testImage,ImgTwoSarcSize=ImgTwoSarcSize)

  ## run test
  cI = results.coloredImg
  wt = np.zeros_like( cI[:,:,0])
  wt[ np.where(cI[:,:,2] > 100) ] = 1
  lt = np.zeros_like( wt )
  lt[ np.where(cI[:,:,1] > 100) ] = 1
  loss = np.zeros_like( wt )
  loss[ np.where(cI[:,:,0] > 100) ] = 1
  # apply masks
  wtMasked = ReadResizeApplyMask(wt,testImage,ImgTwoSarcSize)
  ltMasked = ReadResizeApplyMask(lt,testImage,ImgTwoSarcSize)
  lossMasked = ReadResizeApplyMask(loss,testImage,ImgTwoSarcSize)
  # find total content
  wtContent = np.sum(wtMasked)
  ltContent = np.sum(ltMasked)
  lossContent = np.sum(lossMasked)
  # compare to previously tested output
  print "WARNING: Dig into why longitdunial content = 0"
  assert(abs(wtContent - 35.0) < 1)
  assert(abs(ltContent - 0) < 1)
  assert(abs(lossContent - 300.0) < 1)
  #  print wtContent, ltContent, lossContent
  print "PASSED!"


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
      # DC: same as fig 3, but w HF data 
      1
    if(arg=="-fig5"):               
      fig5()

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
      testMF(      
        ttFilterName=sys.argv[i+1],
        ltFilterName=sys.argv[i+2],
        testImage=sys.argv[i+3],           
        ttThresh=np.float(sys.argv[i+4]),           
        ltThresh=np.float(sys.argv[i+5]),
        gamma=np.float(sys.argv[i+6]),
	tag = tag,
	writeImage = True)            
      quit()




  raise RuntimeError("Arguments not understood")




