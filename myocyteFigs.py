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
import painter
import time
class empty:pass

root = "myoimages/"

## WT 
def fig3(): 
  #testImage = root+"Sham_11_processed.png"
  testImage = root+"Sham_M_65_processed.png"
  #testImage = root+"Sham_11.png"
  twoSarcSize = 21

  rawImg = util.ReadImg(testImage,cvtColor=False)

  iters = [-25,-20, -15, -10, -5, 0, 5, 10, 15, 20,25]
  coloredImg, coloredAngles, angleCounts = giveMarkedMyocyte(testImage=testImage,
                        ImgTwoSarcSize=twoSarcSize,
                        returnAngles=True,
                        #returnAngles=False,
                        iters=iters,
                        #returnPastedFilter=True,
                        #writeImage=True)
                        )
  #plt.figure()
  #plt.imshow(coloredAngles)
  #plt.show()
  #quit()
  correctColoredAngles = switchBRChannels(coloredAngles)
  correctColoredImg = switchBRChannels(coloredImg)

  # make bar chart for content
  wtContent, ltContent, lossContent = assessContent(coloredImg)
  contents = np.asarray([wtContent, ltContent, lossContent],dtype='float')
  dims = np.shape(coloredImg[:,:,0])
  area = float(dims[0]) * float(dims[1])
  contents = np.divide(contents, area)
  normedContents = contents / np.max(contents)

  # generating figure
  width = 0.25
  colors = ["blue","green","red"]
  marks = ["WT","LT","Loss"]

  # make a single bar chart
  N = 1
  indices = np.arange(N) + width
  fig,ax = plt.subplots()
  rects1 = ax.bar(indices, normedContents[0], width, color=colors[0])
  rects2 = ax.bar(indices+width, normedContents[1], width, color=colors[1])
  rects3 = ax.bar(indices+2*width, normedContents[2], width, color=colors[2])
  ax.set_ylabel('Normalized Content')
  ax.legend(marks)
  ax.set_xticks([])
  plt.gcf().savefig('fig3_BarChart.png')

  # displaying raw, marked, and marked angle images
  fig, axarr = plt.subplots(3,1)
  axarr[0].imshow(rawImg,cmap='gray')
  axarr[0].axis('off')
  axarr[1].imshow(correctColoredImg)
  axarr[1].axis('off')
  axarr[2].imshow(correctColoredAngles)
  axarr[2].axis('off')
  plt.gcf().savefig("fig3_RawAndMarked.png")

  # save histogram of angles
  plt.figure()
  n, bins, patches = plt.hist(angleCounts, len(iters), normed=1, 
                              facecolor='green', alpha=0.5)
  plt.xlabel('Rotation Angle')
  plt.ylabel('Probability')
  plt.gcf().savefig("fig3_histogram.png")

  

## HF 
def fig4(): 

  filterTwoSarcSize = 25

  #imgName = root+"HF_1.png"
  imgName = root + "HF_1_processed.png"
  twoSarcSize = 21

  rawImg = util.ReadImg(imgName)

  markedImg = giveMarkedMyocyte(testImage=imgName,ImgTwoSarcSize=twoSarcSize)

  wtContent, ltContent, lossContent = assessContent(markedImg.copy())
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
  print "TwoSarcSize argument is deprecated with new preprocssing function. Remove."
  #DImageName = root+"MI_D_76_processed.png"
  #DTwoSarcSize = 22
  DImageName = root+"MI_D_73_processed.png"
  DTwoSarcSize = 22
  MImageName = root+"MI_M_45_processed.png"
  MTwoSarcSize = 21
  PImageName = root+"MI_P_16_processed.png"
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
  keys = ['Distal', 'Medial', 'Proximal']
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

def fig6():
  ### taking notch filtered image and marking where WT is located
  name = "testingNotchFilter.png"
  #threshed = giveMarkedMyocyte(testImage=name,
  #                  tag="fig6",
  #                  iters=[-5,0,5],
  #                  returnAngles=False,
  #                  writeImage=False,
  #                  useGPU=False #for now
  #                  )
  
  ## opting to write specific routine since the stacked hits have to be pulled out separately
  ttFilterName = root+"WTFilter.png"
  iters = [-25,20,-15,-10-5,0,5,10,15,20,25]
  #iters = [0]
  returnAngles = False
  img = util.ReadImg(name,renorm=False)
  inputs = empty()
  inputs.imgOrig = img

  nonFilteredImg = util.ReadImg('rotatedTissue.png', renorm=False)

  if np.shape(img) != np.shape(nonFilteredImg):
    raise RuntimeError("The filtered image is not the same dimensions as the non filtered image")

  # setup WT parameters
  WTparams = optimizer.ParamDict(typeDict='WT')
  WTparams['covarianceMatrix'] = np.ones_like(img)
  WTparams['mfPunishment'] = util.ReadImg("./myoimages/WTPunishmentFilter.png",renorm=True)
  WTparams['useGPU'] = False # for now
  ttFilter = util.ReadImg(ttFilterName, renorm=True)
  inputs.mfOrig = ttFilter

  # obtain super threshold filter hits
  WTresults = bD.DetectFilter(inputs,WTparams,iters,returnAngles=returnAngles)  
  WTstackedHits = WTresults.stackedHits

  plt.figure()
  plt.imshow(WTstackedHits)
  plt.title('stacked hits')
  plt.show()

  ### potentially experiment with using the pasting filter function to make smoothing better
  ### Currently too many hits for this to work
  # pasting filter sized unit square on image to make smoothing look realistic
  #threshed = np.zeros_like(WTstackedHits)
  #threshed[WTstackedHits > 0] = 255
  #halfFilterSize = 10
  #result = empty()
  #result.stackedHits = threshed
  #smoothed = painter.doLabel(result,dx=halfFilterSize,thresh=254)
  #print type(smoothed)

  #plt.figure()
  #plt.imshow(smoothed)
  #plt.title('smoothed')
  #plt.show()
  #quit()

  # utilize previously written routine to smooth hits and show WT regions
  import tissue
  #tissue.DisplayHits(nonFilteredImg, smoothed)
  tissue.DisplayHits(nonFilteredImg, WTstackedHits)
  plt.gcf().savefig("fig6.png",dpi=300)

def figS1():
  ## routine to generate the necessary ROC figures
  1

  
  
  

def analyzeAllMyo():
  root = "/home/AD/dfco222/Desktop/LouchData/processedImgs/"
  twoSarcSizeDict = {'Sham_P_23':21, 'Sham_M_65':21, 'Sham_D_100':20, 'Sham_23':22,
                      'Sham_11':21, 'MI_P_8':21, 'MI_P_5':21, 'MI_P_16':21, 'MI_M_46':22,
                      'MI_M_45':21, 'MI_M_44':21, 'HF_1':21, 'HF_13':21,'MI_D_78':22,
                      'MI_D_76':21, 'MI_D_73':22,
                      'HF_5':21 
                       }
  # instantiate dicitionary to hold content values
  Sham = {}; MI_D = {}; MI_M = {}; MI_P = {}; HF = {};
  for name,twoSarcSize in twoSarcSizeDict.iteritems():
     print name
     # iterate through names and mark the images
     realName = name+"_processed.png"
     markedMyocyte = giveMarkedMyocyte(testImage=root+realName,
                                       ImgTwoSarcSize=twoSarcSize,
                                       tag=name,
                                       writeImage=True,
                                       returnAngles=False)
     # assess content
     wtC, ltC, lossC = assessContent(markedMyocyte)
     content = np.asarray([wtC, ltC, lossC],dtype=float)
     content /= np.max(content)
     # utilizing mask to calculate cell area and normalize content from there
     #mask = cv2.imread(root+name+"_processed_mask.png")
     #mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
     #scale = 25. / float(twoSarcSize)
     #mask = cv2.resize(mask, None, fx=scale,fy=scale, interpolation=cv2.INTER_CUBIC)	
     #mask[mask != 255] = 0
     #mask = np.asarray(mask, dtype=float)
     #mask /= np.max(mask)
     #cellArea = np.sum(mask)
     #content = np.divide(content, cellArea)

     # above didn't actually work. I think I would need to paste the filter for each pixel hit to 
     # accurately do this

     # store content in respective dictionary
     if 'Sham' in name:
       Sham[name] = content
     elif 'HF' in name:
       HF[name] = content
     elif 'MI' in name:
       if '_D' in name:
         MI_D[name] = content
       elif '_M' in name:
         MI_M[name] = content
       elif '_P' in name:
         MI_P[name] = content

  # use function to construct and write bar charts for each content dictionary
  giveBarChartfromDict(Sham,'Sham')
  giveBarChartfromDict(HF,'HF')
  #giveBarChartfromDict(MI_D,root+'MI_D')
  #giveBarChartfromDict(MI_M,root+'MI_M')
  #giveBarChartfromDict(MI_P,root+'MI_P')
  wtAvgs = {}; wtStds = {}; ltAvgs = {}; ltStds = {}; lossAvgs = {}; lossStds = {};

  DwtC = []; DltC = []; DlossC = [];
  for name, content in MI_D.iteritems():
    wtC = content[0]
    ltC = content[1]
    lossC = content[2]
    DwtC.append(wtC)
    DltC.append(ltC)
    DlossC.append(lossC)
  wtAvgs['D'] = np.mean(DwtC)
  wtStds['D'] = np.std(DwtC)
  ltAvgs['D'] = np.mean(DltC)
  ltStds['D'] = np.std(DltC)
  lossAvgs['D'] = np.mean(DlossC)
  lossStds['D'] = np.std(DlossC)

  MwtC = []; MltC = []; MlossC = [];
  for name, content in MI_M.iteritems():
    wtC = content[0]
    ltC = content[1]
    lossC = content[2]
    MwtC.append(wtC)
    MltC.append(ltC)
    MlossC.append(lossC)
  wtAvgs['M'] = np.mean(MwtC)
  wtStds['M'] = np.std(MwtC)
  ltAvgs['M'] = np.mean(MltC)
  ltStds['M'] = np.std(MltC)
  lossAvgs['M'] = np.mean(MlossC)
  lossStds['M'] = np.std(MlossC)


  PwtC = []; PltC = []; PlossC = [];
  for name, content in MI_P.iteritems():
    wtC = content[0]
    ltC = content[1]
    lossC = content[2]
    PwtC.append(wtC)
    PltC.append(ltC)
    PlossC.append(lossC)
  wtAvgs['P'] = np.mean(PwtC)
  wtStds['P'] = np.std(PwtC)
  ltAvgs['P'] = np.mean(PltC)
  ltStds['P'] = np.std(PltC)
  lossAvgs['P'] = np.mean(PlossC)
  lossStds['P'] = np.std(PlossC)

  # opting to combine MI results
  colors = ["blue","green","red"]
  marks = ["WT", "LT", "Loss"]
  #width = 0.25
  width = 1.0
  N = 11
  indices = np.arange(N)*width + width/4.
  fig,ax = plt.subplots()

  # plot WT
  rects1 = ax.bar(indices[0], wtAvgs['D'], width, color=colors[0],yerr=wtStds['D'],ecolor='k',label='WT')
  rects2 = ax.bar(indices[1], wtAvgs['M'], width, color=colors[0],yerr=wtStds['M'],ecolor='k',label='WT')
  rects3 = ax.bar(indices[2], wtAvgs['P'], width, color=colors[0],yerr=wtStds['P'],ecolor='k',label='WT')

  # plot LT
  rects4 = ax.bar(indices[4], ltAvgs['D'], width, color=colors[1],yerr=ltStds['D'],ecolor='k',label='LT')
  rects5 = ax.bar(indices[5], ltAvgs['M'], width, color=colors[1],yerr=ltStds['M'],ecolor='k',label='LT')
  rects6 = ax.bar(indices[6], ltAvgs['P'], width, color=colors[1],yerr=ltStds['P'],ecolor='k',label='LT')

  # plot Loss
  rects7 = ax.bar(indices[8], lossAvgs['D'], width, color=colors[2],yerr=lossStds['D'],ecolor='k',label='Loss')
  rects8 = ax.bar(indices[9], lossAvgs['M'], width, color=colors[2],yerr=lossStds['M'],ecolor='k',label='Loss')
  rects9 = ax.bar(indices[10],lossAvgs['P'], width, color=colors[2],yerr=lossStds['P'],ecolor='k',label='Loss')

  ax.set_ylabel('Normalized Content')
  ax.legend(handles=[rects1,rects4,rects7])
  newInd = indices + width/2.
  ax.set_xticks(newInd)
  #ax.xaxis.set_tick_params(horizontalalignment='center')
  ax.set_xticklabels(['D', 'M','P','','D','M','P','','D','M','P'])
  plt.gcf().savefig('MI_BarChart.png')

def analyzeSingleMyo(name,twoSarcSize):
   realName = name+"_processed.png"
   markedMyocyte = giveMarkedMyocyte(testImage=realName,
                                     ImgTwoSarcSize=twoSarcSize,
                                     tag=name,
                                     writeImage=True,
                                     returnAngles=False)
   # assess content
   wtC, ltC, lossC = assessContent(markedMyocyte)
   content = np.asarray([wtC, ltC, lossC],dtype=float)
   content /= np.max(content)

   # making into dictionary to utilize bar graph function
   dictionary = {name:content}
   giveBarChartfromDict(dictionary,name)

def giveBarChartfromDict(dictionary,tag):
  # instantiate lists to contain contents
  wtC = []; ltC = []; lossC = [];
  for name,content in dictionary.iteritems():
    wtC.append(content[0])
    ltC.append(content[1])
    lossC.append(content[2])

  maxContent = np.max([np.max(wtC), np.max(ltC), np.max(lossC)])

  wtC = np.divide(wtC,maxContent)
  ltC = np.divide(ltC,maxContent)
  lossC = np.divide(lossC, maxContent)

  wtC = np.asarray(wtC)
  ltC = np.asarray(ltC)
  lossC = np.asarray(lossC)

  wtAvg = np.mean(wtC)
  ltAvg = np.mean(ltC)
  lossAvg = np.mean(lossC)

  wtStd = np.std(wtC)
  ltStd = np.std(ltC)
  lossStd = np.std(lossC)

  # now make a bar chart from this
  colors = ["blue","green","red"]
  marks = ["WT", "LT", "Loss"]
  width = 0.25
  N = 1
  indices = np.arange(N) + width
  fig,ax = plt.subplots()
  rects1 = ax.bar(indices, wtAvg, width, color=colors[0],yerr=wtStd,ecolor='k')
  rects2 = ax.bar(indices+width, ltAvg, width, color=colors[1],yerr=ltStd,ecolor='k')
  rects3 = ax.bar(indices+2*width, lossAvg, width, color=colors[2],yerr=lossStd,ecolor='k')
  ax.set_ylabel('Normalized Content')
  ax.legend(marks)
  ax.set_xticks([])
  plt.gcf().savefig(tag+'_BarChart.png')

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
      ttThresh=None,
      ltThresh=None,
      lossThresh=None,
      gamma=None,
      iters=[-25,-20,-15,-10,-5,0,5,10,15,20,25],
      returnAngles=False,
      returnPastedFilter=True,
      useGPU=False
      ):
 
  start = time.time()
   
  #img = util.ReadImg(testImage,renorm=True)
  # No need to renorm the image since preprocessing should be done already
  img = util.ReadImg(testImage,renorm=False)

  # defining inputs in structure to be read by DetectFilter function
  inputs = empty()
  inputs.imgOrig = img

  # WT filtering
  WTparams = optimizer.ParamDict(typeDict='WT')
  WTparams['covarianceMatrix'] = np.ones_like(img)
  WTparams['mfPunishment'] = util.ReadImg("./myoimages/WTPunishmentFilter.png",renorm=True)
  WTparams['useGPU'] = useGPU
  if ttThresh != None:
    WTparams['snrThresh'] = ttThresh
  if gamma != None:
    WTparams['gamma'] = gamma
  ttFilter = util.ReadImg(ttFilterName, renorm=True)
  ## Attempting new punishment filter routine
  #WTparams['mfPunishmentMax'] = np.sum(WTparams['mfPunishment'])
  #WTparams['snrThresh'] = 74.1
  ##
  #WTparams['gamma'] = 3

  # turns out the above worked terribly. Need to play around with this more.

  inputs.mfOrig = ttFilter
  WTresults = bD.DetectFilter(inputs,WTparams,iters,returnAngles=returnAngles)  
  WTstackedHits = WTresults.stackedHits

  # LT filtering
  LTparams = optimizer.ParamDict(typeDict='LT')
  if ltThresh != None:
    LTparams['snrThresh'] = ltThresh
  LTFilter = util.ReadImg(ltFilterName, renorm = True)
  inputs.mfOrig = LTFilter

  inputs.mfOrig = util.ReadImg(root+'newLTfilter.png', renorm = True)
  LTparams['filterMode'] = 'punishmentFilter'
  LTparams['mfPunishment'] = util.ReadImg(root+"newLTPunishmentFilter.png",renorm=True)
  LTparams['gamma'] = 0.05 
  LTparams['covarianceMatrix'] = np.ones_like(inputs.imgOrig)
  LTparams['snrThresh'] = 12.5
  LTparams['useGPU'] = useGPU

  LTresults = bD.DetectFilter(inputs,LTparams,iters,returnAngles=returnAngles)#,display=True)
  LTstackedHits = LTresults.stackedHits

  # Loss filtering
  Lossparams = optimizer.ParamDict(typeDict='Loss')
  Lossparams['useGPU'] = useGPU
  Lossiters = [0, 45] # don't need many rotations for loss filtering
  LossFilter = util.ReadImg(lossFilterName, renorm = True)
  inputs.mfOrig =  LossFilter
  if lossThresh != None:
    Lossparams['snrThresh'] = lossThresh
  Lossresults = bD.DetectFilter(inputs,Lossparams,Lossiters,returnAngles=returnAngles)
  LossstackedHits = Lossresults.stackedHits
 
  cI = util.ReadImg(testImage,cvtColor=False)

  ### Must subtract 1 from the image since all hits are marked 255 and orig img is normed to 255
  cI -= 1
  ###

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

  if not returnPastedFilter:
    # create holders for marking channels
    WTcopy = cI[:,:,0]
    LTcopy = cI[:,:,1]
    Losscopy = cI[:,:,2]

    # color corrresponding channels
    WTcopy[wtMasked == 255] = 255
    LTcopy[ltMasked == 255] = 255
    Losscopy[lossMasked == 255] = 255
    if writeImage:
      #write output image
      cv2.imwrite(tag+"output.png",cI)

  if returnPastedFilter:
    # exploiting architecture of painter function to mark hits for me
    Lossholder = empty()
    #Lossholder.stackedHits = Losscopy
    Lossholder.stackedHits = lossMasked
    LTholder = empty()
    #LTholder.stackedHits = LTcopy
    LTholder.stackedHits = ltMasked
    WTholder = empty()
    #WTholder.stackedHits = WTcopy
    WTholder.stackedHits = wtMasked

    # we want to mark WT last since that should be the most stringent
    # Opting to mark Loss, then Long, then WT
    halfCellSizeLoss = 5 # should think of how to automate
    labeledLoss = painter.doLabel(Lossholder,dx=halfCellSizeLoss,thresh=254)
    halfCellSizeLT = 10
    labeledLT = painter.doLabel(LTholder,dx=halfCellSizeLT,thresh=254)
    halfCellSizeWT = 10
    labeledWT = painter.doLabel(WTholder,dx=halfCellSizeWT,thresh=254)

    ### perform masking
    WTmask = labeledWT.copy()
    LTmask = labeledLT.copy()
    Lossmask = labeledLoss.copy()

    WTmask[labeledLoss] = False
    #labeledLT[labeledLoss] = False
    WTmask[labeledLT] = False
    LTmask[labeledLoss] = False

    #Lossmask[labeledWT] = False
    LTmask[WTmask] = False

    cI[:,:,2][Lossmask] = 255
    cI[:,:,1][LTmask] = 255
    cI[:,:,0][WTmask] = 255
  
    if writeImage:
      # write outputs	  
      cv2.imwrite(tag+"_output.png",cI)       

  if returnAngles:
    cImg = util.ReadImg(testImage,cvtColor=False)
    coloredAngles = painter.colorAngles(cImg,WTresults.stackedAngles,iters)
    coloredAnglesMasked = ReadResizeApplyMask(coloredAngles,testImage,
                                              ImgTwoSarcSize,
                                              filterTwoSarcSize=ImgTwoSarcSize)
    # mask the container holding the angles
    stackedAngles = np.add(WTresults.stackedAngles, 1)
    stackedAngles = ReadResizeApplyMask(stackedAngles,testImage,
                                            ImgTwoSarcSize,
                                            filterTwoSarcSize=ImgTwoSarcSize)
    stackedAngles = np.asarray(stackedAngles, dtype='int')
    stackedAngles = np.subtract(stackedAngles, 1)
    dims = np.shape(stackedAngles)
    angleCounts = []
    for i in range(dims[0]):
      for j in range(dims[1]):
        rotArg = stackedAngles[i,j]
        if rotArg != -1:
          # indicates this is a hit
          angleCounts.append(iters[rotArg])

    if writeImage:
      cv2.imwrite(tag+"_angles_output.png",coloredAnglesMasked)
    
    end = time.time()
    tElapsed = end - start
    print "Total Elapsed Time: {}s".format(tElapsed)
    return cI, coloredAnglesMasked, angleCounts

  end = time.time()
  tElapsed = end - start
  print "Total Elapsed Time: {}s".format(tElapsed)
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

  # rotation angles
  iters = [-25,-20,-15,-10,-5,0,5,10,15,20,25]


  ## Testing TT first 
  dataSet.filter1PositiveChannel= 0
  dataSet.filter1Label = "TT"
  dataSet.filter1Name = root+'WTFilter.png'
  optimizer.SetupTests(dataSet)
  paramDict = optimizer.ParamDict(typeDict='WT')
  paramDict['covarianceMatrix'] = np.ones_like(dataSet.filter1TestData)
  paramDict['mfPunishment'] = util.ReadImg(root+"WTPunishmentFilter.png",renorm=True)
  
  #optimizer.GenFigROC_TruePos_FalsePos(
  #      dataSet,
  #      paramDict,
  #      filter1Label = dataSet.filter1Label,
  #      f1ts = np.linspace(15,45,15),
  #      iters=iters
        #display=True
  #    )

  ## Testing LT now
  dataSet.filter1PositiveChannel=1
  dataSet.filter1Label = "LT"
  #dataSet.filter1Name = root+'LongFilter.png'
  # opting to test H filter now
  #dataSet.filter1Name = root+'newLTfilter.png'
  dataSet.filter1Name = root+'simpleLTfilter.png'
  optimizer.SetupTests(dataSet)
  paramDict = optimizer.ParamDict(typeDict='LT')  

  paramDict['filterMode'] = 'punishmentFilter'
  #paramDict['mfPunishment'] = util.ReadImg(root+"newLTPunishmentFilter.png",renorm=True)
  #paramDict['gamma'] = 0.05 
  paramDict['covarianceMatrix'] = np.ones_like(dataSet.filter1TestData)

  # new longitudinal simple filtering
  paramDict['mfPunishment'] = util.ReadImg(root+"simpleLTPunishmentfilter.png",renorm=True)
  paramDict['gamma'] = 0.25

  
  optimizer.GenFigROC_TruePos_FalsePos(
        dataSet,
        paramDict,
        filter1Label = dataSet.filter1Label,
        #f1ts = np.linspace(10,25,10),
        #f1ts = np.linspace(18,28,10),
        f1ts = np.linspace(0.1, 10, 15),
        iters=iters
        #display=True
      )

  ## Testing Loss
  dataSet.filter1PositiveChannel = 2
  dataSet.filter1Label = "Loss"
  dataSet.filter1Name = root+"LossFilter.png"
  optimizer.SetupTests(dataSet)
  paramDict = optimizer.ParamDict(typeDict='Loss')
  lossIters = [0,45]

  #optimizer.GenFigROC_TruePos_FalsePos(
  #       dataSet,
  #       paramDict,
  #       filter1Label = dataSet.filter1Label,
  #       f1ts = np.linspace(4,15,11),
  #       iters=lossIters
         #display=True
  #     )

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
             display=False
             ):
  # run algorithm
  markedImg = giveMarkedMyocyte(testImage=testImage,ImgTwoSarcSize=ImgTwoSarcSize)

  if display:
    import matplotlib.pyplot as plt
    plt.figure()
    plt.imshow(markedImg)
    plt.show()

  # calculate wt, lt, and loss content  
  wtContent, ltContent, lossContent = assessContent(markedImg)

  print "WT Content:",wtContent
  print "LT Content:", ltContent
  print "Loss Content:", lossContent
  
  assert(abs(wtContent - 65014) < 1), "WT validation failed."
  assert(abs(ltContent - 76830) < 1), "LT validation failed."
  assert(abs(lossContent - 158317) < 1), "Loss validation failed."
  print "PASSED!"

# A minor validation function to serve as small tests between commits
def minorValidate(testImage=root+"MI_D_73_annotation.png",
                  ImgTwoSarcSize=25, #img is already resized to 25 px
                  iters=[-10,0,10],
                  display=False):

  # run algorithm
  markedImg = giveMarkedMyocyte(testImage=testImage, 
                                ImgTwoSarcSize=ImgTwoSarcSize,iters=iters)
  if display:
    import matplotlib.pyplot as plt
    plt.figure()
    plt.imshow(markedImg)
    plt.show()

  # assess content
  wtContent, ltContent, lossContent = assessContent(markedImg) 
  
  print "WT Content:",wtContent
  print "Longitudinal Content", ltContent
  print "Loss Content", lossContent

  val = 6987 
  assert(abs(wtContent - val) < 1),"%f != %f"%(wtContent, val)       
  val = 15978
  assert(abs(ltContent - val) < 1),"%f != %f"%(ltContent, val) 
  val = 1420
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
    iters=[-25,-20,-15,-10,-5,0,5,10,15,20,25],
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
      fig3()
      quit()

    if(arg=="-fig4"):               
      fig4()
      quit()

    if(arg=="-fig5"):               
      fig5()
      quit()

    if(arg=="-fig6"):               
      fig6()
      quit()
    if(arg=="-figS1"):
      figS1()
      quit()
    # generates all figs
    if(arg=="-allFigs"):
      fig3()     
      fig4()     
      fig5()
      fig6()     
      figS1()
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
        ImgTwoSarcSize=(sys.argv[i+7]),
	tag = tag,
	writeImage = True)            
      quit()
    if(arg=="-scoretest"):
      scoreTest()             
      quit()
    if(arg=="-minorValidate"):
      minorValidate()
      quit()
    if(arg=="-testMyocyte"):
      testImage = sys.argv[i+1]
      giveMarkedMyocyte(testImage=testImage,
                        tag="Testing",
                        writeImage=True)
      quit()
    if(arg=="-workflowFig"):
      giveMarkedMyocyte(testImage="./myoimages/MI_D_73_annotation.png",
                        tag="WorkflowFig",
                        returnAngles=True,
                        writeImage=True)
    if(arg=="-analyzeAllMyo"):
      analyzeAllMyo()
      quit()
    if(arg=="-analyzeSingleMyo"):
      name = sys.argv[i+1]
      twoSarcSize = float(sys.argv[i+2])
      analyzeSingleMyo(name,twoSarcSize)
      quit()
    if(arg=="-testTissue"):
      name = "testingNotchFilter.png"
      giveMarkedMyocyte(testImage=name,
                        tag="TestingNotchedFilter",
                        iters=[-5,0,5],
                        returnAngles=False,
                        writeImage=True,
                        useGPU=True)
    if(arg=="-testPaster"):
      # routine to test the output of function that pastes filter onto myocyte
      # will delete eventually
      iters = [-25,-20, -15, -10, -5, 0, 5, 10, 15, 20,25]

      SHAMtest = root+"Sham_M_65_processed.png"
      SHAMtwoSarcSize = 21
      giveMarkedMyocyte(testImage=SHAMtest,
                        ImgTwoSarcSize=SHAMtwoSarcSize,
                        tag="SHAM_Pasted",
                        iters=iters,
                        returnPastedFilter=True,
                        writeImage=True
                        )

      MItest = root+"MI_D_73_processed.png"
      MItwoSarcSize = 22
      giveMarkedMyocyte(testImage=MItest,
                        ImgTwoSarcSize=MItwoSarcSize,
                        tag="MI_Pasted",
                        iters=iters,
                        returnPastedFilter=True,
                        writeImage=True
                        )

      quit()

  raise RuntimeError("Arguments not understood")
