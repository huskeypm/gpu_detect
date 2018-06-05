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
import os
class empty:pass

#root = "myoimages/"
#root = "/net/share/dfco222/data/TT/LouchData/processed/"
root = "/net/share/dfco222/data/TT/LouchData/processedWithIntelligentThresholding/"

## WT 
def fig3(): 
  #root = "./myoimages/"
  testImage = root+"Sham_M_65_processed.png"
  #twoSarcSize = 21

  rawImg = util.ReadImg(testImage,cvtColor=False)

  iters = [-25,-20, -15, -10, -5, 0, 5, 10, 15, 20,25]
  coloredImg, coloredAngles, angleCounts = giveMarkedMyocyte(testImage=testImage,
                        returnAngles=True,
                        iters=iters,
                        )
  correctColoredAngles = switchBRChannels(coloredAngles)
  correctColoredImg = switchBRChannels(coloredImg)

  ### make bar chart for content
  wtContent, ltContent, lossContent = assessContent(coloredImg,testImage)
  normedContents = [wtContent, ltContent, lossContent]

  ### generating figure
  width = 0.25
  colors = ["blue","green","red"]
  marks = ["WT","LT","Loss"]

  ### make a single bar chart
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
  plt.close()

  ### displaying raw, marked, and marked angle images
  fig, axarr = plt.subplots(3,1)
  axarr[0].imshow(rawImg,cmap='gray')
  axarr[0].axis('off')
  axarr[1].imshow(correctColoredImg)
  axarr[1].axis('off')
  axarr[2].imshow(correctColoredAngles)
  axarr[2].axis('off')
  plt.gcf().savefig("fig3_RawAndMarked.png")
  plt.close()

  ### save histogram of angles
  plt.figure()
  n, bins, patches = plt.hist(angleCounts, len(iters), normed=1, 
                              facecolor='green', alpha=0.5)
  plt.xlabel('Rotation Angle')
  plt.ylabel('Probability')
  plt.gcf().savefig("fig3_histogram.png")
  plt.close()
  

## HF 
def fig4(): 
  ### initial arguments
  #root = "./myoimages/"
  filterTwoSarcSize = 25
  imgName = root + "HF_1_processed.png"
  rawImg = util.ReadImg(imgName)
  markedImg = giveMarkedMyocyte(testImage=imgName)

  ### make bar chart for content
  wtContent, ltContent, lossContent = assessContent(markedImg,imgName)
  normedContents = [wtContent, ltContent, lossContent]

  ### generating figure
  width = 0.25
  colors = ["blue","green","red"]
  marks = ["WT","LT","Loss"]

  ### opting to make a single bar chart
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
  plt.close()
 
  ### constructing actual figure
  fig, axarr = plt.subplots(2,1)
  axarr[0].imshow(rawImg,cmap='gray')
  axarr[0].set_title("HF Raw")
  axarr[0].axis('off') 

  switchedImg = switchBRChannels(markedImg)
  axarr[1].imshow(switchedImg)
  axarr[1].set_title("HF Marked")
  axarr[1].axis('off')
  plt.gcf().savefig("fig4_RawAndMarked.png")
  plt.close()

## MI 
def fig5(): 
  ### update if need be
  #root="./myoimages/"
  filterTwoSarcSize = 25

  ### Distal, Medial, Proximal
  DImageName = root+"MI_D_73_processed.png"
  MImageName = root+"MI_M_45_processed.png"
  PImageName = root+"MI_P_16_processed.png"

  imgNames = [DImageName, MImageName, PImageName]

  ### Read in images for figure
  DImage = util.ReadImg(DImageName)
  MImage = util.ReadImg(MImageName)
  PImage = util.ReadImg(PImageName)
  images = [DImage, MImage, PImage]

  # BE SURE TO UPDATE TESTMF WITH OPTIMIZED PARAMS
  Dimg = giveMarkedMyocyte(testImage=DImageName)
  Mimg = giveMarkedMyocyte(testImage=MImageName)
  Pimg = giveMarkedMyocyte(testImage=PImageName)

  results = [Dimg, Mimg, Pimg]
  keys = ['Distal', 'Medial', 'Proximal']
  areas = {}

  ttResults = []
  ltResults = []
  lossResults = []

  ### report responses for each case
  for i,img in enumerate(results):
    ### assess content based on cell area
    wtContent, ltContent, lossContent = assessContent(img,imgNames[i])
    ### store in lists
    ttResults.append(wtContent)
    ltResults.append(ltContent)
    lossResults.append(lossContent)

  ### generating figure
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
  plt.close()

  ### saving individual marked images
  fig, axarr = plt.subplots(3,2)
  for i,img in enumerate(images):
    axarr[i,0].imshow(img,cmap='gray')
    axarr[i,0].axis('off')
    axarr[i,0].set_title(keys[i]+" Raw")

    ### switch color channels due to discrepency between cv2 and matplotlib conventions
    newResult = switchBRChannels(results[i])

    axarr[i,1].imshow(newResult)
    axarr[i,1].axis('off')
    axarr[i,1].set_title(keys[i]+" Marked")
  plt.tight_layout()
  plt.gcf().savefig("fig5_RawAndMarked.png")
  plt.close()

def fig6():
  ### taking notch filtered image and marking where WT is located
  root = "./myoimages/"
  name = "testingNotchFilter.png"
  
  ## opting to write specific routine since the stacked hits have to be pulled out separately
  ttFilterName = root+"WTFilter.png"
  iters = [-25,20,-15,-10-5,0,5,10,15,20,25]
  returnAngles = False
  img = util.ReadImg(name,renorm=False)
  inputs = empty()
  inputs.imgOrig = img

  nonFilteredImg = util.ReadImg('rotatedTissue.png', renorm=False)

  if np.shape(img) != np.shape(nonFilteredImg):
    raise RuntimeError("The filtered image is not the same dimensions as the non filtered image")

  ### setup WT parameters
  WTparams = optimizer.ParamDict(typeDict='WT')
  WTparams['covarianceMatrix'] = np.ones_like(img)
  #WTparams['mfPunishment'] = util.LoadFilter("./myoimages/WTPunishmentFilter.png")
  ### TODO change this routine to work with new util.LoadFilter function
  WTparams['mfPunishment'] = util.ReadImg("./myoimages/WTPunishmentFilter.png",renorm=True)
  #ttFilter = util.LoadFilter(ttFilterName)
  ttFilter = util.ReadImg(ttFilterName,renorm=True)
  WTparams['useGPU'] = False # for now
  inputs.mfOrig = ttFilter

  ### obtain super threshold filter hits
  WTresults = bD.DetectFilter(inputs,WTparams,iters,returnAngles=returnAngles)  
  WTstackedHits = WTresults.stackedHits

  ### Generate Figure
  plt.figure()
  plt.imshow(WTstackedHits)
  plt.title('stacked hits')
  plt.show()

  # utilize previously written routine to smooth hits and show WT regions
  import tissue
  tissue.DisplayHits(nonFilteredImg, WTstackedHits)
  plt.gcf().savefig("fig6.png",dpi=300)
  plt.close()

###
### Generates the large ROC figure for each of the annotated images
###
def figS1():
  '''
  Routine to generate the necessary ROC figures
  '''

  root = "./myoimages/"

  # images that were hand annotated
  imgNames = {'HF':'HF_1_annotation.png',
              'Control':'Sham_M_65_annotation.png',
              'MI':'MI_D_73_annotation.png'
               }

  # images that have hand annotation marked
  annotatedImgNames = {'HF':'HF_1_annotation_channels.png',
                       'Control':'Sham_M_65_annotation_channels.png',
                       'MI':'MI_D_73_annotation_channels.png'
                       }

  for key,imgName in imgNames.iteritems():
      print imgName
      ### setup dataset
      dataSet = optimizer.DataSet(
                  root = root,
                  filter1TestName = root + imgName,
                  filter1TestRegion=None,
                  filter1PositiveTest = root + annotatedImgNames[key]
                  )

      ### run func that writes scores to hdf5 file
      myocyteROC(dataSet,key,threshes = np.linspace(5,30,15))
  
  ### read data from hdf5 files
  import pandas as pd
  # make big ol' dictionary
  bigData = {}
  bigData['MI'] = {}
  bigData['HF'] = {}
  bigData['Control'] = {}
  ### read in data from each myocyte
  for key,nestDict in bigData.iteritems():
    nestDict['WT'] = pd.read_hdf(key+"_WT.h5",'table')
    nestDict['LT'] = pd.read_hdf(key+"_LT.h5",'table')
    nestDict['Loss'] = pd.read_hdf(key+"_Loss.h5",'table')


  ### Figure generation
  f, axs = plt.subplots(3,2,figsize=[7,12])
  plt.subplots_adjust(wspace=0.5,bottom=0.05,top=0.95,hspace=0.25)
  locDict = {'Control':0,'MI':1,'HF':2}
  for key,loc in locDict.iteritems():
    ### writing detection rate fig
    axs[loc,0].scatter(bigData[key]['WT']['filter1Thresh'], 
                     bigData[key]['WT']['filter1PS'],label='WT',c='b')
    axs[loc,0].scatter(bigData[key]['LT']['filter1Thresh'], 
                     bigData[key]['LT']['filter1PS'],label='LT',c='g')
    axs[loc,0].scatter(bigData[key]['Loss']['filter1Thresh'],
                     bigData[key]['Loss']['filter1PS'],label='Loss',c='r')
    axs[loc,0].set_title(key+" Detection Rate",size=12)
    axs[loc,0].set_xlabel('Threshold')
    axs[loc,0].set_ylabel('Detection Rate')
    axs[loc,0].set_ylim([0,1])
    axs[loc,0].set_xlim(xmin=0)
    #axs[loc,0].legend(prop={'size':8})
  
    ### writing ROC fig
    axs[loc,1].set_title(key+" ROC",size=12)
    axs[loc,1].scatter(bigData[key]['WT']['filter1NS'], 
                       bigData[key]['WT']['filter1PS'],label='WT',c='b')
    axs[loc,1].scatter(bigData[key]['LT']['filter1NS'],    
                       bigData[key]['LT']['filter1PS'],label='LT',c='g')
    axs[loc,1].scatter(bigData[key]['Loss']['filter1NS'],    
                       bigData[key]['Loss']['filter1PS'],label='Loss',c='r')
    ### giving 50% line
    vert = np.linspace(0,1,10)
    axs[loc,1].plot(vert,vert,'k--')

    axs[loc,1].set_xlim([0,1])
    axs[loc,1].set_ylim([0,1])
    axs[loc,1].set_xlabel('False Positive Rate')
    axs[loc,1].set_ylabel('True Positive Rate')

  #plt.show()
  plt.gcf().savefig('figS1.png')
  
  

def analyzeAllMyo():
  #root = "/home/AD/dfco222/Desktop/LouchData/processedImgs_May23/"
  #root = "/net/share/dfco222/data/TT/LouchData/processed/"
  root = "/net/share/dfco222/data/TT/LouchData/processedWithIntelligentThresholding/"

  ### instantiate dicitionary to hold content values
  Sham = {}; MI_D = {}; MI_M = {}; MI_P = {}; HF = {};

  for name in os.listdir(root):
    if "mask" in name:
      continue
    print name
    ### iterate through names and mark the images
    markedMyocyte = giveMarkedMyocyte(testImage=root+name,
                                       tag=name[:-4],
                                       writeImage=True,
                                       returnAngles=False)

    ### assess content
    wtC, ltC, lossC = assessContent(markedMyocyte,imgName=root+name)
    content = np.asarray([wtC, ltC, lossC],dtype=float)
    

    ### store content in respective dictionary
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

  ### use function to construct and write bar charts for each content dictionary
  giveBarChartfromDict(Sham,'Sham')
  giveBarChartfromDict(HF,'HF')
  giveMIBarChart(MI_D,MI_M,MI_P)

def analyzeDirectory(directory):
  '''
  Function to iterate through a directory containing images that have already
  been preprocessed by preprocessing.py
  This directory can contain masks but it is not necessary for this to be the
  case.
  '''

  ### instantiate dictionaries for results
  Sham = {}; HF = {}; MI_D = {}; MI_M = {}; MI_P = {}

  for name in os.listdir(directory):
    if "mask" not in name:
      ### mark myocyte
      markedMyocyte = giveMarkedMyocyte(testImage=root+name,
                                         tag=name[:-4],
                                         writeImage=True,
                                         returnAngles=False)

      ### assess content
      wtC, ltC, lossC = assessContent(markedMyocyte,imgName=root+name)
      content = np.asarray([wtC, ltC, lossC],dtype=float)

      ### store content in respective dictionary
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

  ### Make bar charts for results
  giveBarChartFromDict(Sham,'Sham')
  giveBarChartFromDict(HF,'HF')
  giveMIBarChart(MI_D,MI_M,MI_P)

def analyzeSingleMyo(name,twoSarcSize):
   realName = name+"_processed.png"
   markedMyocyte = giveMarkedMyocyte(testImage=realName,
                                     ImgTwoSarcSize=twoSarcSize,
                                     tag=name,
                                     writeImage=True,
                                     returnAngles=False)
   ### assess content
   wtC, ltC, lossC = assessContent(markedMyocyte)
   content = np.asarray([wtC, ltC, lossC],dtype=float)
   content /= np.max(content)

   ### making into dictionary to utilize bar graph function
   dictionary = {name:content}
   giveBarChartfromDict(dictionary,name)

def giveBarChartfromDict(dictionary,tag):
  ### instantiate lists to contain contents
  wtC = []; ltC = []; lossC = [];
  for name,content in dictionary.iteritems():
    wtC.append(content[0])
    ltC.append(content[1])
    lossC.append(content[2])

  #maxContent = np.max([np.max(wtC), np.max(ltC), np.max(lossC)])

  #wtC = np.divide(wtC,maxContent)
  #ltC = np.divide(ltC,maxContent)
  #lossC = np.divide(lossC, maxContent)

  wtC = np.asarray(wtC)
  ltC = np.asarray(ltC)
  lossC = np.asarray(lossC)

  wtAvg = np.mean(wtC)
  ltAvg = np.mean(ltC)
  lossAvg = np.mean(lossC)

  wtStd = np.std(wtC)
  ltStd = np.std(ltC)
  lossStd = np.std(lossC)

  ### now make a bar chart from this
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
  ax.set_ylim([0,1])
  plt.gcf().savefig(tag+'_BarChart.png')

def giveMIBarChart(MI_D, MI_M, MI_P):
  '''
  Gives combined bar chart for all three proximities to the infarct.
  MI_D, MI_M, and MI_P are all dictionaries with structure:
    dict['file name'] = [wtContent, ltContent, lossContent]
  where the contents are floats
  '''

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

  colors = ["blue","green","red"]
  marks = ["WT", "LT", "Loss"]
  width = 1.0
  N = 11
  indices = np.arange(N)*width + width/4.
  fig,ax = plt.subplots()

  ### plot WT
  rects1 = ax.bar(indices[0], wtAvgs['D'], width, color=colors[0],yerr=wtStds['D'],ecolor='k',label='WT')
  rects2 = ax.bar(indices[1], wtAvgs['M'], width, color=colors[0],yerr=wtStds['M'],ecolor='k',label='WT')
  rects3 = ax.bar(indices[2], wtAvgs['P'], width, color=colors[0],yerr=wtStds['P'],ecolor='k',label='WT')

  ### plot LT
  rects4 = ax.bar(indices[4], ltAvgs['D'], width, color=colors[1],yerr=ltStds['D'],ecolor='k',label='LT')
  rects5 = ax.bar(indices[5], ltAvgs['M'], width, color=colors[1],yerr=ltStds['M'],ecolor='k',label='LT')
  rects6 = ax.bar(indices[6], ltAvgs['P'], width, color=colors[1],yerr=ltStds['P'],ecolor='k',label='LT')

  ### plot Loss
  rects7 = ax.bar(indices[8], lossAvgs['D'], width, color=colors[2],yerr=lossStds['D'],ecolor='k',label='Loss')
  rects8 = ax.bar(indices[9], lossAvgs['M'], width, color=colors[2],yerr=lossStds['M'],ecolor='k',label='Loss')
  rects9 = ax.bar(indices[10],lossAvgs['P'], width, color=colors[2],yerr=lossStds['P'],ecolor='k',label='Loss')

  ax.set_ylabel('Normalized Content')
  ax.legend(handles=[rects1,rects4,rects7])
  newInd = indices + width/2.
  ax.set_xticks(newInd)
  ax.set_xticklabels(['D', 'M','P','','D','M','P','','D','M','P'])
  ax.set_ylim([0,1])
  plt.gcf().savefig('MI_BarChart.png')

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

  ### report responses for each channel   
  dimensions = np.shape(stackedHits.WT)
  area = float(dimensions[0] * dimensions[1])
  results.ttContent = np.sum(stackedHits.WT)/ area
  #print results.ttContent
  results.ltContent = np.sum(stackedHits.Long) / area
  #print results.ltContent
  results.lossContent = 0.


  ### write bar plot of feature content  
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

def markPastedFilters(
      lossMasked, ltMasked, wtMasked, cI,
      lossSize=12, LTx=14, LTy=3, wtSize=14
      ):
  '''
  Given masked stacked hits for the 3 filters and a doctored colored image, 
  function will paste filter sized boxes around the characterized regions
  and return the colored image with filter sized regions colored.

  NOTE: Colored image was read in (not grayscale) and 1 was subtracted from
  the image. This was necessary for the thresholding to work with the painter
  function
  '''
  # exploiting architecture of painter function to mark hits for me
  Lossholder = empty()
  Lossholder.stackedHits = lossMasked
  LTholder = empty()
  LTholder.stackedHits = ltMasked
  WTholder = empty()
  WTholder.stackedHits = wtMasked

  ### we want to mark WT last since that should be the most stringent
  # Opting to mark Loss, then Long, then WT
  halfCellSizeLoss = 16 # should think of how to automate
  labeledLoss = painter.doLabel(Lossholder,dx=halfCellSizeLoss,thresh=254)
  LTx = 14
  LTy = 3
  labeledLT = painter.doLabel(LTholder,dx=LTx,dy=LTy,thresh=254)
  halfCellSizeWT = 19
  labeledWT = painter.doLabel(WTholder,dx=halfCellSizeWT,thresh=254)

  ### perform masking
  WTmask = labeledWT.copy()
  LTmask = labeledLT.copy()
  Lossmask = labeledLoss.copy()

  WTmask[labeledLoss] = False
  WTmask[labeledLT] = False
  LTmask[labeledLoss] = False
  LTmask[WTmask] = False # prevents double marking of WT and LT

  alpha = 1.0
  cI[:,:,2][Lossmask] = int(round(alpha * 255))
  cI[:,:,1][LTmask] = int(round(alpha * 255))
  cI[:,:,0][WTmask] = int(round(alpha * 255))

  return cI


def giveMarkedMyocyte(
      ttFilterName="./myoimages/newSimpleWTFilter.png",
      ltFilterName="./myoimages/newLTfilter.png",
      lossFilterName="./myoimages/LossFilter.png",
      wtPunishFilterName="./myoimages/newSimpleWTPunishmentFilter.png",
      ltPunishFilterName="./myoimages/newLTPunishmentFilter.png",
      testImage="./myoimages/MI_D_73_annotation.png",
      ImgTwoSarcSize=None,
      tag = "default_",
      writeImage = False,
      ttThresh=None,
      ltThresh=None,
      lossThresh=None,
      wtGamma=None,
      ltGamma=None,
      iters=[-25,-20,-15,-10,-5,0,5,10,15,20,25],
      returnAngles=False,
      returnPastedFilter=True,
      useGPU=False
      ):
 
  start = time.time()
   
  ### Read in preprocessed image
  img = util.ReadImg(testImage,renorm=False)

  ### defining inputs to be read by DetectFilter function
  inputs = empty()
  inputs.imgOrig = ReadResizeApplyMask(img,testImage,25,25) # just applies mask

  ### WT filtering
  print "WT Filtering"
  inputs.mfOrig = util.LoadFilter(ttFilterName)
  WTparams = optimizer.ParamDict(typeDict='WT')
  WTparams['covarianceMatrix'] = np.ones_like(img)
  WTparams['mfPunishment'] = util.LoadFilter(wtPunishFilterName)
  WTparams['useGPU'] = useGPU
  if ttThresh != None:
    WTparams['snrThresh'] = ttThresh
  if wtGamma != None:
    WTparams['gamma'] = wtGamma
  WTresults = bD.DetectFilter(inputs,WTparams,iters,returnAngles=returnAngles)  
  WTstackedHits = WTresults.stackedHits

  ### LT filtering
  print "LT filtering"
  inputs.mfOrig = util.LoadFilter("./myoimages/LongitudinalFilter.png")
  LTparams = optimizer.ParamDict(typeDict='LT')
  if ltThresh != None:
    LTparams['snrThresh'] = ltThresh
  if ltGamma != None:
    LTparams['gamma'] = ltGamma
  LTparams['useGPU'] = useGPU
  LTresults = bD.DetectFilter(inputs,LTparams,iters,returnAngles=returnAngles)
  LTstackedHits = LTresults.stackedHits

  ### Loss filtering
  print "Loss filtering"
  inputs.mfOrig = util.LoadFilter(lossFilterName)
  Lossparams = optimizer.ParamDict(typeDict='Loss')
  Lossparams['useGPU'] = useGPU
  Lossiters = [0, 45] # don't need many rotations for loss filtering
  if lossThresh != None:
    Lossparams['snrThresh'] = lossThresh
  Lossresults = bD.DetectFilter(inputs,Lossparams,Lossiters,returnAngles=returnAngles)
  LossstackedHits = Lossresults.stackedHits
 
  ### Read in colored image for marking hits
  cI = util.ReadImg(testImage,cvtColor=False)

  # Must subtract 1 from the image since all hits are marked 255 and orig img is normed to 255
  # have to be careful since uint8 format means that 0 - 1 = 255
  cI[cI == 0] = 1
  cI -= 1

  ### Marking superthreshold hits for loss filter
  LossstackedHits[LossstackedHits != 0] = 255
  LossstackedHits = np.asarray(LossstackedHits, dtype='uint8')

  ### applying a loss mask to attenuate false positives from WT and Longitudinal filter
  WTstackedHits[LossstackedHits == 255] = 0
  LTstackedHits[LossstackedHits == 255] = 0

  ### marking superthreshold hits for longitudinal filter
  LTstackedHits[LTstackedHits != 0] = 255
  LTstackedHits = np.asarray(LTstackedHits, dtype='uint8')

  ### masking WT response with LT mask so there is no overlap in the markings
  WTstackedHits[LTstackedHits == 255] = 0

  ### marking superthreshold hits for WT filter
  WTstackedHits[WTstackedHits != 0] = 255
  WTstackedHits = np.asarray(WTstackedHits, dtype='uint8')

  ### apply preprocessed masks
  wtMasked = ReadResizeApplyMask(WTstackedHits,testImage,ImgTwoSarcSize,
                                 filterTwoSarcSize=ImgTwoSarcSize)
  ltMasked = ReadResizeApplyMask(LTstackedHits,testImage,ImgTwoSarcSize,
                                 filterTwoSarcSize=ImgTwoSarcSize)
  lossMasked = ReadResizeApplyMask(LossstackedHits,testImage,ImgTwoSarcSize,
                                   filterTwoSarcSize=ImgTwoSarcSize)

  if not returnPastedFilter:
    ### create holders for marking channels
    WTcopy = cI[:,:,0]
    LTcopy = cI[:,:,1]
    Losscopy = cI[:,:,2]

    ### color corrresponding channels
    WTcopy[wtMasked == 255] = 255
    LTcopy[ltMasked == 255] = 255
    Losscopy[lossMasked == 255] = 255
    if writeImage:
      ### write output image
      cv2.imwrite(tag+"output.png",cI)

  if returnPastedFilter:
    cI = markPastedFilters(lossMasked, ltMasked, wtMasked, cI)
  
    if writeImage:
      ### write outputs	  
      cv2.imwrite(tag+"_output.png",cI)       

  if returnAngles:
    print "Consider using the previously computed rigorous WT hits as a mask",\
          "and rerun with a simple WT filter or the same filter without punishment.",\
          "Using the punishment filter results in wonky striation angles."
    cImg = util.ReadImg(testImage,cvtColor=False)
    coloredAngles = painter.colorAngles(cImg,WTresults.stackedAngles,iters)
    coloredAnglesMasked = ReadResizeApplyMask(coloredAngles,testImage,
                                              ImgTwoSarcSize,
                                              filterTwoSarcSize=ImgTwoSarcSize)
    ### mask the container holding the angles
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
          ### indicates this is a hit
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

def setupAnnotatedImage(annotatedName, baseImageName):
  '''
  Function to be used in conjunction with Myocyte().
  Uses the markPastedFilters() function to paste filters onto the annotated image.
  This is so we don't have to generate a new annotated image everytime we 
  change filter sizes.
  '''
  ### Read in images
  #baseImage = util.ReadImg(baseImageName,cvtColor=False)
  markedImage = util.ReadImg(annotatedName, cvtColor=False)
  
  ### Preprare base image for markPastedFilters function
  #baseImage[baseImage == 0] = 1
  #baseImage -= 1

  ### Divide up channels of markedImage to represent hits
  wtHits, ltHits = markedImage[:,:,0],markedImage[:,:,1]
  wtHits[wtHits > 0] = 255
  ltHits[ltHits > 0] = 255
  # loss is already adequately marked so we don't want it ran through the routine
  lossHits = np.zeros_like(wtHits)
  coloredImage = markPastedFilters(lossHits,ltHits,wtHits,markedImage)
  # add back in the loss hits
  coloredImage[:,:,2] = markedImage[:,:,2]  

  ### Save image to run with optimizer routines
  newName = annotatedName[:-4]+"_pasted"+annotatedName[-4:]
  cv2.imwrite(newName,coloredImage)

  return newName
##
## Defines dataset for myocyte (MI) 
##
def Myocyte():
    # where to look for images
    root = "myoimages/"

    # name of data used for testing algorithm 
    filter1TestName = root + 'MI_D_73_annotation.png'
    #filter1TestName = root + "MI_D_73_annotated_May30.png"
    # version of filter1TestName marked 'white' where you expect to get hits for filter1
    # or by marking 'positive' channel 
    filter1PositiveTest = root+"MI_D_73_annotation_channels.png"
    #filter1PositiveTest = root + "MI_D_73_annotated_channels_May30.png"
    #filter1PositiveTest = setupAnnotatedImage(filter1PositiveTest,filter1TestName)

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


    # flag to paste filters on the myocyte to smooth out results
    dataSet.pasteFilters = True

    return dataSet


def rocData(): 
  dataSet = Myocyte() 

  # rotation angles
  iters = [-25,-20,-15,-10,-5,0,5,10,15,20,25]

  root = "./myoimages/"

  # flag to turn on the pasting of unit cell on each hit
  dataSet.pasteFilters = True

  ## Testing TT first 
  dataSet.filter1PositiveChannel= 0
  dataSet.filter1Label = "TT"
  dataSet.filter1Name = root+'WTFilter.png'
  optimizer.SetupTests(dataSet,meanFilter=False)
  #optimizer.SetupTests(dataSet,meanFilter=True)
  paramDict = optimizer.ParamDict(typeDict='WT')
  paramDict['covarianceMatrix'] = np.ones_like(dataSet.filter1TestData)
  paramDict['mfPunishment'] = util.LoadFilter(root+"WTPunishmentFilter.png")
  
  optimizer.GenFigROC_TruePos_FalsePos(
        dataSet,
        paramDict,
        filter1Label = dataSet.filter1Label,
  #      f1ts = np.linspace(0.13,0.35,15),
        f1ts = np.linspace(15,30, 10),
        iters=iters,
        )

  ## Testing LT now
  dataSet.filter1PositiveChannel=1
  dataSet.filter1Label = "LT"
  dataSet.filter1Name = root+'LongitudinalFilter.png'
  optimizer.SetupTests(dataSet,meanFilter=True)
  paramDict = optimizer.ParamDict(typeDict='LT')  

  optimizer.GenFigROC_TruePos_FalsePos(
        dataSet,
        paramDict,
        filter1Label = dataSet.filter1Label,
        f1ts = np.linspace(0.1, 0.5, 15),
        iters=iters
      )

  ## Testing Loss
  dataSet.filter1PositiveChannel = 2
  dataSet.filter1Label = "Loss"
  dataSet.filter1Name = root+"LossFilter.png"
  optimizer.SetupTests(dataSet,meanFilter=True)
  paramDict = optimizer.ParamDict(typeDict='Loss')
  lossIters = [0,45]

  optimizer.GenFigROC_TruePos_FalsePos(
         dataSet,
         paramDict,
         filter1Label = dataSet.filter1Label,
         f1ts = np.linspace(0.01,0.2,15),
         iters=lossIters,
       )


###
### Function to calculate data for a full ROC for a given myocyte and return
### scores for each filter at given thresholds
###
def myocyteROC(data, myoName,
               threshes = np.linspace(5,30,10),
               iters=[-25,-20,-15,-10,-5,0,5,10,15,20,25]
               ):
  # TODO: Fix thresholds so < 1
  ### WT
  # setup WT data in class structure
  data.filter1PositiveChannel= 0
  data.filter1Label = "TT"
  data.filter1Name = root+'WTFilter.png'
  optimizer.SetupTests(data)
  WTparams = optimizer.ParamDict(typeDict='WT')
  WTparams['covarianceMatrix'] = np.ones_like(data.filter1TestData)
  WTparams['mfPunishment'] = util.LoadFilter(root+"WTPunishmentFilter.png")

  # write filter performance data for WT into hdf5 file
  optimizer.Assess_Single(data, 
                          WTparams, 
                          filter1Threshes=threshes, 
                          hdf5Name=myoName+"_WT.h5",
                          display=False,
                          iters=iters)
  
  ### LT
  # setup LT data
  data.filter1PositiveChannel=1
  data.filter1Label = "LT"
  #dataSet.filter1Name = root+'LongFilter.png'
  # opting to test H filter now
  #dataSet.filter1Name = root+'newLTfilter.png'
  data.filter1Name = root+'simpleLTfilter.png'
  optimizer.SetupTests(data)
  LTparams = optimizer.ParamDict(typeDict='LT')

  # write filter performance data for LT into hdf5 file
  optimizer.Assess_Single(data, 
                          LTparams, 
                          filter1Threshes=threshes, 
                          hdf5Name=myoName+"_LT.h5",
                          display=False,
                          iters=iters)

  ### Loss  
  # setup Loss data
  data.filter1PositiveChannel = 2
  data.filter1Label = "Loss"
  data.filter1Name = root+"LossFilter.png"
  optimizer.SetupTests(data)
  Lossparams = optimizer.ParamDict(typeDict='Loss')
  LossIters = [0,45]

  # write filter performance data for Loss into hdf5 file
  optimizer.Assess_Single(data, 
                          Lossparams, 
                          filter1Threshes=threshes, 
                          hdf5Name=myoName+"_Loss.h5",
                          display=False,
                          iters=LossIters)


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
  if ImgTwoSarcSize != None:
    scale = float(filterTwoSarcSize) / float(ImgTwoSarcSize)
    maskResized = cv2.resize(maskGray,None,fx=scale,fy=scale,interpolation=cv2.INTER_CUBIC)
  else:
    maskResized = maskGray
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

def assessContent(markedImg,imgName=None):
  # create copy
  imgCopy = markedImg.copy()
  # pull out channels
  wt = imgCopy[:,:,0]
  lt = imgCopy[:,:,1]
  loss = imgCopy[:,:,2]

  # get rid of everything that isn't a hit (hits are marked as 255)
  wt[wt != 255] = 0
  lt[lt != 255] = 0
  loss[loss != 255] = 0

  # normalize
  wtNormed = np.divide(wt, np.max(wt))
  ltNormed = np.divide(lt, np.max(lt))
  lossNormed = np.divide(loss, np.max(loss))

  # calculate content
  wtContent = np.sum(wtNormed)
  ltContent = np.sum(ltNormed)
  lossContent = np.sum(lossNormed)

  if isinstance(imgName, (str)):
    # if imgName is included, we normalize content to cell area
    dummy = np.multiply(np.ones_like(markedImg[:,:,0]), 255)
    mask = ReadResizeApplyMask(dummy,imgName,25,25)
    mask[mask <= 254] = 0
    mask[mask > 0] = 1
    cellArea = np.sum(mask,dtype=float)
    wtContent /= cellArea
    ltContent /= cellArea
    lossContent /= cellArea
    print "WT Content:", wtContent
    print "LT Content:", ltContent
    print "Loss Content:", lossContent
    print "Sum of Content:", wtContent+ltContent+lossContent
    # these should sum to 1 exactly but I'm leaving wiggle room
    assert (wtContent+ltContent+lossContent) < 1.2, ("Something went " 
            +"wrong with the normalization of content to the cell area calculated "
            +"by the mask. Double check the masking routine.") 
  else:
    print "WT Content:", wtContent
    print "LT Content:", ltContent
    print "Loss Content:", lossContent  

  return wtContent, ltContent, lossContent

def minDistanceROC(dataSet,paramDict,param1Range,param2Range,
                   param1="snrThresh",
                   param2="stdDevThresh",
                   FPthresh=0.1,
                   iters=[-25,-20,-15,-10,-5,0,5,10,15,20,25],
                   ):
  '''
  Function that will calculate the minimum distance to the perfect detection point
  (0,1) on a ROC curve and return those parameters.
  '''
  perfectDetection = (0,1)

  distanceStorage = np.ones((len(param1Range),len(param2Range)),dtype=np.float32)
  TruePosStorage = np.ones_like(distanceStorage)
  FalsePosStorage = np.ones_like(distanceStorage)
  for i,p1 in enumerate(param1Range):
    paramDict[param1] = p1
    for j,p2 in enumerate(param2Range):
      paramDict[param2] = p2
      print "Param 1:",p1
      print "Param 2:",p2
      # having to manually assign the thresholds due to structure of TestParams function
      if param1 == "snrThresh":
        dataSet.filter1Thresh = p1
      elif param2 == "snrThresh":
        dataSet.filter1Thresh = p2
      posScore,negScore = optimizer.TestParams_Single(dataSet,paramDict,iters=iters)
      TruePosStorage[i,j] = posScore
      FalsePosStorage[i,j] = negScore
      if negScore < FPthresh:
        distanceFromPerfect = np.sqrt(((perfectDetection[0]-negScore)**2 +\
                                      (perfectDetection[1]-posScore)**2))
        distanceStorage[i,j] = distanceFromPerfect

  idx = np.unravel_index(distanceStorage.argmin(), distanceStorage.shape)
  optP1idx,optP2idx = idx[0],idx[1]
  optimumP1 = param1Range[optP1idx]
  optimumP2 = param2Range[optP2idx]
  optimumTP = TruePosStorage[optP1idx,optP2idx]
  optimumFP = FalsePosStorage[optP1idx,optP2idx]

  print ""
  print 100*"#"
  print "Minimum Distance to Perfect Detection:",distanceStorage.min()
  print "True Postive Rate:",optimumTP
  print "False Positive Rate:",optimumFP
  print "Optimum",param1,"->",optimumP1
  print "Optimum",param2,"->",optimumP2
  print 100*"#"
  print ""
  return optimumP1, optimumP2, distanceStorage

def optimizeWT():
  root = "./myoimages/"
  dataSet = Myocyte()
  dataSet.filter1PositiveChannel= 0
  dataSet.filter1Label = "TT"
  dataSet.filter1Name = root+'WTFilter.png'
  #optimizer.SetupTests(dataSet,meanFilter=True)
  optimizer.SetupTests(dataSet)
  #print dataSet.pasteFilters
  dataSet.pasteFilters = False

  paramDict = optimizer.ParamDict(typeDict='WT')
  paramDict['covarianceMatrix'] = np.ones_like(dataSet.filter1TestData)
  paramDict['mfPunishment'] = util.LoadFilter(root+"WTPunishmentFilter.png") 
  #snrThreshRange = np.linspace(0.01, 0.15, 35)
  #gammaRange = np.linspace(4., 25., 35)
  snrThreshRange = np.linspace(.1, 10, 20)
  gammaRange = np.linspace(0.75, 3, 20)

  optimumSNRthresh, optimumGamma, distToPerfect= minDistanceROC(dataSet,paramDict,
                                                  snrThreshRange,gammaRange,
                                                  param1="snrThresh",
                                                  param2="gamma", FPthresh=1.)

  distToPerfect *= 255
  distToPerfect = np.asarray(distToPerfect, dtype=np.uint8)
  cv2.imwrite("ROC_Optimization_WT.png",distToPerfect)

def optimizeLT():
  root = "./myoimages/"
  dataSet = Myocyte()
  dataSet.filter1PositiveChannel= 1
  dataSet.filter1Label = "LT"
  dataSet.filter1Name = root+'LongitudinalFilter.png'
  #optimizer.SetupTests(dataSet,meanFilter=True)
  optimizer.SetupTests(dataSet)
  #print dataSet.pasteFilters
  dataSet.pasteFilters = False

  paramDict = optimizer.ParamDict(typeDict='LT')
  snrThreshRange = np.linspace(2, 7, 30)
  stdDevThreshRange = np.linspace(0.05, 0.4, 30)

  # maximum rate of false positives
  FPthresh = 0.3

  optimumSNRthresh, optimumGamma, distToPerfect= minDistanceROC(dataSet,paramDict,
                                                  snrThreshRange,stdDevThreshRange,
                                                  param1="snrThresh",
                                                  param2="stdDevThresh",
                                                  FPthresh=FPthresh)

  distToPerfect *= 255
  distToPerfect = np.asarray(distToPerfect, dtype=np.uint8)
  cv2.imwrite("ROC_Optimization_LT.png",distToPerfect)

def optimizeLoss():
  root = "./myoimages/"
  dataSet = Myocyte()
  dataSet.filter1PositiveChannel= 2
  dataSet.filter1Label = "Loss"
  dataSet.filter1Name = root+'LossFilter.png'
  #optimizer.SetupTests(dataSet,meanFilter=True)
  optimizer.SetupTests(dataSet)
  #print dataSet.pasteFilters
  dataSet.pasteFilters = False

  paramDict = optimizer.ParamDict(typeDict='Loss')
  snrThreshRange = np.linspace(1, 10, 20)
  stdDevThreshRange = np.linspace(0.05, 0.5, 20)

  optimumSNRthresh, optimumGamma, distToPerfect= minDistanceROC(dataSet,paramDict,
                                                  snrThreshRange,stdDevThreshRange,
                                                  param1="snrThresh",
                                                  param2="stdDevThresh",
                                                  FPthresh=1.)

  distToPerfect *= 255
  distToPerfect = np.asarray(distToPerfect, dtype=np.uint8)
  cv2.imwrite("ROC_Optimization_Loss.png",distToPerfect)


# function to validate that code has not changed since last commit
def validate(testImage="./myoimages/MI_D_78_processed.png",
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

  assert(abs(wtContent - 66435) < 1), "WT validation failed."
  assert(abs(ltContent - 13669) < 1), "LT validation failed."
  assert(abs(lossContent - 14458) < 1), "Loss validation failed."
  print "PASSED!"

# A minor validation function to serve as small tests between commits
def minorValidate(testImage="./myoimages/MI_D_73_annotation.png",
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

  val = 18722 
  assert(abs(wtContent - val) < 1),"%f != %f"%(wtContent, val)       
  val = 3669
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

    ### Validation Routines
    if(arg=="-validate"):
      print "Consider developing a more robust behavior test"
      validate()
      quit()

    if(arg=="-minorValidate"):
      minorValidate()
      quit()

    if(arg=="-scoretest"):
      scoreTest()             
      quit()
    

    ### Figure Generation Routines

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

    if(arg=="-workflowFig"):
      giveMarkedMyocyte(testImage="./myoimages/MI_D_73_annotation.png",
                        tag="WorkflowFig",
                        returnAngles=True,
                        writeImage=True)

    ### Testing/Optimization Routines
    if(arg=="-roc"): 
      rocData()
      quit()

    if(arg=="-optimizeWT"):
      optimizeWT()
      quit()

    if(arg=="-optimizeLT"):
      optimizeLT()
      quit()

    if(arg=="-optimizeLoss"):
      optimizeLoss()
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

    if(arg=="-testMyocyte"):
      testImage = sys.argv[i+1]
      giveMarkedMyocyte(testImage=testImage,
                        tag="Testing",
                        writeImage=True)
      quit()

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
      quit()

    if(arg=="-analyzeDirectory"):
      analyzeDirectory()
      quit()

    ### Additional Arguments
    if(arg=="-tag"):
      tag = sys.argv[i+1]

    if(arg=="-noPrint"):
      import os,sys
      sys.stdout = open(os.devnull, 'w')

  raise RuntimeError("Arguments not understood")
