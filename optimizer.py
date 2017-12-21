"""
Purpose of this program is to optimize the threshold parameters 
to cleanly pick out data features

"""
import matplotlib
matplotlib.use('Agg')
import cv2
import sys
import bankDetect as bD
import numpy as np
import matplotlib.pylab as plt

class empty():pass

##
## dataset for param optimziation 
## This class is intended to contain all the necessary information to optmize detection parameters. 
## They will generally be specific to each type of data set. The definition below defaults to 
## parameters used for a silica imaging project, but they should be updated as appropriate for new
## data sets 
## 
root = "pnpimages/"
class DataSet:
  def __init__(self, 
    root = None,
    filter1TestName = root + 'clahe_Best.jpg',
    filter1TestRegion = [340,440,400,500],
    filter1PositiveTest = root+"fusedMarked.png",
    filter2PositiveTest = root+"bulkMarked.png",
    filter2TestName = root + 'clahe_Best.jpg',
    filter2TestRegion = [250,350,50,150],
    filter1Name = root+'fusedCellTEM.png',
    filter1Thresh=1000.,
    filter2Name = root+'bulkCellTEM.png',
    filter2Thresh=1050.
    ): 

    self.root = root 
    self.filter1TestName =filter1TestName #  root + 'clahe_Best.jpg'
    self.filter1TestRegion =filter1TestRegion #  [340,440,400,500]
    self.filter1PositiveTest =filter1PositiveTest #  root+"fusedMarked.png"
    self.filter2PositiveTest =filter2PositiveTest #  root+"bulkMarked.png"
    self.filter2TestName =filter2TestName #  root + 'clahe_Best.jpg'
    self.filter2TestRegion =filter2TestRegion #  [250,350,50,150]
    self.filter1Name =filter1Name #  root+'fusedCellTEM.png'
    self.filter1Thresh=filter1Thresh #h1000.
    self.filter2Name =filter2Name #  root+'bulkCellTEM.png'
    self.filter2Thresh=filter2Thresh #h1050.

##
## This function essentially measures overlap between detected regions and hand-annotated regions
## Positive hits are generated when correctly detected/annotated regions are aligned
##
def ScoreOverlap(positiveHits,negativeHits,
          positiveTest,               
          mode="default", # negative hits are assessed by 'negativeHits' within positive Hits region
                          # negative hits are penalized throughout entire image 
          display=True):
    # read in 'truth image' 
    truthMarked = cv2.imread(positiveTest)
    truthMarked=cv2.cvtColor(truthMarked, cv2.COLOR_BGR2GRAY)
    truthMarked= np.array(truthMarked> 0, dtype=np.float)
    #imshow(fusedMarked)

    # positive hits 
    positiveMasked = np.array(positiveHits > 0, dtype=np.float)
    if display:
      plt.figure()    
      plt.subplot(1,3,1)
      plt.imshow(positiveMasked)
      plt.subplot(1,3,2)
      plt.imshow(truthMarked)
      plt.subplot(1,3,3)
      composite = 2.*truthMarked + positiveMasked
      plt.imshow(composite)
      plt.close()
    #plt.imsave("Test.png",composite)
    positiveScoreOverlapImg = truthMarked*positiveMasked
    positiveScoreOverlap = np.sum(positiveScoreOverlapImg)/np.sum(truthMarked)

    # negative hits 
    negativeMasked = np.array(negativeHits > 0, dtype=np.float)
    if display: 
      plt.figure()    
      plt.subplot(1,3,1)
      plt.imshow(negativeMasked)
      plt.subplot(1,3,2)
      plt.imshow(truthMarked)
      plt.subplot(1,3,3)
      composite = 2.*truthMarked + negativeMasked
      plt.imshow(composite)
      plt.close()
    negativeScoreOverlapImg = truthMarked*negativeMasked

    if mode=="default": 
      negativeScoreOverlap = np.sum(negativeScoreOverlapImg)/np.sum(truthMarked)
    elif mode=="nohits":
      dims = np.shape(negativeScoreOverlapImg)
      negativeScoreOverlap = np.sum(negativeScoreOverlapImg)/np.float(np.prod(dims))

    return positiveScoreOverlap, negativeScoreOverlap


## 
##  Returns true positive/false positive rates for two filters, given respective true positive 
##  annotated data 
## 
def TestParams(
    dataSet,
    display=False):
   
    ### Filter1 (was fusedPore) 
    #dataSet.iters = [30], # focus on best angle for fused pore data
    ## Test both filters on filter1 test data 
    optimalAngleFused = 30
    filter1_filter1Test, filter2_filter1Test = bD.TestFilters(
      dataSet.filter1TestName, # testData
      dataSet.filter1Name,                # fusedfilter Name
      dataSet.filter2Name,              # bulkFilter name
      subsection=dataSet.filter1TestRegion, #[200,400,200,500],   # subsection of testData
      filter1Thresh = dataSet.filter1Thresh,
      filter2Thresh = dataSet.filter2Thresh,
      sigma_n = dataSet.sigma_n,
      #iters = [optimalAngleFused],
      useFilterInv=dataSet.useFilterInv,
      scale=dataSet.scale,
      colorHitsOutName="filter1Marked_%f_%f.png"%(dataSet.filter2Thresh,dataSet.filter1Thresh),
      display=display
    )        

    ### Filter2 (was bulkPore) 
    #daImg = cv2.imread(dataSet.name)
    #cut = daImg[dataSet.subsection[0]:dataSet.subsection[1],dataSet.subsection[2]:dataSet.subsection[3]]
    #imshow(cut)

    ## Test both filters on filter2 test data 
    optimalAngleBulk = 5.
    filter1_filter2Test, filter2_filter2Test = bD.TestFilters(
      dataSet.filter2TestName, # testData
      dataSet.filter1Name,                # fusedfilter Name
      dataSet.filter2Name,              # bulkFilter name
      subsection=dataSet.filter2TestRegion, #[200,400,200,500],   # subsection of testData
      filter1Thresh = dataSet.filter1Thresh,
      filter2Thresh = dataSet.filter2Thresh,
      sigma_n = dataSet.sigma_n,
      #iters = [optimalAngleFused],
      useFilterInv=dataSet.useFilterInv,
      scale=dataSet.scale,
      colorHitsOutName="filter2Marked_%f_%f.png"%(dataSet.filter2Thresh,dataSet.filter1Thresh),
      display=display
     )        
    
    # This approach assess the number of hits of filter A overlapping with regions marked as 'A' in the test data
    # negatives refer to hits of filter B on marked 'A' regions
    #if 0:   
    #  fusedPS, bulkNS= ScoreOverlap(filter1_filter1Test.stackedHits,filter2_filter1Test.stackedHits,
    #                       root+"fusedMarked.png", 
    #                       mode="nohits",
    #                       display=display)
#
#      bulkPS, fusedNS = ScoreOverlap(filter2_filter2Test.stackedHits,filter1_filter2Test.stackedHits,
#                            root+"bulkMarked.png",
#                            mode="nohits",
#                            display=display)   
    # This approach assess filter A hits in marked regions of A, penalizes filter A hits in marked regions 
    # of test set B  
    if 1: 
      filter1PS, filter1NS= ScoreOverlap(filter1_filter1Test.stackedHits,filter1_filter2Test.stackedHits,
                           positiveTest=dataSet.filter1PositiveTest,
                           #negativeTest="testimages/bulkMarked.png", 
                           mode="nohits",
                           display=display)

      filter2PS, filter2NS = ScoreOverlap(filter2_filter2Test.stackedHits,filter2_filter1Test.stackedHits,
                            positiveTest=dataSet.filter2PositiveTest,
                            #negativeTest="testimages/fusedMarked.png",
                            mode="nohits",
                            display=display)   
    
    ## 
    print dataSet.filter1Thresh,dataSet.filter2Thresh,filter1PS,filter2NS,filter2PS,filter1NS
    return filter1PS,filter2NS,filter2PS,filter1NS

##
## Plots data (as read from pandas dataframe) 
##
def AnalyzePerformanceData(dfOrig,tag='filter1',label=None,normalize=False,roc=True,scale=None,outName=None):
    df = dfOrig
    if scale!=None:
      df=dfOrig[dfOrig.scale==scale]
    

    #plt.figure()
    threshID=tag+'Thresh'
    result = df.sort_values(by=[threshID], ascending=[1])

    if roc:
      f,(ax1,ax2) = plt.subplots(1,2)     
    else: 
      f,(ax1) = plt.subplots(1,1)     
    if label==None:
      title = threshID+" threshold"
    else: 
      title = label+" threshold"

    if scale!=None:
      title+=" scale %3.1f"%scale 
    ax1.set_title(title)  
    if normalize:
      maxNS = np.max( df[tag+'NS'].values ) 
      dfNS=df[tag+'NS']/maxNS
      maxPS = np.max( df[tag+'PS'].values ) 
      dfPS=df[tag+'PS']/maxPS
    else:
      maxNS = 1; maxPS=1.
      dfNS=df[tag+'NS']
      dfPS=df[tag+'PS']

    ax1.scatter(df[threshID], dfPS,label=label+"/positive",c='b')
    ax1.scatter(df[threshID], dfNS,label=label+"/negative",c='r')
    ax1.set_ylabel("Normalized rate") 
    ax1.set_xlabel("threshold") 
    ax1.set_ylim([0,1]) 
    ax1.set_xlim(xmin=0)
    ax1.legend(loc=0)
    
    if roc==False:
      return 


    ax=ax2   
    ax.set_title("ROC")
    ax.scatter(dfNS,dfPS)
    ax.set_ylim([0,1])
    ax.set_xlim(xmin=0)

    i =  np.int(0.45*np.shape(result)[0])
    numbers = np.arange( np.shape(result)[0])
    numbers = numbers[::50]
    #numbers = [i]
   
    for i in numbers:
        #print i
        thresh= result[threshID].values[i]
        ax.scatter(result[tag+'NS'].values[i]/maxNS,result[tag+'PS'].values[i]/maxPS,c="r")
        loc = (result[tag+'NS'].values[i]/maxNS,-0.1+result[tag+'PS'].values[i]/maxPS)
        ax.annotate("%4.2f"%thresh, loc)
    ax.set_ylabel("True positive rate (Normalized)") 
    ax.set_xlabel("False positive rate (Normalized)") 
    plt.tight_layout()
    if outName:
      plt.gcf().savefig(outName,dpi=300)


##
## Iterates over parameter compbinations to find optimal 
## ROC data 
## 
## dataSet - a dataset object specific to case you're optimizing (see definition) 
## 
import pandas as pd
def Assess(
  dataSet,
  filter1Threshes = np.linspace(800,1100,10), 
  filter2Threshes = np.linspace(800,1100,10), 
  scales=[1.2],  
  hdf5Name = "optimizer.h5",
  sigma_n = 1.,
  useFilterInv=False,
  display=False
  ):
  
  # create blank dataframe
  df = pd.DataFrame(columns = ['filter1Thresh','filter2Thresh','filter1PS','filter2NS','filter2PS','filter1NS'])
  
  # iterate of thresholds
  for i,filter1Thresh in enumerate(filter1Threshes):
    for j,filter2Thresh in enumerate(filter2Threshes):
      for k,scale      in enumerate(scales):       
        dataSet.filter1Thresh=filter1Thresh
        dataSet.filter2Thresh=filter2Thresh
        dataSet.sigma_n = sigma_n
        dataSet.scale = scale 
        dataSet.useFilterInv = useFilterInv
        filter1PS,filter2NS,filter2PS,filter1NS = TestParams(
          dataSet,
          display=display)

        raw_data =  {\
         'filter1Thresh': dataSet.filter1Thresh,
         'filter2Thresh': dataSet.filter2Thresh,
         'scale': dataSet.scale,                
         'filter1PS': filter1PS,
         'filter2NS': filter2NS,
         'filter2PS': filter2PS,
         'filter1NS': filter1NS}
        #print raw_data
        dfi = pd.DataFrame(raw_data,index=[0])#columns = ['fusedThresh','bulkThresh','fusedPS','bulkNS','bulkPS','fusedNS'])
        df=df.append(dfi)

  # store in hdf5 file
  print "Printing " , hdf5Name 
  df.to_hdf(hdf5Name,'table', append=False)
  
  return df,hdf5Name     


##
## Generates ROC data 
##
def GenFigROC(
  dataSet,
  loadOnly=False,
  useFilterInv=True,
  filter1Label = "fused",
  filter2Label = "bulk",
  bt = np.linspace(0.05,0.50,10),
  ft = np.linspace(0.05,0.30,10),
  scales = [1.2],# tried optimizing, but performance seemed to decline quickly far from 1.2 nspace(1.0,1.5,6)  
  hdf5Name = "optimizeinvscale.h5"
  ):

  ##
  ## perform trials using parameter ranges 
  ##
  if loadOnly:
    print "Reading ", hdf5Name 
  else:
    Assess(
        dataSet,
        filter1Threshes = ft,
        filter2Threshes = bt,
        scales = scales,
        sigma_n = 1.,
        useFilterInv=True,
        hdf5Name = hdf5Name,
        display=False
      )


  ##
  ## Now analyze and make ROC plots 
  ## 
  import pandas as pd
  df = pd.read_hdf(hdf5Name,'table') 
  tag = 'filter1'
  AnalyzePerformanceData(df,tag=tag,label=filter1Label,    
    normalize=True, roc=True,outName=tag+ "ROC.png")
  tag = 'filter2'
  AnalyzePerformanceData(df,tag=tag,   label=filter2Label, 
    normalize=True,roc=True,outName=tag+"ROC.png")



  
  


#!/usr/bin/env python
##################################
#
# Revisions
#       10.08.10 inception
#
##################################


#
# Message printed when program run without arguments 
#
def helpmsg():
  scriptName= sys.argv[0]
  msg="""
Purpose: 
 
Usage:
"""
  msg+="  %s -optimize" % (scriptName)
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
    if(arg=="-optimize3"):
      dataSet = DataSet()
    # coarse/fine
      #ft = np.concatenate([np.linspace(0.5,0.7,7),np.linspace(0.7,0.95,15)   ])
      #bt = np.concatenate([np.linspace(0.4,0.55,7),np.linspace(0.55,0.65,15)   ])
      bt = np.linspace(0.05,0.50,10)  
      ft = np.linspace(0.05,0.30,10) 
      scales = [1.2]  # tried optimizing, but performance seemed to decline quickly far from 1.2 nspace(1.0,1.5,6)  
      Assess(
        dataSet,
        filter1Threshes = ft,
        filter2Threshes = bt,
        hdf5Name = "optimizeinvscale.h5",
        display=False
      )
      quit()
    if(arg=="-optimizeLight"):
      dataSet = DataSet()
      GenFigROC(
        dataSet,
        bt = np.linspace(0.05,0.50,3),   
        ft = np.linspace(0.05,0.30,3),   
        scales = [1.2],
        useFilterInv=True,   
      ) 
      quit()
  





  raise RuntimeError("Arguments not understood")




