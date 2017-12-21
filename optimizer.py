"""

Purpose of this program is to optimize the threshold parameters 
to cleanly pick out data features
"""
import cv2
import sys
import bankDetect as bD
import numpy as np
import matplotlib.pylab as plt

# fused Pore 
class empty():pass

##
## PNP specific data 
##
root = "pnpimages/"
#root = "testimages/"
sigma_n = 22. # based on Ryan's data 
fusedThresh = 1000.
bulkThresh = 1050. 
def TestParamsNEW(fusedThresh=1000.,bulkThresh=1050.,scale=1.2,sigma_n=1.,display=False,useFilterInv=False):
    ### Fused pore
    testCase = empty()
    testCase.name = root + 'clahe_Best.jpg'
    testCase.subsection=[340,440,400,500]
    #daImg = cv2.imread(testCase.name)
    #cut = daImg[testCase.subsection[0]:testCase.subsection[1],testCase.subsection[2]:testCase.subsection[3]]
    #imshow(cut)

    testCase.filter1 = root+'fusedCellTEM.png' 
    testCase.threshFilter1 = fusedThresh
    testCase.filter2 = root+'bulkCellTEM.png' 
    testCase.threshFilter2 = bulkThresh
    testCase.sigma_n= sigma_n, # focus on best angle for fused pore data 
    testCase.iters = [30], # focus on best angle for fused pore data 
    testCase.label = None

    fusedPore_fusedTEM, bulkPore_fusedTEM = bD.TestFilters(
      testCase.name, # testData
      testCase.filter1,         # fusedfilter Name
      testCase.filter2,       # bulkFilter name
      subsection=testCase.subsection, #[200,400,200,500],   # subsection of testData
      threshFilter1 = testCase.filter1,  
      threshFilter2 = testCase.filter2,
      label = testCase.label,
      sigma_n = testCase.sigma_n,
      iters = testCase.iters,
      useFilterInv=useFilterInv,
      display=display
    )        

    ### Bulk pore
    testCase = empty()
    testCase.name = root+'clahe_Best.jpg'
    testCase.subsection=[250,350,50,150]
    #daImg = cv2.imread(testCase.name)
    #cut = daImg[testCase.subsection[0]:testCase.subsection[1],testCase.subsection[2]:testCase.subsection[3]]
    #imshow(cut)
    testCase.filter1 = root+'fusedCellTEM.png' 
    testCase.threshFilter1 = fusedThresh
    testCase.filter2 = root+'bulkCellTEM.png' 
    testCase.threshFilter2 = bulkThresh
    testCase.sigma_n= sigma_n, 
    testCase.iters = [5], # focus on best angle for bulk pore data 
    testCase.label = "filters_on_pristine.png"

    fusedPore_bulkTEM, bulkPore_bulkTEM = bD.TestFilters(
      testCase.name,
      testCase.filter1,                # fusedfilter Name
      testCase.filter2,              # bulkFilter name
      subsection=testCase.subsection, #[200,400,200,500],   # subsection of testData
      threshFilter1 = testCase.filter1,  
      threshFilter2 = testCase.filter2,
      label = testCase.label,
      sigma_n = testCase.sigma_n,
      iters = testCase.iters,
      useFilterInv=useFilterInv,
      display=display
     )        
    
    ## This approach assess the number of hits of filter A overlapping with regions marked as 'A' in the test data
    ## negatives refer to hits of filter B on marked 'A' regions
    #if 0:   
    #  fusedPS, bulkNS= Score(fusedPore_fusedTEM.stackedHits,bulkPore_fusedTEM.stackedHits,
    #                       "testimages/fusedMarked.png", 
    #                       mode="nohits",
    #                       display=display)
#
    #  bulkPS, fusedNS = Score(bulkPore_bulkTEM.stackedHits,fusedPore_bulkTEM.stackedHits,
    #                        "testimages/bulkMarked.png",
    #                        mode="nohits",
    #                        display=display)   

    ## This approach assess filter A hits in marked regions of A, penalizes filter A hits in marked regions 
    ## of test set B 
    if 1: 
      fusedPS, fusedNS= Score(fusedPore_fusedTEM.stackedHits,fusedPore_bulkTEM.stackedHits,
                           positiveTest=root+"fusedMarked.png", 
                           #negativeTest="testimages/bulkMarked.png", 
                           mode="nohits",
                           display=display)

      bulkPS, bulkNS = Score(bulkPore_bulkTEM.stackedHits,bulkPore_fusedTEM.stackedHits,
                            positiveTest=root+"bulkMarked.png",
                            #negativeTest="testimages/fusedMarked.png",
                            mode="nohits",
                            display=display)   
    
    ## 
    print fusedThresh,bulkThresh,fusedPS,bulkNS,bulkPS,fusedNS
    return fusedPS,bulkNS,bulkPS,fusedNS


#

##
## This function essentially measures overlap between detected regions and hand-annotated regions
## Positive hits are generated when correctly detected/annotated regions are aligned
##
def Score(positiveHits,negativeHits,
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
    #plt.imsave("Test.png",composite)
    positiveScoreImg = truthMarked*positiveMasked
    positiveScore = np.sum(positiveScoreImg)/np.sum(truthMarked)

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
    negativeScoreImg = truthMarked*negativeMasked

    if mode=="default": 
      negativeScore = np.sum(negativeScoreImg)/np.sum(truthMarked)
    elif mode=="nohits":
      dims = np.shape(negativeScoreImg)
      negativeScore = np.sum(negativeScoreImg)/np.float(np.prod(dims))

    return positiveScore, negativeScore

def TestParams(fusedThresh=1000.,bulkThresh=1050.,scale=1.2,
    sigma_n=1.,display=False,useFilterInv=False):
    ### Fused pore
    testCase = empty()
    testCase.name = root + 'clahe_Best.jpg'
    testCase.subsection=[340,440,400,500]
    #daImg = cv2.imread(testCase.name)
    #cut = daImg[testCase.subsection[0]:testCase.subsection[1],testCase.subsection[2]:testCase.subsection[3]]
    #imshow(cut)

    optimalAngleFused = 30
    fusedPore_fusedTEM, bulkPore_fusedTEM = bD.TestFilters(
      testCase.name, # testData
      root+'fusedCellTEM.png',         # fusedfilter Name
      root+'bulkCellTEM.png',        # bulkFilter name
      subsection=testCase.subsection, #[200,400,200,500],   # subsection of testData
      fusedThresh = fusedThresh,#.25,
      bulkThresh = bulkThresh, #.5,
      #label = "opt.png",
      sigma_n = sigma_n,
      #iters = [optimalAngleFused],
      useFilterInv=useFilterInv,
      scale=scale,
      colorHitsOutName="fusedMarked_%f_%f.png"%(bulkThresh,fusedThresh),
      display=display
    )        

    ### Bulk pore
    testCase = empty()
    testCase.name = root+'clahe_Best.jpg'
    testCase.subsection=[250,350,50,150]
    #daImg = cv2.imread(testCase.name)
    #cut = daImg[testCase.subsection[0]:testCase.subsection[1],testCase.subsection[2]:testCase.subsection[3]]
    #imshow(cut)

    optimalAngleBulk = 5.
    fusedPore_bulkTEM, bulkPore_bulkTEM = bD.TestFilters(
      testCase.name,
      root+'fusedCellTEM.png',         # fusedfilter Name
      root+'bulkCellTEM.png',        # bulkFilter name
      subsection=testCase.subsection, #[200,400,200,500],   # subsection of testData
      fusedThresh = fusedThresh,#80.,
      bulkThresh = bulkThresh,#130.,
      label = "filters_on_pristine.png",
      sigma_n=sigma_n,
      #iters = [optimalAngleBulk],
      useFilterInv=useFilterInv,
      scale=scale,
      colorHitsOutName="bulkMarked_%f_%f.png"%(bulkThresh,fusedThresh),
      display=display
     )        
    
    # This approach assess the number of hits of filter A overlapping with regions marked as 'A' in the test data
    # negatives refer to hits of filter B on marked 'A' regions
    if 0:   
      fusedPS, bulkNS= Score(fusedPore_fusedTEM.stackedHits,bulkPore_fusedTEM.stackedHits,
                           root+"fusedMarked.png", 
                           mode="nohits",
                           display=display)

      bulkPS, fusedNS = Score(bulkPore_bulkTEM.stackedHits,fusedPore_bulkTEM.stackedHits,
                            root+"bulkMarked.png",
                            mode="nohits",
                            display=display)   
    # This approach assess filter A hits in marked regions of A, penalizes filter A hits in marked regions 
    # of test set B 
    if 1: 
      fusedPS, fusedNS= Score(fusedPore_fusedTEM.stackedHits,fusedPore_bulkTEM.stackedHits,
                           positiveTest=root+"fusedMarked.png", 
                           #negativeTest="testimages/bulkMarked.png", 
                           mode="nohits",
                           display=display)

      bulkPS, bulkNS = Score(bulkPore_bulkTEM.stackedHits,bulkPore_fusedTEM.stackedHits,
                            positiveTest=root+"bulkMarked.png",
                            #negativeTest="testimages/fusedMarked.png",
                            mode="nohits",
                            display=display)   
    
    ## 
    print fusedThresh,bulkThresh,fusedPS,bulkNS,bulkPS,fusedNS
    return fusedPS,bulkNS,bulkPS,fusedNS

def AnalyzePerformanceData(dfOrig,tag='bulk',normalize=False,roc=True,scale=None,outName=None):
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
    title = threshID+" threshold"
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

    ax1.scatter(df[threshID], dfPS,label=tag+"/positive",c='b')
    ax1.scatter(df[threshID], dfNS,label=tag+"/negative",c='r')
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

import pandas as pd

#
# Assess output for single rotation 
# 
def Assess(
  fusedThreshes = np.linspace(800,1100,10), 
  bulkThreshes = np.linspace(800,1100,10), 
  scales=[1.2],  
  hdf5Name = "optimizer.h5",
  sigma_n = 1.,
  useFilterInv=False,
  display=False
  ):
  
  # create blank dataframe
  df = pd.DataFrame(columns = ['fusedThresh','bulkThresh','fusedPS','bulkNS','bulkPS','fusedNS'])
  
  # iterate of thresholds
  for i,fusedThresh in enumerate(fusedThreshes):
    for j,bulkThresh in enumerate(bulkThreshes):
      for k,scale      in enumerate(scales):       
        fusedPS,bulkNS,bulkPS,fusedNS = TestParams(
          fusedThresh=fusedThresh,bulkThresh=bulkThresh,
          sigma_n=sigma_n,
          scale=scale,
          useFilterInv=useFilterInv,
          display=display)

        raw_data =  {\
         'fusedThresh': fusedThresh,
         'bulkThresh': bulkThresh,
         'scale': scale,                
         'fusedPS': fusedPS,
         'bulkNS': bulkNS,
         'bulkPS': bulkPS,
         'fusedNS': fusedNS}
        #print raw_data
        dfi = pd.DataFrame(raw_data,index=[0])#columns = ['fusedThresh','bulkThresh','fusedPS','bulkNS','bulkPS','fusedNS'])
        df=df.append(dfi)

  # store in hdf5 file
  print "Printing " , hdf5Name 
  df.to_hdf(hdf5Name,'table', append=False)
  
  return df,hdf5Name     
  
  


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
    # coarse/fine
      #ft = np.concatenate([np.linspace(0.5,0.7,7),np.linspace(0.7,0.95,15)   ])
      #bt = np.concatenate([np.linspace(0.4,0.55,7),np.linspace(0.55,0.65,15)   ])
      bt = np.linspace(0.05,0.50,10)  
      ft = np.linspace(0.05,0.30,10) 
      scales = [1.2]  # tried optimizing, but performance seemed to decline quickly far from 1.2 nspace(1.0,1.5,6)  
      Assess(
        fusedThreshes = ft,
        bulkThreshes = bt,
        scales = scales,
        sigma_n = 1.,
        useFilterInv=True,   
        hdf5Name = "optimizeinvscale.h5",
        display=False
      )
      quit()
    if(arg=="-optimizeLight"):
    # coarse/fine
      #ft = np.concatenate([np.linspace(0.5,0.7,7),np.linspace(0.7,0.95,15)   ])
      #bt = np.concatenate([np.linspace(0.4,0.55,7),np.linspace(0.55,0.65,15)   ])
      bt = np.linspace(0.05,0.50,1)    
      ft = np.linspace(0.05,0.30,1)    
      scales = [1.2]  # tried optimizing, but performance seemed to decline quickly far from 1.2 nspace(1.0,1.5,6)  
      Assess(
        fusedThreshes = ft,
        bulkThreshes = bt,
        scales = scales,
        sigma_n = 1.,
        useFilterInv=True,   
        hdf5Name = "optimizeinvscale.h5",
        display=False
      )
      quit()
  





  raise RuntimeError("Arguments not understood")




