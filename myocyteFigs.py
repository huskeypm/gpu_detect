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
class empty:pass

root = "myoimages/"

## WT 
def fig3(): 

  print "DC: need to commit WT data to repo and pass in here" 
  figAnalysis(
    testImage=root+"MI_D_73_annotation.png",
    tag = "MI", 
    writeImage=True) 

  # write angle 
  print "DC: print mesh of angles here" 


## HF 
def fig4(): 

  print "DC: need to commit HF data to repo and pass in here" 
  figAnalysis(
    testImage=root+"MI_D_73_annotation.png",
    tag = "MI", 
    writeImage=True) 

## MI 
def fig5(): 

  print "DC: need to commit prox, med, dist MI data to repo and pass in here" 
  print "DC: combine bar plots into single image?"
  figAnalysis(
    testImage=root+"MI_D_73_annotation.png",
    tag = "MIprox", 
    writeImage=True) 

  figAnalysis(
    testImage=root+"MI_D_73_annotation.png",
    tag = "MImed", 
    writeImage=True) 

  figAnalysis(
    testImage=root+"MI_D_73_annotation.png",
    tag = "MIdist", 
    writeImage=True) 




def figAnalysis(
      ttFilterName=root+"WTFilter.png",
      ltFilterName=root+"LongFilter.png",
      testImage=root+"MI_D_73_annotation.png",
      ttThresh=0.06 ,
      ltThresh=0.38 ,
      gamma=3.,
      tag = "valid", # tag to prepend to images 
      writeImage = False):

  results = testMF(
      ttFilterName=ttFilterName,#root+"WTFilter.png",
      ltFilterName=ltFilterName,#root+"LongFilter.png",
      testImage=testImage,#root+"MI_D_73_annotation.png",
      ttThresh=ttThresh,#0.06 ,
      ltThresh=ltThresh,#0.38 ,
      gamma=gamma,        
      writeImage = writeImage)

  stackedHits = results.stackedHits

  # report responses for each channel   
  dimensions = np.shape(stackedHits.WT)
  area = float(dimensions[0] * dimensions[1])
  results.ttContent = np.sum(stackedHits.WT)/ area
  #print results.ttContent
  results.ltContent = np.sum(stackedHits.Long) / area
  #print results.ltContent
  results.lossContent = 0.


  # write bar plot of feature contnent  
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
  plt.gcf().savefig(tag+"_content.png") 


def testMF(
      ttFilterName=root+"WTFilter.png",
      ltFilterName=root+"LongFilter.png",
      testImage=root+"MI_D_73_annotation.png",
      ttThresh=0.06 ,
      ltThresh=0.38 ,
      gamma=3.,
      tag = "default",
      writeImage = False):

  #results = empty()
  results = Rs.giveStackedHits(testImage, 
    ttThresh, ltThresh, gamma, WTFilterName=ttFilterName,
    LongitudinalFilterName=ltFilterName)
  stackedHits = results.stackedHits

  # BE SURE TO REMOVE ME!!
  print "WARNING: nan returned from stackedHits, so 'circumventing this'"
  cI = results.coloredImg
  wt = np.zeros_like( cI[:,:,0]) 
  wt[ np.where(cI[:,:,2] > 100) ] = 1
  results.ttContent = stackedHits.WT = wt 
  lt = np.zeros_like( cI[:,:,0]) 
  lt[ np.where(cI[:,:,1] > 100) ] = 1   
  stackedHits.Long = lt 

  if writeImage:
    # write putputs	  
    cv2.imwrite(tag+"_output.png",results.coloredImg)       


  return results 

##
## Defines dataset for myocyte (MI) 
##
import optimizer
root = "myoimages/"
def Myocyte(
  filter1TestName = root + 'MI_D_73_annotation.png',
  filter1PositiveTest = root+"MI_D_73_annotation_channels.png",
  filter1Name = root+'WTFilter.png',          
  filter1Thresh=1000.,
  filter2Name = root+'LongFilter.png',        
  filter2Thresh=1050.
  ):

  dataSet = optimizer.DataSet(
    root = root,
    filter1TestName = filter1TestName,
    filter1TestRegion = None,
    filter1PositiveTest = filter1PositiveTest,
    filter1PositiveChannel= 0,  # blue, WT 
    filter1Name = root+'WTFilter.png',          
    filter1Thresh=filter1Thresh,

    filter2TestName = filter1TestName,
    filter2TestRegion = None,
    filter2PositiveTest = filter1PositiveTest,
    filter2PositiveChannel= 1,  # green, longi
    filter2Name = root+'LongFilter.png',        
    filter2Thresh=filter2Thresh,
    )

  return dataSet

def rocData(): 
  dataSet = Myocyte() 
  optimizer.SetupTests(dataSet)

  #pass in data like you are doing in your other tests 
  #threshold? 
  #filter1Thresh = 0.01,
  #  filter2Thresh = 0.0035
  optimizer.GenFigROC(
        dataSet,
        f1ts = np.linspace(0.005,0.03,3),
        f2ts = np.linspace(0.0025,0.0045,3),
        penaltyscales = [1.0],
        useFilterInv=False
      )


# print for debugging 
def test2():
# python myocyteFigs.py -tag "MI" -test ./myoimages/WTFilter.png ./myoimages/LongFilter.png ./myoimages/MI_D_73_annotation.png 0.06 0.38 3.
  # This works; need to check with TestFilters protocol
  if 0: 
    figAnalysis(
      ttFilterName=root+"WTFilter.png",
      ltFilterName=root+"LongFilter.png",
      testImage=root+"MI_D_73_annotation.png",
      ttThresh=0.06 ,
      ltThresh=0.38 ,
      gamma=3.,
      tag = "default",
      writeImage = True)    

  print "REMOVE +test" 
  dataSet = Myocyte(
    filter1TestName = root + 'MI_D_73_annotation.png',
    filter1PositiveTest = root+"MI_D_73_annotation_channels.png",
    filter1Name = root+'WTFilter.png', 
    filter1Thresh = 0.01, 
    filter2Name = root+'LongFilter.png',
    filter2Thresh = 0.0035
    )
  optimizer.SetupTests(dataSet)

  print "WARNING: need to expand angle range" 
  iters = [-30,-20,-10,0,10,20,30]
  import bankDetect as bD
  filter1_filter1Test, filter2_filter1Test = bD.TestFilters(
      dataSet.filter1TestName, # testData
      dataSet.filter1Name,                # fusedfilter Name
      dataSet.filter2Name,              # bulkFilter name
      testData = dataSet.filter1TestData,
      #subsection=dataSet.filter1TestRegion, #[200,400,200,500],   # subsection of testData
      filter1Thresh = dataSet.filter1Thresh,
      filter2Thresh = dataSet.filter2Thresh,
      sigma_n = dataSet.sigma_n,
      iters=iters,
      useFilterInv=False, # dataSet.useFilterInv,
      penaltyscale=0.,# dataSet.penaltyscale,
      colorHitsOutName="filter1Marked_%f_%f.png"%(dataSet.filter2Thresh,dataSet.filter1Thresh),
      display=True     
    )

   
  
##
##
## 
def validate(): 
  dataSet = Myocyte( 
    filter1TestName = root + 'MI_D_73_annotation.png',
    filter1PositiveTest = root+"MI_D_73_annotation_channels.png",
    filter1Name = root+'WTFilter.png', 
    filter1Thresh = 0.01, 
    filter2Name = root+'LongFilter.png',
    filter2Thresh = 0.0035
    )
  optimizer.SetupTests(dataSet)

  optimizer.SetupTests(dataSet) 
  filter1PS,filter2NS,filter2PS,filter1NS = optimizer.TestParams(
    dataSet,
    display=False)    

  # 122217
  # 0.0637741766858 0.00157362602777 0.0378071833648 0.00479955938471
  assert(np.abs(filter1PS-0.0637)<0.001), "Test1 failed"
  assert(np.abs(filter2NS-0.0015)<0.001), "Test2 failed"
  assert(np.abs(filter2PS-0.0378)<0.001), "Test3 failed"
  assert(np.abs(filter1NS-0.0048)<0.001), "Test4 failed"
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
    # will return a single marked image 
    if(arg=="-validation"):
      testmf()     
      print "WARNING: add unit test here!!" 
      quit()

    # this function will generate input data for the current fig #3 in the paper 
    if(arg=="-fig3"):               
      # DC: what is my WT test image (top panel)  
      # DC: call to generate middle panel 
      # DC: call to generate bottom panel 
      # PKH: I'll provide bar graphs 
      fig3()

    if(arg=="-fig4"):               
      # DC: same as fig 3, but w HF data 
      fig4()
      1
    if(arg=="-fig5"):               
      # DC: same as fig 3, but will want to pass in the three MI tissue images 
      fig5()
      1
    if(arg=="-fig6"):               
      # RB: generate detected version of Fig 6
      # PKH: add in scaling plot 
      fig6()
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

    if(arg=="-test2"):
      test2()
      quit()
    if(arg=="-validate"): 
      validate()
      quit()


  raise RuntimeError("Arguments not understood")




