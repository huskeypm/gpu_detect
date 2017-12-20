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

## WT 
def fig3(): 

  print "DC: need to commit WT data to repo and pass in here" 
  figAnalysis(
    testImage="./images/MI_D_73_annotation.png",
    tag = "MI", 
    writeImage=True) 

  # write angle 
  print "DC: print mesh of angles here" 


## HF 
def fig4(): 

  print "DC: need to commit HF data to repo and pass in here" 
  figAnalysis(
    testImage="./images/MI_D_73_annotation.png",
    tag = "MI", 
    writeImage=True) 

## MI 
def fig5(): 

  print "DC: need to commit prox, med, dist MI data to repo and pass in here" 
  print "DC: combine bar plots into single image?"
  figAnalysis(
    testImage="./images/MI_D_73_annotation.png",
    tag = "MIprox", 
    writeImage=True) 

  figAnalysis(
    testImage="./images/MI_D_73_annotation.png",
    tag = "MImed", 
    writeImage=True) 

  figAnalysis(
    testImage="./images/MI_D_73_annotation.png",
    tag = "MIdist", 
    writeImage=True) 




def figAnalysis(
      ttFilterName="./images/WTFilter.png",
      ltFilterName="./images/LongFilter.png",
      testImage="./images/MI_D_73_annotation.png",
      ttThresh=0.06 ,
      ltThresh=0.38 ,
      gamma=3.,
      tag = "valid", # tag to prepend to images 
      writeImage = False):

  results = testMF(
      ttFilterName=ttFilterName,#"./images/WTFilter.png",
      ltFilterName=ltFilterName,#"./images/LongFilter.png",
      testImage=testImage,#"./images/MI_D_73_annotation.png",
      ttThresh=ttThresh,#0.06 ,
      ltThresh=ltThresh,#0.38 ,
      gamma=gamma)        

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
      ttFilterName="./images/WTFilter.png",
      ltFilterName="./images/LongFilter.png",
      testImage="./images/MI_D_73_annotation.png",
      ttThresh=0.06 ,
      ltThresh=0.38 ,
      gamma=3.,
      tag = "default_",
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

    # generates all figs
    if(arg=="-allFigs"):
      fig3()     
      fig4()     
      fig5()     
      quit()

    if(arg=="-tag"):
      tag = sys.argv[i+1]
	   
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



    elif(i>2):
      raise RuntimeError("Arguments not understood")




