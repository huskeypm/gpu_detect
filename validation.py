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
class empty:pass
def doit(
      ttFilterName="",
      ltFilterName="",
      testImage="",
      ttThresh=1.,
      ltThresh=1.,
      gamma=3.):


  #results = empty()
  results = Rs.giveStackedHits(testImage, ttThresh, ltThresh, gamma, WTFilterName=ttFilterName,
                               LongitudinalFilterName=ltFilterName)
  stackedHits = results.stackedHits
  # following is pseudocode, essentially
  dimensions = np.shape(stackedHits.WT)
  area = float(dimensions[0] * dimensions[1])
  results.ttContent = stackedHits.WT / area
  results.ltContent = stackedHits.Long / area
  results.lossContent = 0.
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
    # will return a single marked image 
    if(arg=="-validation"):

      doit(      
        ttFilterName=sys.argv[i+1],
        ltFilterName=sys.argv[i+2],
        testImage=sys.argv[i+3],           
        ttThresh=sys.argv[i+4],           
        ltThresh=sys.argv[i+5],
        gamma=sys.argv[i+6])           
      quit()
    elif(i>2):
      raise RuntimeError("Arguments not understood")




