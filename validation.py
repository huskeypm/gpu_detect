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
class empty:pass
def doit(
      ttFilterName="",
      ltFilterName="",
      testImage="",
      ttThresh=1.,
      ltThresh=1.,
      gamma=3.):


  #results = empty()
  results = Rs.giveStackedHits(ttFilterName, ttThresh, ltThresh, gamma, WTFilterName=ttFilterName,
                               LongitudinalFilterName=ltFilterName)
  print type(results)
  results.ttContent = 0.
  results.ltContent = 0.
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
  





  raise RuntimeError("Arguments not understood")




