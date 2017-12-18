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
def doit(fileIn):
  1


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

    # this function will generate input data for the current fig #3 in the paper 
    if(arg=="-fig3"):               
      # DC: what is my WT test image (top panel)  
      # DC: call to generate middle panel 
      # DC: call to generate bottom panel 
      # PKH: I'll provide bar graphs 

    if(arg=="-fig4"):               
      # DC: same as fig 3, but w HF data 

    if(arg=="-fig5"):               
      # DC: same as fig 3, but will want to pass in the three MI tissue images 

    if(arg=="-fig6"):               
      # RB: generate detected version of Fig 6
      # PKH: add in scaling plot 

      # apply TT, LT and loss filters to 
      #doit(arg1)
  





  raise RuntimeError("Arguments not understood")




