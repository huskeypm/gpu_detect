"""
Packages routines used to determine if correlation response
constitutes a detection
"""

import numpy as np 
import matchedFilter as mf

# This script determines detections by integrating the correlation response
# over a small area, then dividing that by the response of a 'lobe' filter 
# need to write this in paper, if it works 
def lobeDetect(
  inputs,    # data sets, filters etc 
  paramDict  # dictionary of parameters needed for detection
  ):
  img = inputs.img # raw (preprocessed image) 
  

