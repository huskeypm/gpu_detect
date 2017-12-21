"""
Purpose of this program is to optimize the threshold parameters 
to cleanly pick out data features
"""
import cv2
import bankDetect as bD
import numpy as np
import matplotlib.pylab as plt


# fused Pore 
class empty():pass
root = "./testimages/"
sigma_n = 22. # based on Ryan's data 
fusedThresh = 1000.
bulkThresh = 1050. 

#import optimizer as opt

def TestParams(fusedThresh=1000.,bulkThresh=1050.,sigma_n=1.,display=False,useFilterInv=False):
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
                           positiveTest="testimages/fusedMarked.png", 
                           #negativeTest="testimages/bulkMarked.png", 
                           mode="nohits",
                           display=display)

      bulkPS, bulkNS = Score(bulkPore_bulkTEM.stackedHits,bulkPore_fusedTEM.stackedHits,
                            positiveTest="testimages/bulkMarked.png",
                            #negativeTest="testimages/fusedMarked.png",
                            mode="nohits",
                            display=display)   
    
    ## 
    print fusedThresh,bulkThresh,fusedPS,bulkNS,bulkPS,fusedNS
    return fusedPS,bulkNS,bulkPS,fusedNS

#!/usr/bin/env python
import sys
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
    # calls 'doit' with the next argument following the argument '-validation'
    if(arg=="-optimize"):
      Assess()
      quit()
    if(arg=="-optimize2"):
    # coarse/fine
      ft = np.concatenate([np.linspace(0.5,0.7,7),np.linspace(0.7,0.95,15)   ])
      bt = np.concatenate([np.linspace(0.4,0.55,7),np.linspace(0.55,0.65,15)   ])
      Assess(
        fusedThreshes = ft,
        bulkThreshes = bt,
        sigma_n = 1.,
        useFilterInv=True,  
        display=False
      )
      quit()
  





  raise RuntimeError("Arguments not understood")




