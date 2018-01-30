

# pass in image 
# passin one or more filters
# pass in a custom param dict, if needed
# return snr results 
# plot those results for each channel  
import matplotlib.pylab as plt 

def LoadImages():
  print "Call me when command-line png files are given"
  1

import display_util as du
import matchedFilter as mf 

def DisplayHits(img,threshed):
        # smooth out image to make it easier to visualize hits 
        daround=np.ones([40,40])
        sadf=mF.matchedFilter(threshed,daround,parsevals=False,demean=False)

        # merge two fields 
        du.StackGrayRedAlpha(img,sadf)


class empty:pass   
import optimizer 
def docalc(img,
           mf,
           lobemf=None,
           #corrThresh=0.,
           #s=1.,
           paramDict = optimizer.ParamDict(),
           name="corr.png"):



    ## Store info 
    inputs=empty()
    inputs.imgOrig = img
    inputs.mfOrig  = mf
    inputs.lobemf = lobemf


    import bankDetect as bD
    results = bD.DetectFilter(inputs,paramDict,iters=[0])
    result = results.correlated[0]
    #corr = np.asarray(results.correlated[0],dtype=float) # or
    results.threshed = results.stackedHits
    DisplayHits(case.subregion,results.threshed)
    plt.gcf().savefig(fileName,dpi=300)



