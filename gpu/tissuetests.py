import numpy as np

imgName = "/home/AD/pmke226/DataLocker/cardiac/Sachse/171127_tissue/tissue.tif"


# If we take PSD of image, can we filter out low freq myocyte border info? (or is it even necessary if we apply sidelobe)

# In[3]:

import cv2


# In[4]:

img = cv2.imread(imgName)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# WT-like region (far from MI)
# x=4mm, y=0
# grab 200 um region

# In[5]:

fov = np.array([3916.62,4093.31]) # um (from image caption in imagej. NOTE: need to reverse spatial dimensions to correspond to the way cv2 loads image)
dim = np.shape(gray)

px_per_um = dim/fov
#print px_per_um

def conv_fiji(x_um,y_um): # in um
    y_px = int(x_um * px_per_um[1])
    x_px = int(y_um * px_per_um[0])
    return x_px,y_px

def get_fiji(gray, loc_um,d_um):
    loc= conv_fiji(loc_um[0],loc_um[1])
    d = conv_fiji(d_um[0],d_um[1])
    subregion = gray[loc[0]:(loc[0]+d[0]),loc[1]:(loc[1]+d[1])]
    return subregion

# want 200x200 uM subsection in WT region
loc_um = [3923,0] # need to define from upper-left corner of ROI in fiji
extent_um = [200,200]
loc= conv_fiji(loc_um[0],loc_um[1])
extent = conv_fiji(extent_um[0],extent_um[1])

# ### Create MF based on TT structure
# - doing this by hand, since the stacking is non-trivial
# - by manual measurement, z-lines are typically about 3.5 um long, 2-3px wide and spaced by 2.1 um and rotated by 22 degrees from verticle

# In[49]:

vert = int(3.5 * px_per_um[0])
horz = int(2.1 * px_per_um[1])
w = 2 # px
rot = 30.
mf = np.zeros([vert,horz])
mf[:,0:w]=1.
mf[:,(horz-w):horz]=1.

#imshow(mf,cmap='gray')

import imutils
mfr = imutils.rotate_bound(mf,-rot)

# need this to be square for GPU (probably buried in a cv2 function somewhere)
print "This is a hack for now"
def makeSquare(mfr):
    dims = np.shape(mfr)
    maxD = np.max(dims)
    square =np.zeros([maxD,maxD])
    square[:,1:(dims[1]+1)] = mfr  # not robust, will work for a specific case
    return square

mfr = makeSquare(mfr)


# ### Create lobe filter
# - The point of this filter is to penalize 'side lobe' responses from the MF, which would typically arise from signals not resembling the MF. 
# - This is an ad hoc approach that is best informed from running the MF first, assessing the correlation plane, and discounting the corr accordingly
# 
# Further down in the notebook I determined that the 'correct' signal generates corr. responses that are about 4px wide, while 4px to either side of that response should be 'side lobes'
# 

# In[53]:

# test with embedded signal
vert = int(3.5 * px_per_um[0])
sigW = 4  # px
lobeW = 4 # px
lobemf = np.ones([vert,lobeW + sigW + lobeW])
lobemf[:,lobeW:(lobeW+sigW)]=0.

#imshow(mf,cmap='gray')

import imutils
lobemfr = imutils.rotate_bound(lobemf,-rot)
lobemfr = makeSquare(lobemfr)

# ### Embed signal in image as a sanity check

# In[21]:

# test with embedded signal
def embed(img,mf):
    plt.figure()
    s= np.max(img)*0.5
    imgEmb = np.copy(img)
    dimr = np.shape(mf)
    imgEmb[0:dimr[0],0:dimr[1]] = mf*s + imgEmb[0:dimr[0],0:dimr[1]]
    imshow(imgEmb)
    return imgEmb

# ### Tensor flow stuff

# In[80]:

import tensorflow as tf
import tensorflow_mf as tmf


# In[94]:
## First test 
if 0: 
  subregion = get_fiji(gray,loc_um,extent_um)
  corrTFCPU = tmf.MF(subregion,mfr,useGPU=False)
  corrTFGPU = tmf.MF(subregion,mfr,useGPU=True)
  quit()


# In[93]:

subregionLarge = get_fiji(gray,[2000,2000],[1000,1000])


# In[107]:

print np.shape(gray)
print np.shape(subregionLarge)


# In[ ]:
cpuTest = False
if cpuTest:
  corrTFCPU = tmf.MF(subregionLarge,mfr,useGPU=False) 
corrTFGPU = tmf.MF(subregionLarge,mfr,useGPU=True)
#corrTFGPU = tmf.MF(gray,mfr,useGPU=True)
#



def renorm(img):
  return img*255/np.max(img)

cv2.imwrite("raw.png", renorm(subregionLarge))
cv2.imwrite("corr.png", renorm(corrTFGPU))




# In[ ]:



