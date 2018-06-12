!!!! Working to direct everything through detect.py right now, which should be extremely general (down to cmd line operation for arbitrary data) 
python tissue.py -test1
need to figure out what's different by detect.py vs origin implementation in tissue.py

NOTE: Before running algorithm on user provided images, be sure to run through
<code>
python preprocessing.py -preprocess "IMGNAME" "FILTERTWOSARCSIZE"
</code>
Where IMGNAME is the path/name of your image and FILTERTWOSARCSIZE 
is the default two sarcomere size for the filters used. Default filter size is 25 pixels.

# Package Dependencies and Installation Pages
- OpenCV (cv2)
- imutils
- Numpy
- Matplotlib
- PyYAML
- Scipy
  - Scipy.fftpack
  - Scipy.signal
- Pandas
- Pygame
- Python Image Library (PIL)
- SciKit Learn
  - sklearn.decomposition
- TensorFlow



# Upon Pulling a Clean Repo
Initialize the repo by running the following commands:

1. Run "python util.py -genAllMyo" to generate all necessary filters

2. Run "python preprocessing.py -preprocessAll" to process all of the included myocyte images

# Preprocesing User Supplied Images
To preprocess a directory containing user supplied images, run:

<code>
python preprocessing.py -preprocessDirectory PATH_TO_DIRECTORY
</code>

# MASTER SCRIPT 
## detect.py 
To validate:
python detect.py -validation 

To run from command line with user-provided images:
python detect.py -simple myoimages/Sham_11.png myoimages/WTFilter.png 0.01

To run with yaml: (follow ex.yml for an example) 
python detect.py -simpleYaml ex.yml


## gpu_detect
GPU-accelerated matched filter detection.

Requires tensorflow cuda (python), which is currently installed on kafka and hesse.
source config.bash to load tensor-flow libs


# GPU 
tensorflow_fftconv.py - compares performances of tensorflow FFT/matrix multiply/iFFT vs FFTPack

tensorflow_mf.py - matched filtering via tensorflow

commfigs.bash - generates images for paper 

# Todo
see TODO for todo items 

# Miscellaneous 

### validation test
<code>
python myocyteFigs.py -validate
</code>

### Generate Paper Figures
To preprocess images:
<code>
python preprocessing.py -preprocessAll
</code>
To generate the paper figures:
<code>
python myocyteFigs.py -allFigs 
</code>

### Minor validation test
<code>
python myocyteFigs.py -minorValidate
</code>
Meant to serve as a rapid validation test between commits
Note: Currently broken

### execute images 
<font color=red> Deprecated </color>
<code>
python myocyteFigs.py -tag "MI" -test ./myoimages/WTFilter.png ./myoimages/LongFilter.png ./myoimages/MI_D_73_annotation.png 0.06 0.38 3.
</code>

### ROC optimization
- Need to define 'dataSet' object (see inside myocyteFigs)
- Annotated and original images must be same dimensions 
<code>
python myocyteFigs.py -roc
</code>


## PNP project
<code>
python silicaFigs.py -rocfigs 
-- under development, new to reoptimize
</code>

## Misc. ROC? 
<font color=red> Need to update </color>
To generate roc data:
<code>
python optimizer.py -optimize
python figs.py -rocfigs # generate roc figures (after running optimizer.py)
python figs.py -paperfigs # generate figures for paper
-- May be broken, but see sparkdetection repo for working version 
</code>

## Other notes
optimizer.py was copied from spark detection repo, which was design for silica films 
before commiting code, be sure to run unittest.bash first 



