
# gpu_detect
GPU-accelerated matched filter detection.

Requires tensorflow cuda (python), which is currently installed on kafka and hesse.
source config.bash to load tensor-flow libs


## GPU 
tensorflow_fftconv.py - compares performances of tensorflow FFT/matrix multiply/iFFT vs FFTPack

tensorflow_mf.py - matched filtering via tensorflow

commfigs.bash - generates images for paper 

## Todo
see TODO for todo items 

## Myocyte 
Initialize by running
<code>
python util.py -genAllMyo
</code>

### validation test
<code>
python myocyteFigs.py -validation
</code>

#$# generate paper figs
<code>
python myocyteFigs.py -allFigs 
</code>

### execute images 
<code>
python myocyteFigs.py -tag "MI" -test ./myoimages/WTFilter.png ./myoimages/LongFilter.png ./myoimages/MI_D_73_annotation.png 0.06 0.38 3.
</code>

### Roc optimization
- Need to define 'dataSet' object (see inside myocyteFigs)
- Annotated and original images must be same dimensions 
<code>
 python myocyteFigs.py -roc
</code>


## PNP project
<code>
python silicaFigs.py -rocfigs 
</code>

To generate roc data:
<code>
python figs.py -rocfigs # generate roc figures (after running optimizer.py)
python optimer.py -optimize3
python figs.py -paperfigs # generate figures for paper
-- May be broken, but see sparkdetection repo for working version 
</code>

## Other notes
optimizer.py was copied from spark detection repo, which was design for silica films 
before commiting code, be sure to run unittest.bash first 


