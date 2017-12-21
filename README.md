
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
### validation test
<code>
python util.py -genWT -genLong -genLoss -genPunishment
python myocyteFigs.py -validation
</code>

# generate paper figs
<code>
python myocyteFigs.py -allFigs 
</code>

# normal test
## - generate images 
<code>
python util.py -genWT -genLong -genLoss -genPunishment
</code>
## - execute images 
<code>
python myocyteFigs.py -tag "MI" -test ./images/WTFilter.png ./images/LongFilter.png ./images/MI_D_73_annotation.png 0.06 0.38 3.
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
