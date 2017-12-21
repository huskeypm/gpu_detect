## Todo
- results are 'nan' for validation.py 
-- stackedHITs messed up? 

# gpu_detect

Requires tensorflow cuda (python), which is currently installed on kafka and hesse.

GPU-accelerated matched filter detection.

source config.bash to load tensor-flow libs

tensorflow_fftconv.py - compares performances of tensorflow FFT/matrix multiply/iFFT vs FFTPack

tensorflow_mf.py - matched filtering via tensorflow

commfigs.bash - generates images for paper 

# validation test
python util.py -genWT -genLong -genLoss -genPunishment
python validation.py -validation

# generate paper figs
python validation.py -allFigs 

# normal test
## - generate images 
python util.py -genWT -genLong -genLoss -genPunishment
## - execute images 
python validation.py -tag "MI" -test ./images/WTFilter.png ./images/LongFilter.png ./images/MI_D_73_annotation.png 0.06 0.38 3.

## Silica paper 
python silicaFigs.py -rocfigs 
