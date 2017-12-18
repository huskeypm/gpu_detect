# gpu_detect

Requires tensorflow cuda (python), which is currently installed on kafka and hesse.

GPU-accelerated matched filter detection.

source config.bash to load tensor-flow libs

tensorflow_fftconv.py - compares performances of tensorflow FFT/matrix multiply/iFFT vs FFTPack

tensorflow_mf.py - matched filtering via tensorflow

commfigs.bash - generates images for paper 

# normal test
python validation.py images/ttFilterName.png images/ttFilterName.png images/testImage.png 1. 2.

