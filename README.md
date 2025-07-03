# Seismic_Ideas
Some ideas for seismic signal processing

## some useful Madagascar function

### 一、process

1. [ortho](https://www.ahay.org/RSF/sfortho.html) Orthogonolize signal and noise


### 二、plot

1. [similarity](https://www.ahay.org/RSF/sfsimilarity.html) Local similarity measure between two datasets

## Local_Orthgonalization

1. Chenyk
2. xxxx


## Hilbert transform

1. [adaptive Subtraction of Post-Stack Surface  Multiples Using the Pseudo-Seismic-Data-Based  Convolutional Neural Network](https://ieeexplore.ieee.org/document/10545581/) we can use 2 channel(data, hilbert(data)) for network


```python
from scipy.signal import hilbert
z = hilbert(data, axis=0) # time axis
real_part = np.real(z) # equal data
imag_part = np.imag(z) # hilbert domain data
```

## Plane-Wave Destruction
