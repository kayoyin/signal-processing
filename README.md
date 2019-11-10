# signal-processing
This repository gathers implementations of machine learning algorithms for signal processing using Python from scratch.

## Dimensionality Reduction
Principal Components Analysis is implemented for dimensionality reduction. The following will run the demo:
```
python dimensionality_reduction.py
```
This performs PCA on the 982 MNIST images of the number 4, and saves a sample reconstruction from reduced images with 2, 16, 64 and 256 principal components.

## Source Separation
Independent Components Analysis and Nonnegative Matrix Factorizationo are implemented for source separation.
The following will run the demo:
```
python source_separation.py
```
This performs ICA and NMF on 1000 MNIST images of the numbers 0, 1, 4, 7 that have been mixed together with varying ratio, and saves the extracted source images.