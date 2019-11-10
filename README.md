# signal-processing
This repository gathers implementations of machine learning algorithms for signal processing using Python from scratch.

## Dimensionality Reduction
Principal Components Analysis is implemented for dimensionality reduction. The following will run the demo:
```
python dimensionality_reduction.py
```
This performs PCA on the 982 MNIST images of the number 4, and saves a sample reconstruction from reduced images with 2, 16, 64 and 256 principal components.

Input image: ![Original image](https://github.com/kayoyin/signal-processing/blob/master/four_dataset/four0.jpg)

Reconstruction with 2, 16, 64, 128 principal components: ![2 image](https://github.com/kayoyin/signal-processing/blob/master/output/pca_2.jpg) ![Original image](https://github.com/kayoyin/signal-processing/blob/master/output/pca_16.jpg) ![Original image](https://github.com/kayoyin/signal-processing/blob/master/output/pca_64.jpg) ![Original image](https://github.com/kayoyin/signal-processing/blob/master/output/pca_256.jpg)

## Source Separation
Independent Components Analysis and Nonnegative Matrix Factorizationo are implemented for source separation.
The following will run the demo:
```
python source_separation.py
```
This performs ICA and NMF on 1000 MNIST images of the numbers 0, 1, 4, 7 that have been mixed together with varying ratio, and saves the extracted source images.

Input images: ![Mixed image](https://github.com/kayoyin/signal-processing/blob/master/mixture_dataset(0147)/img0.jpg) ![Mixed image 2](https://github.com/kayoyin/signal-processing/blob/master/mixture_dataset(0147)/img1.jpg) ![Mixed image 3](https://github.com/kayoyin/signal-processing/blob/master/mixture_dataset(0147)/img2.jpg) ![Mixed image 4](https://github.com/kayoyin/signal-processing/blob/master/mixture_dataset(0147)/img3.jpg)

Source separation with ICA: ![ICA 1](https://github.com/kayoyin/signal-processing/blob/master/output/ica_0.jpg) ![Ica 2](https://github.com/kayoyin/signal-processing/blob/master/output/ica_1.jpg) ![ICA 3](https://github.com/kayoyin/signal-processing/blob/master/output/ica_2.jpg) ![Ica 4](https://github.com/kayoyin/signal-processing/blob/master/output/ica_3.jpg)

Source separation with NMF: ![ICA 1](https://github.com/kayoyin/signal-processing/blob/master/output/nmf_0.jpg) ![Ica 2](https://github.com/kayoyin/signal-processing/blob/master/output/nmf_1.jpg) ![ICA 3](https://github.com/kayoyin/signal-processing/blob/master/output/nmf_2.jpg) ![Ica 4](https://github.com/kayoyin/signal-processing/blob/master/output/nmf_3.jpg)
