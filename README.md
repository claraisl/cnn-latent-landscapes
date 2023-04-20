# cnn-latent-landscapes

This repository contains a package, PersLands, and a set of experiments for analyzing Convolutional Neural Networks using TDA. In particular, Persistence Landscapes.

## Dependencies

- Python3
- numpy
- matplotlib
- ghudi
- scipy
- random
- sklearn

## Code
The main files for computation of persistence landscapes of latent spaces of Convolutional Neural Networks (CNNs) are stored under `PersLands/`.

Experiments performed can be found under `Esperiments/`. On the one hand, `landscapes_*.py` obtains and saves the latent landscapes of all the studied layers of the CNN model. On the other hand, `*_tests.py` loads this data to perform different experiments.

## Datasets
- Vidović, I., Cupec, R., Željko Hocenski: Crop row detection by global energy minimization. Pattern Recognition 55, 68–86 (2016). https://doi.org/10.1016/j.patcog.2016.01.013
- Krizhevsky, A.: Learning multiple layers of features from tiny images. Toronto, ON, Canada (2009). https://www.cs.toronto.edu/~kriz/cifar.html

## CNNs
- U-net mathworks, https://es.mathworks.com/help/vision/ref/unetlayers.html
- Li, H., Kadav, A., Durdanovic, I., Samet, H., Graf, H.P.: Pruning filters for efficient convnets. arXiv preprint arXiv:1608.08710 (2016). https://doi.org/10.48550/arXiv.1608.08710
