"""This script computes and saves the latent landscapes for U-Nets from the 
activations maps of the layers studied. In addition, it plots the (average) 
latent landscapes of every layer, performs PCA and different experiments.


In our case, we study the ReLU layers, storing the output activation maps in 
'activationCell.mat' from Matlab and loading them in Python. 

We recommend to work with the saved latent landscapes and perform with them 
different experiments using 'unet_tests.py'.

- For single images, it compares them computing the distance between each layer 
landscape, the distance of the whole network representation, the permutation 
test taking as samples each of the layer landscape, and the complexity of each 
image latent representation along the network.

- For the whole batch of images, it compares the different U-Net models using 
the permutation test of each network with itself layer by layer 
(_layer_pvalue), the test between whole networks ('_pvalue'), and the test 
between the layers within a network('_samenet_pvalue'). It is also possible to 
compute the complexity of the latent representations of the images along the 
network for certain U-net model (num_batch_complexity).


Parameters
----------
resol : resolution of landscapes
numkthlands : number of k-th landscapes
layer_names : names of the layers studied (for the graphic)
dir_data : directory path where the activation maps are stored
dir_results : directory path where results are saved
name_act_maps : common part of the name of the activation maps files
name_single_imgs : name of the activation maps for which single image 
landscapes are computed
names_unets : name of the CNN models
images : images for which single-image landscapes are computed
num_batch_complexity : CNN for which complexity is computed
file_name_single_imgs : name of the file where single-image results are saved
file_name_batch : name of the file where different CNN results are saved
file_name_batch_complexity : name of the file where complexity results are saved
"""

import scipy.io as sio
import numpy as np

import PersLands as PL
import parameters

def load_matlab(dir_path, name):
    activations_path = dir_path + name + '.mat'
    dataMatrix = sio.loadmat(activations_path)
    data = dataMatrix['activationCell']
    return data


if __name__=='__main__':

    LS = parameters.LS
    Ess = parameters.Ess

    resol = parameters.RESOL
    numkthlands = parameters.NUMKTHLANDS
    layer_names = parameters.NAMES_LAYERS_UNET
    dir_data = parameters.DIR_DATA
    dir_results = parameters.DIR_RESULTS

    name_act_maps = 'ctivationCell_'
    name_single_imgs = name_act_maps + 'rgb'
    names_unets = ['unet1', 'unet3', 'unet6', 'unet7', 'rgb']
    images = [2, 3, 5, 7]
    num_batch_complexity = 2

    file_name_single_imgs = 'landscapes_rgb23'
    file_name_batch = 'landscapes_Multi1234Rgb'
    file_name_batch_complexity = 'landscapesM3'


    if name_single_imgs:
        print('U-Nets single images...')
        #Load activation maps
        data = load_matlab(dir_data, name_single_imgs)
        numLayers = data.shape[1]

        #Compute latent landscapes
        landscapes_single_imgs = []
        for st in images:
            landscapes_single = PL.compute_llands_single_img(data, resol, numkthlands, LS, Ess, layer_names, dir_results, file_name_single_imgs, k=st, perf_pca=False, plot_lands=True)
            landscapes_single_imgs.append(landscapes_single)
        PL.store_data(file_name_single_imgs, landscapes_single_imgs)

        #Perform tests
        print('Test...')
        distance = PL.distance_single_imgs(landscapes_single_imgs, numLayers)
        p_value = PL.permutation_nets(landscapes_single_imgs, dir_results, file_name_single_imgs)
        PL.complexity_single_imgs(landscapes_single_imgs, numLayers, dir_results, file_name_single_imgs)

        #Save distance results
        name_str = dir_results + file_name_single_imgs + '_dist.csv'
        np.savetxt(name_str, distance, delimiter=",")


    if names_unets:
        print('U-Nets batch images...')
        landscapes_MultiRgb = []
        for st in names_unets:
            #Load activation maps
            file = name_act_maps + st
            data = load_matlab(dir_data, file)
            numLayers = data.shape[1]

            #Compute latent landscapes
            file_name = 'landscapes_' + st
            landscapes_batch = PL.compute_llands_batch(data, resol, numkthlands, LS, Ess, layer_names, dir_results, file_name, perf_pca=False, plot_lands=True)
            landscapes_MultiRgb.append(landscapes_batch)
        
        PL.store_data(file_name_batch, landscapes_MultiRgb)
        
        #Perform tests
        print('Test...')
        PL.perform_permutation_tests(landscapes_MultiRgb, file_name_batch, dir_results, resol, numLayers, numkthlands)
        PL.complexity_batch_imgs(landscapes_MultiRgb[num_batch_complexity], file_name_batch_complexity, numLayers, dir_results, resol, numkthlands)