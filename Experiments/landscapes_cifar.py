"""This script computes and saves the latent landscapes for CIFAR-10 from the 
activations maps of the VGG-16 layers studied. In addition, it plots the 
(average) latent landscapes of every layer, performs PCA and different 
experiments.

In our case, we study the ReLU layers, storing the output activation maps in 
'activationCell.mat' from Matlab and loading them in Python.

We recommend to work with the saved latent landscapes and perform with them 
different experiments using 'cifar_tests.py'.

- For single images (a0) (or enough of them, a00) of different classes, it 
compares them computing the distance between each layer landscape, the distance 
of the whole model representation, and the complexity of each image latent 
representation along the network.

- For the whole batch, it compares the different models using the permutation 
test of each model with itself layer by layer (_layer_pvalue), the test between 
whole models ('_pvalue'), and the test between the layers within a model
('_samenet_pvalue'). Moreover, it computes the mean distance between landscapes 
of the same layer (_meanDist) and the distance between average landscapes 
(_distAv). It is also possible to compute the complexity of the latent 
representations of the batch along a certain model (num_batch_complexity).

Parameters
----------
resol : resolution of landscapes
numkthlands : number of k-th landscapes
batch : batch size
layer_names : names of the layers studied (for the graphic)
dir_data : directory path where the activation maps are stored
dir_results : directory path where results are saved
name_act_maps : common part of the name of the activation maps files
name_single_imgs : name of the activation maps for which single image 
landscapes are computed
names_models : name of the CNN models
images : images for which single-image landscapes are computed
num_batch_complexity : CNN model for which complexity is computed
file_name_single_imgs : name of the file where single-image results are saved
file_name_batch : name of the file where batch results are saved
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
    batch = parameters.BATCH
    layer_names = parameters.NAMES_LAYERS_VGG
    dir_data = parameters.DIR_DATA
    dir_results = parameters.DIR_RESULTS

    name_act_maps = 'ctivationCell_cifar'
    name_single_imgs = ['_cat', '_dog', '_airplane', '']
    name_models = ['_cat', '_dog', '_airplane', '']
    images = [0]
    num_batch_complexity = 3

    file_name_single_imgs = 'landscapes_catdogairmulti_img'
    file_name_batch = 'landscapes_catdogairmulti_batch'
    file_name_batch_complexity = 'landscapesmulti'

    if name_single_imgs:
        print('CIFAR single images...')
        landscapes_single = []
        landscapes_singleneeded = []
        for st in name_single_imgs:
            #Load activation maps
            file = name_act_maps + st
            data = load_matlab(dir_data, file)
            numLayers = data.shape[1]

            #Compute latent landscapes
            file_name0 = 'landscapes_img' + st
            file_name00 = 'landscapes_imgneed_' + st
            for s in images:
                Landscapes_a0 = PL.compute_llands_single_img(data, resol, numkthlands, LS, Ess, layer_names, dir_results, file_name0, k=s, perf_pca=False, plot_lands=True)
                Landscapes_a00 = PL.compute_llands_single_neededimg(data, resol, numkthlands, LS, Ess, layer_names, dir_results, file_name00, k=s, perf_pca=False, plot_lands=True)
                
                landscapes_single.append(Landscapes_a0)
                landscapes_singleneeded.append(Landscapes_a00)
            
        PL.store_data(file_name_single_imgs, landscapes_single)
        PL.store_data(file_name_single_imgs + 'need', landscapes_singleneeded)

        #Perform tests
        print('Test...')
        distance0 = PL.distance_single_imgs(landscapes_single, numLayers)
        distance00 = PL.distance_single_imgs(landscapes_singleneeded, numLayers)
        #Save distance results
        name_str0 = dir_results + file_name_single_imgs + '_dist.csv'
        name_str00 = dir_results + file_name_single_imgs + 'need_dist.csv'
        np.savetxt(name_str0, distance0, fmt='%.5e', delimiter=",")
        np.savetxt(name_str00, distance00, fmt='%.5e', delimiter=",")

        PL.complexity_single_imgs(landscapes_singleneeded, numLayers, dir_results, file_name_single_imgs + 'need')


    if name_models:
        print('CIFAR image batch...')
        landscapes_models = []
        for st in name_models:
            #Load activation maps
            file = name_act_maps + st
            data = load_matlab(dir_data, file)
            numLayers = data.shape[1]

            #Compute latent landscapes
            file_name = 'landscapes' + st
            landscapes_batch = PL.compute_llands_neededbatch(data, resol, numkthlands, batch, LS, Ess, layer_names, dir_results, file_name, perf_pca=False, plot_lands=True)
            landscapes_models.append(landscapes_batch)

        PL.store_data(file_name_batch, landscapes_models)
        
        #Perform tests
        print('Test...')
        PL.perform_permutation_tests(landscapes_models, file_name_batch, dir_results, resol, numLayers, numkthlands, nets=False, layers=True)

        mean_distances, distance_avLands = PL.distance_mean(landscapes_models, numLayers, resol, numkthlands)
        #Save distance results
        name_strMn = dir_results + file_name_batch + '_meanDist.csv'
        name_strAv = dir_results + file_name_batch + '_distAv.csv'
        np.savetxt(name_strMn, mean_distances, fmt='%.5e', delimiter=",")
        np.savetxt(name_strAv, distance_avLands, fmt='%.5e', delimiter=",")

        PL.complexity_batch_imgs(landscapes_models[num_batch_complexity], file_name_batch_complexity, numLayers, dir_results, resol, numkthlands)