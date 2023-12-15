"""Performs experiments on VGG-16/CIFAR-10 models:
- Distance
- Average latent landscapes plot
- K-means

Parameters
----------
resol : resolution of landscapes
numkthlands : number of k-th landscapes
layer_names : names of the layers studied (for the graphic)
models : names of the CNN models (for the graphic)
dir_results : directory path where results are saved
file_name : name of landscapes file to load
"""

import numpy as np

import PersLands as PL
import parameters

if __name__=='__main__':

    resol = parameters.RESOL
    numkthlands = parameters.NUMKTHLANDS
    layer_names = parameters.NAMES_LAYERS_VGG
    models = parameters.CLASSES
    dir_results = parameters.DIR_RESULTS
    layers = parameters.LAYERS_PLOT_VGG

    numLayers = len(layer_names)

    file_name = 'landscapes_catdogairmulti_batch'

    landscapes_models = PL.load_data(file_name)

    mean_distances, distance_avLands = PL.distance_mean(landscapes_models[:3], numLayers, resol, numkthlands)
    #Save distance results
    name_strMn = dir_results + file_name + '_meanDist.csv'
    name_strAv = dir_results + file_name + '_distAv.csv'
    np.savetxt(name_strMn, mean_distances, fmt='%.5e', delimiter=",")
    np.savetxt(name_strAv, distance_avLands, fmt='%.5e', delimiter=",")

    some_layer_names = [layer_names[i] for i in layers]
    PL.plot_avLandscapes_rownets_columnlayers(landscapes_models, layers, file_name, resol, some_layer_names, numkthlands, dir_results, models)
    PL.plot_avLandscapes_rowlayers_columnnets(landscapes_models, file_name, resol, layer_names, numkthlands, dir_results, models)

    num_clusters = [3, 2]; layers = [5, 12]
    for l in layers:
        for idx, i in enumerate(num_clusters):
            landscapes = landscapes_models[: 3 + idx]
            models_name = models[: 3 + idx]
            LandSPCA, LandS_labels, cm = PL.compute_kmeans(landscapes, resol, numkthlands, i, l, dir_results, file_name, models_name, plot_cm=True)