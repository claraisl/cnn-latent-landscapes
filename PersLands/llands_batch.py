import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn import preprocessing
from scipy.special import comb

from . import base
from . import test_mod

def compute_llands_batch(data, resol, numkthlands, LS, Ess, names, dir_results, file_name, perf_pca=False, plot_lands=False):
    """Compute the latent landscapes of all the images of the batch for each 
    layer of the CNN. Display average landscapes and perform PCA. Use this if 
    there are enough pixels in a single image to compute them

    Parameters
    ----------
    data: list of shape (1, numLayers)
        Each element is a list of shape (1, size_batch)
        Then, each element is an array-like of shape (numPixels, num_channels) 
        where numPixels are the number of pixels of the activation map
        Layer output activation maps off all layers studied for each image of
        the batch

    resol : int
        Resolution of landscapes

    numkthlands : int
        Number of k-th landscapes

    LS : gudhi class for computing persistence landscapes

    Ess : gudhi class for extracting essential points in persistence diagrams

    names : list of strings
        Layer names (for the graphic)

    dir_results : string
        Directory path where results are saved

    file_name : string
        Name of the saved file

    perf_pca : bool, default=False
        Whether or not to perform PCA

    plot_lands : bool, default=False
        Whether or not to plot average latent landscapes

    Results
    -------
    Landscapes : array-like of shape (numImages, numLayers*resol*numkthlands)
    """
    #Initialize
    print('Obtaining landscapes...')
    numLayers = data.shape[1]
    numImages = data[0][0].shape[1]
    Average_landscapes = np.empty((numLayers, resol*numkthlands)) #each row is the average landscape of a layer. This is for visualization
    LandscapesPCA = np.empty((numImages*numLayers, resol*numkthlands))
    Landscapes = np.empty((numImages, numLayers*resol*numkthlands)) #matrix where each row are the landscapes of certain image for all the layers, one after the other. This is for permutation test

    #Compute landscapes
    for i in range(numLayers):
        PD_list = [] # list with a diagram per image
        numPixels = data[0][i][0][0].shape[0]
        #Get channel vectors
        num = np.minimum(numPixels, 500)
        randomNum = random.sample(range(0,numPixels), num)
        for k in range(numImages):
            data_img = data[0][i][0][k][randomNum,:]
            #Normalized data
            dataNorm = preprocessing.StandardScaler().fit_transform(data_img)
            #Persistence diagrams
            _, I1 = base.persistenceDiagram(dataNorm, radius=70)
            PD_list.append(Ess.fit_transform([I1])[0]) #Ess returns a list with just one element

        print(i)
        landscape_layer = LS.fit_transform(PD_list)  #num_diagrams x (resol * num_kthlandscapes) In this case there are as many landscapes as images
        Landscapes[:,i*resol*numkthlands:(i+1)*resol*numkthlands] = landscape_layer
        LandscapesPCA[i*numImages:(i+1)*numImages,:] = landscape_layer
        Average_landscapes[i,:] = base.average_landscape(landscape_layer)

    #Plot Average Landscape of each layer
    if plot_lands:
        base.plot_landscapes(Average_landscapes, numkthlands, resol, names, dir_results, file_name)

    #PCA
    if perf_pca:
        test_mod.perform_pca(LandscapesPCA, numLayers, numImages, dir_results, file_name, plot_pca=True)

    return Landscapes

def compute_llands_neededbatch(data, resol, numkthlands, batch, LS, Ess, names, dir_results, file_name, perf_pca=False, plot_lands=False):
    """Compute the latent landscapes for all the layers of the CNN. Display 
    average landscapes and perform PCA. Instead of each image of the batch, get 
    as many images as needed to obtain 500 latent points. We repeat this batch 
    times, taking each time random images
    
    Parameters
    ----------
    data: list of shape (1, numLayers)
        Each element is a list of shape (1, size_batch)
        Then, each element is an array-like of shape (numPixels, num_channels)
        where numPixels are the number of pixels of the activation map
        Layer output activation maps off all layers studied for each image of
        the batch

    resol : int
        Resolution of landscapes

    numkthlands : int
        Number of k-th landscapes

    batch : int
        Batch size

    LS : gudhi class for computing persistence landscapes

    Ess : gudhi class for extracting essential points in persistence diagrams

    names : list of strings
        Layer names (for the graphic)

    dir_results : string
        Directory path where results are saved

    file_name : string
        Name of the saved file

    perf_pca : bool, default=False
        Whether or not to perform PCA

    plot_lands : bool, default=False
        Whether or not to plot average latent landscapes

    Results
    -------
    Landscapes : array-like of shape (batch, numLayers*resol*numkthlands)
    """
    #Initialize
    print('Obtaining landscapes...')
    numLayers = data.shape[1]
    numImages = data[0][0].shape[1]
    Average_landscapes = np.zeros((numLayers, resol*numkthlands)) #each row is the average landscape of a layer. This is for visualization
    LandscapesPCA = np.zeros((batch*numLayers, resol*numkthlands))
    Landscapes = np.zeros((batch, numLayers*resol*numkthlands)) #matrix where each row are the landscapes of an image for all the layers of the net, one after the other. This is for permutation test
    
    #Compute landscapes
    for i in range(numLayers):
        PD_list = [] # list with a diagram per element in batch
        numPixels = data[0][i][0][0].shape[0]
        num_channels = data[0][i][0][0].shape[1]
        num = np.minimum(numPixels, 500)
        randomNum = random.sample(range(0,numPixels), num)
        len_random = len(randomNum)
        needed_num = np.ceil(500/len_random).astype('int')
        data_layer = np.zeros((needed_num*len_random, num_channels))
        for k in range(batch):
            random_img = random.sample(range(0,numImages),needed_num)
            s = 0
            for j in random_img:
                data_img = data[0][i][0][j][randomNum,:]
                data_layer[s*len_random:(s+1)*len_random,:] = data_img
                s = s+1
            #Normalized data
            dataNorm = preprocessing.StandardScaler().fit_transform(data_layer)
            #Persistence diagrams
            _, I1 = base.persistenceDiagram(dataNorm, radius=70)
            PD_list.append(Ess.fit_transform([I1])[0]) #Ess returns a list with just one element

        print(i)
        landscape_layer = LS.fit_transform(PD_list) #num_diagrams x (resol * num_kthlandscapes) In this case as many landscapes as size batch
        Landscapes[:,i*resol*numkthlands:(i+1)*resol*numkthlands] = landscape_layer
        LandscapesPCA[i*batch:(i+1)*batch,:] = landscape_layer
        Average_landscapes[i,:] = base.average_landscape(landscape_layer)

    #Plot Average Landscape of each layer
    if plot_lands:
        base.plot_landscapes(Average_landscapes, numkthlands, resol, names, dir_results, file_name)

    #PCA
    if perf_pca:
        test_mod.perform_pca(LandscapesPCA, numLayers, batch, dir_results, file_name, plot_pca=True)

    return Landscapes

def distance_mean(landscapes, numLayers, resol, numkthlands):
    """Compare latent landscapes of different CNN models by computing for each 
    layer studied the mean of the distances between the latent representations 
    of each image of the batch and the distance between average latent 
    landscapes.
    
    Parameters
    ----------
    landscapes : list where each element is array-like of shape (numImages, 
    numLayers*resol*numkthlands)
        Persistence landscapes of each CNN model

    numLayers : int
        Number of layers studied

    resol : int
        Resolution of landscapes

    numkthlands : int
        Number of k-th landscapes

    Returns
    -------
    mean_distance : array-like of shape (comb(numClasses,2), numLayers+1)

    distance_avLands : array-like of shape (comb(numClasses,2), numLayers+1)
    """
    numModels = len(landscapes)
    #distances mean
    mean_distances = np.empty((comb(numModels,2).astype('int'), numLayers+1))
    #distance between average landscapes
    distance_avLands = np.empty((comb(numModels,2).astype('int'), numLayers+1))
    t = 0
    for st in range(numModels):
        landscapes1 = landscapes[st]
        for stt in range(st+1, numModels):
            landscapes2 = landscapes[stt]
            for m in range(numLayers):
                xx0 = landscapes1[:, m*resol*numkthlands:(m+1)*resol*numkthlands]
                y0 = landscapes2[:, m*resol*numkthlands:(m+1)*resol*numkthlands]
                mean_distances[t, m] = np.mean(np.linalg.norm((xx0-y0), axis=-1))
                distance_avLands[t, m] = test_mod.statistic(xx0,y0, axis=0)
            mean_distances[t, m+1] = np.mean(np.linalg.norm((landscapes1-landscapes2), axis=-1))
            distance_avLands[t, m+1] = test_mod.statistic(landscapes1, landscapes2, axis=0)
            t = t + 1

    return mean_distances, distance_avLands

def complexity_batch_imgs(landscape, file_name, numLayers, dir_results, resol, numkthlands):
    """Compute and display the complexity of the latent landscapes of the batch 
    for a certain CNN model

    Parameters
    ----------
    landscapes : array-like of shape (numImages, numLayers*resol*numkthlands)
        Persistence landscapes of the CNN model

    file_name : string
        Name of the saved file

    numLayers : int
        Number of layers studied

    dir_results : string
        Directory path where results are saved

    resol : int
        Resolution of landscapes

    numkthlands : int
        Number of k-th landscapes
    """
    xaxis = np.arange(0,numLayers)
    complex1 = np.empty(numLayers)
    for m in range(landscape.shape[0]):
        for l in range(numLayers):
            complex1[l] = test_mod.complexity(landscape[m, l*resol*numkthlands: (l+1)*resol*numkthlands])

        plt.plot(xaxis,complex1, '-o')
    path_complex = dir_results + file_name + '_batch_complex.png'
    plt.savefig(path_complex)
    plt.show()
    plt.close()