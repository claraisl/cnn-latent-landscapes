import numpy as np
import gudhi as gd
import gudhi.representations
import random
import matplotlib.pyplot as plt
from sklearn import preprocessing
from scipy.special import comb

from . import base
from . import test_mod


def compute_llands_single_img(data, resol, numkthlands, LS, Ess, names, dir_results, file_name, k=0, perf_pca=False, plot_lands=False):
    """Compute the latent landscape of a single image for all the layers of the 
    CNN. Display barcodes, landscapes and PCA 2D-projections. Use this if there 
    are enough pixels in a single image to compute them
    
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

    k : int, default=0
        Number of the image studied

    perf_pca : bool, default=False
        Whether or not to perform PCA

    plot_lands : bool, default=False
        Whether or not to plot barcodes and latent landscapes

    Results
    -------
    Landscapes : array-like of shape (numLayers, resol * numkthlands)
    """
    print('Obtaining persistence diagrams...')
    numLayers = data.shape[1]
    PD_list = [] #list with a persistence diagram per layer
    file_name = file_name + str(k)
    if plot_lands:
        num_rows = np.ceil(numLayers/3).astype('int')
        fig, axs = plt.subplots(num_rows,3, figsize=(17,14),sharex=True)
        axsFlat = axs.flat
        
    for i in range(numLayers):
        numPixels = data[0][i][0][0].shape[0]
        num_channels = data[0][i][0][0].shape[1]
        
        #Get channel vectors
        num = np.minimum(numPixels, 500) 
        randomNum = random.sample(range(0,numPixels), num)
        len_randomNum = len(randomNum)
        data_layer = np.empty((len_randomNum, num_channels))
        data_layer = data[0][i][0][k][randomNum,:]
        
        #Normalized data
        dataNorm = preprocessing.StandardScaler().fit_transform(data_layer)

        #Persistence diagrams
        _, I1 = base.persistenceDiagram(dataNorm, radius=70)
        PD_list.append(Ess.fit_transform([I1])[0]) #Ess returns a list with just one element
        print(i)

        #Plot barcodes
        if plot_lands:
            gd.plot_persistence_barcode(I1, legend=False, axes=axsFlat[i])
            axsFlat[i].set_title(names[i],fontsize=40)
            plt.tight_layout()
    
    if plot_lands:
        path_bar = dir_results + file_name + '_barcodes.png'
        plt.savefig(path_bar)
        plt.show()
        plt.close()

    #Landscapes
    print('Obtaining landscapes...')
    Landscapes = LS.fit_transform(PD_list)
    if plot_lands:
        base.plot_landscapes(Landscapes, numkthlands, resol, names, dir_results, file_name)

    #PCA
    if perf_pca:
        test_mod.perform_pca(Landscapes, numLayers, 1, dir_results, file_name)

    return Landscapes

def compute_llands_single_neededimg(data, resol, numkthlands, LS, Ess, names, dir_results, file_name, k=0, perf_pca=False, plot_lands=False):
    """Compute the latent landscape for all the layers of the CNN. Display 
    barcodes, landscapes and PCA 2D-projections. Instead of a single image, get 
    as much images as needed to obtain 500 latent points. In this case, in 
    order and the same images for all layers

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

    k : int
        Number of the image to begin with

    perf_pca : bool, default=False
        Whether or not to perform PCA

    plot_lands : bool, default=False
        Whether or not to plot barcodes and latent landscapes

    Results
    -------
    Landscapes : array-like of shape (numLayers, resol * numkthlands)
    """
    print('Obtaining PD...')
    numLayers = data.shape[1]
    PD_list = [] #list with a persistence diagram per layer
    if plot_lands:
        num_rows = np.ceil(numLayers/3).astype('int')
        fig, axs = plt.subplots(num_rows,3, figsize=(17,14),sharex=True)
        axsFlat = axs.flat
        
    for i in range(numLayers):
        numPixels = data[0][i][0][0].shape[0]
        num_channels = data[0][i][0][0].shape[1]

        #Get channel vectors
        num = np.minimum(numPixels, 500)
        randomNum = random.sample(range(0,numPixels), num)
        len_randomNum = len(randomNum)
        needed_num = np.ceil(500/len_randomNum).astype('int')
        data_layer = np.empty((needed_num*len_randomNum, num_channels))
        for s in range(needed_num):
            data_img = data[0][i][0][k+s][randomNum,:]
            data_layer[s*len_randomNum:(s+1)*len_randomNum,:] = data_img
        
        #Normalized data
        dataNorm = preprocessing.StandardScaler().fit_transform(data_layer)

        #Persistence diagrams
        _, I1 = base.persistenceDiagram(dataNorm, radius=70)
        PD_list.append(Ess.fit_transform([I1])[0]) #Ess returns a list with just one element
        print(i)

        #Plot barcodes
        if plot_lands:
            gd.plot_persistence_barcode(I1, legend=False, axes=axsFlat[i])
            axsFlat[i].set_title(names[i],fontsize=40)
            plt.tight_layout()

        
    if plot_lands:
        path_bar = dir_results + file_name + '_barcodes.png'
        plt.savefig(path_bar)
        plt.show()
        plt.close()

    #Landscapes
    print('Obtaining landscapes...')
    Landscapes = LS.fit_transform(PD_list)
    if plot_lands:
        base.plot_landscapes(Landscapes, numkthlands, resol, names, dir_results, file_name)

    #PCA
    if perf_pca:
        test_mod.perform_pca(Landscapes, numLayers, 1, dir_results, file_name)

    return Landscapes

def distance_single_imgs(landscapes, numLayers):
    """Compare latent landscapes of different images by computing the distance 
    between each layer landscape and the distance of the entire network 
    representation
    
    Parameters
    ----------
    landscapes : list where each element is array-like of shape (numLayers, 
    resol*numkthlands)
        Persistence landscapes of each image

    numLayers : int
        Number of layers studied

    Returns
    -------
    distance : array-like of shape (comb(numImages,2), numLayers + 1)
    """
    numImages = len(landscapes)
    distance = np.empty((comb(numImages,2).astype('int'), numLayers + 1))
    t = 0
    for st in range(numImages):
        landscapes1 = landscapes[st]
        for stt in range(st+1, numImages):
            landscapes2 = landscapes[stt]
            for m in range(numLayers):
                xx = landscapes1[m,:]
                y = landscapes2[m,:]
                distance[t, m] = np.linalg.norm(xx-y)
            distance[t, m+1] = np.linalg.norm(landscapes1.flatten() - landscapes2.flatten())
            t = t + 1

    return distance

def complexity_single_imgs(landscapes, numLayers, dir_results, file_name):
    """Compute and display the complexity of the latent landscapes of different 
    images
    
    Parameters
    ----------
    landscapes : list where each element is array-like of shape (numLayers, 
    resol*numkthlands)
        Persistence landscapes of each image

    numLayers : int
        Number of layers studied

    dir_results : string
        Directory path where results are saved

    file_name : string
        Name of the saved file
    """
    print('Complexity...')
    xaxis = np.arange(0,numLayers)
    complex1 = np.empty(numLayers)
    plt.figure(figsize=(5,5))
    for n in range(len(landscapes)):
        for m in range(numLayers):
            complex1[m] = test_mod.complexity(landscapes[n][m,:])

        plt.plot(xaxis,complex1, '-o')

    path_complex = dir_results + file_name + '_complexity.png'
    plt.savefig(path_complex)
    plt.show()
    plt.close()