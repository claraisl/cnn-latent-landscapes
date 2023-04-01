import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.decomposition import PCA

from . import base

def a1(data, resol, numkthlands, LS, Ess, names, x, serv_results, file_name, perform_pca=False, plot_lands=False):
    # Compute Landscapes
    print('Obtaining landscapes...')
    numLayers = data.shape[1]
    numImages = data[0][0].shape[1]
    Average_landscapes = np.zeros((numLayers, resol*numkthlands)) #each row is the average landscape of a layer. This is for visualization
    LandscapesPCA = np.zeros((numImages*numLayers, resol*numkthlands))
    Landscapes = np.zeros((numImages, numLayers*resol*numkthlands)) #matrix where each row are the landscapes of certain image for all the layers, one after the other. This is for permutation test

    for i in range(numLayers):
        PD_list = [] # list with a diagram per image
        numPixels = data[0][i][0][0].shape[0]
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
        Average_landscapes[i,:] = base.average_landscape(landscape_layer) #a single landscape, the average landscape of the layer. Matrix with the average landscape for each layer on each row

    # Plot Average Landscapes of every layer
    if plot_lands:
        num_diagrams = Average_landscapes.shape[0]
        fig, axs = plt.subplots(6,3, figsize=(17,14),sharex=True, sharey=True)
        axsFlat = axs.flat
        for l in range(num_diagrams):
            for ll in range(numkthlands):
                axsFlat[l].plot(x,Average_landscapes[l][ll*resol:(ll+1)*resol]) #Blue, Orange, Green, Red
            axsFlat[l].set_title(names[l],fontsize=40)
            plt.tight_layout()

        path_lands = serv_results + file_name + '_landscapes.png' 
        plt.savefig(path_lands)
        plt.show()
        plt.close()

    # PCA
    if perform_pca:
        print('PCA...')
        pca = PCA()
        path_pca = serv_results + file_name + '_pca.png' 
        LandSPCA = pca.fit(LandscapesPCA).transform(LandscapesPCA)
        plt.figure(figsize=(5,5))
        for l in range(numLayers):
            plt.scatter(LandSPCA[numImages*l:numImages*(l+1),0], LandSPCA[numImages*l:numImages*(l+1),1]) 
        plt.savefig(path_pca)
        plt.show()
        plt.close()

    return Landscapes