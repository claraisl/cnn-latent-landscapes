import numpy as np
import gudhi as gd
import gudhi.representations
import random
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.decomposition import PCA

from . import base


def a0(data, resol, numkthlands, LS, Ess, names, x, serv_results, file_name, perform_pca=False, k=0, plot_lands=False):
    print('Obtaining PD...')
    numLayers = data.shape[1]
    numImages = data[0][0].shape[1]
    PD_list = [] # list with a diagram per layer
    fig, axs = plt.subplots(6,3, figsize=(17,14),sharex=True)
    axsFlat = axs.flat
        
    for i in range(numLayers):
        numPixels = data[0][i][0][0].shape[0]
        num = np.minimum(numPixels, 500) 
        randomNum = random.sample(range(0,numPixels), num)
        len_randomNum = len(randomNum)

        num_channels = data[0][i][0][0].shape[1]
        data_layer = np.zeros((len_randomNum, num_channels))
        #print(data_layer.shape)

        data_layer = data[0][i][0][k][randomNum,:]
        #Normalized data
        dataNorm = preprocessing.StandardScaler().fit_transform(data_layer)
        print(dataNorm.shape)
        #Persistence diagrams
        _, I1 = base.persistenceDiagram(dataNorm, radius=70)
        if plot_lands:
            gd.plot_persistence_barcode(I1, legend=False, axes=axsFlat[i])
            axsFlat[i].set_title(names[i],fontsize=40)
            plt.tight_layout()

        PD_list.append(Ess.fit_transform([I1])[0]) #Ess returns a list with just one element
        
    if plot_lands:
        path_bar = serv_results + file_name + '_barcodes.png'
        plt.savefig(path_bar)
        plt.show()
        plt.close()

    #Landscapes
    print('Obtaining landscapes...')
    Landscapes = LS.fit_transform(PD_list)
    #print(Landscapes.shape) #num_diagrams x (resol * num_kthlandscapes)
    num_diagrams = Landscapes.shape[0]
    if plot_lands:
        fig, axs = plt.subplots(6,3, figsize=(17,14),sharex=True, sharey=True)
        axsFlat = axs.flat
        for l in range(num_diagrams):
            for ll in range(numkthlands):
                axsFlat[l].plot(x,Landscapes[l][ll*resol:(ll+1)*resol]) #Blue, Orange, Green, Red
            axsFlat[l].set_title(names[l],fontsize=40)
            plt.tight_layout()

        path_lands = serv_results + file_name + '_landscapes.png'
        plt.savefig(path_lands)
        plt.show()
        plt.close()

    #PCA
    if perform_pca:
        pca = PCA()
        LandSPCA = pca.fit(Landscapes).transform(Landscapes)
        print(LandSPCA.shape)
        # 2D Projection of the data
        plt.figure(figsize=(5,5))
        for m in range(LandSPCA.shape[0]):
            plt.scatter(LandSPCA[m,0], LandSPCA[m,1]) 

        path_pca = serv_results + file_name + '_pca.png'
        plt.savefig(path_pca)
        plt.show()
        plt.close()

    return Landscapes