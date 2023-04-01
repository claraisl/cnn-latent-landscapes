import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.decomposition import PCA

from . import base

def a11(data, resol, numkthlands, batch, LS, Ess, names, x, serv_results, file_name, perform_pca=False, plot_lands=False):
    # Compute Landscapes
    print('Obtaining landscapes...')
    numLayers = data.shape[1]
    numImages = data[0][0].shape[1]
    Average_landscapes = np.zeros((numLayers, resol*numkthlands)) #each row is the average landscape of a layer. This is for visualization
    Landscapes = np.zeros((batch, numLayers*resol*numkthlands)) #matrix where each row are the landscapes of an image for all the layers of the net, one after the other. This is for permutation test
    LandscapesPCA = np.zeros((batch*numLayers, resol*numkthlands))

    #Similar to a00 but it is a1. Instead of computing the landscape for each of the images of the batch and do the average, we get as many images as we need to obtain 500 latent points and then compute the landscape. We repeat this batch times, taking each time random images. Then, we obtain batch landscapes and do the average.
    
    for i in range(numLayers):
        PD_list = [] # list with a diagram per element in batch
        numPixels = data[0][i][0][0].shape[0]
        num_channels = data[0][i][0][0].shape[1]
        num = np.minimum(numPixels, 500)
        randomNum = random.sample(range(0,numPixels), num)
        len_random = len(randomNum)
        needed_num = np.ceil(500/len_random).astype('int')
        data_layer = np.zeros((needed_num*len_random, num_channels))
        #random.seed(0) 
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

    # Plot Landscapes
    num_diagrams = Average_landscapes.shape[0]
    if plot_lands:
        fig, axs = plt.subplots(5,3, figsize=(17,14),sharex=True,sharey=True)
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
            plt.scatter(LandSPCA[batch*l:batch*(l+1),0], LandSPCA[batch*l:batch*(l+1),1]) 
        plt.savefig(path_pca)
        plt.show()
        plt.close()

    return Landscapes