import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import itertools
import scipy
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix

from . import base

def _inner_product(x, y):
    if len(x.shape) == 2:
        return np.sum(x*y, axis=-1)
    else:
        return np.sum(x*y)
    
def complexity(x):
    """Complexity of x is defined as its norm"""
    return np.sqrt(_inner_product(x, x))

def statistic(*samples, axis):
    """Statistic used in the permutation test"""
    return np.linalg.norm(np.mean(samples[0], axis=axis) - np.mean(samples[1], axis=axis), axis=-1)

def permutation(landscapes, numLayers, resol, numkthlands, dir_results, file_name):
    """Compute permutation test between the studied layers of the CNN model
    
    Parameters
    ----------
    landscapes : array-like of shape (m, numLayers*resol*numkthlands)
        Persistence landscapes of a batch of images
    
    numLayers : int
        Number of layers studied

    resol : int
        Resolution of landscapes

    numkthlands : int
        Number of k-th landscapes

    dir_results : string
        Path where the plot should be saved

    file_name : string
        Name of the saved file

    Returns
    ------
    p_value : array-like of shape (numLayers, numLayers)
    """
    p_value = np.zeros((numLayers,numLayers));
    for m in range(numLayers):
        xx = landscapes[:,m*resol*numkthlands:(m+1)*resol*numkthlands]
        for n in range(m+1,numLayers):
            y = landscapes[:,n*resol*numkthlands:(n+1)*resol*numkthlands]
            res = scipy.stats.permutation_test((xx, y), statistic, vectorized=True, axis=0)
            p_value[m,n] = res.pvalue
        name_str = dir_results + file_name + '_samenet_pvalue.csv'
        base.write_data(name_str, p_value[m,:])
        
    return p_value

def permutation_nets(landscapes_nets, dir_results, file_name):
    """Compute permutation test between different CNN models. Each model is 
    represented by a sequence of landscapes (one per layer)
    
    Parameters
    ----------
    landscapes_nets : list where each element is array-like of shape (m, 
    numLayers*resol*numkthlands)
        Persistence landscapes of each CNN model for a batch of images

    dir_results : string
        Path where the plot should be saved

    file_name : string
        Name of the saved file

    Returns
    ------
    p_value : array-like of shape (num_nets, num_nets)
    """
    num_nets = len(landscapes_nets)
    p_value = np.zeros((num_nets,num_nets))
    for m in range(num_nets):
        net1 = landscapes_nets[m]
        for n in range(m+1, num_nets):
            net2 = landscapes_nets[n]
            res = scipy.stats.permutation_test((net1, net2), statistic, vectorized=True, n_resamples=2000, axis=0)
            p_value[m,n] = res.pvalue
        name_str = dir_results + file_name + '_pvalue.csv'
        base.write_data(name_str, p_value[m,:])

    return p_value

def permutation_layer_nets(landscapes, numLayers, resol, numkthlands, dir_results, file_name):
    """Compute permutation test between different CNN models layer by layer and 
    save the results in a csv file
    
    Parameters
    ----------
    landscapes : list where each element is array-like of shape (m, 
    numLayers*resol*numkthlands)
        Persistence landscapes of each CNN model for a batch of images
    
    numLayers : int
        Number of layers studied

    resol : int
        Resolution of landscapes

    numkthlands : int
        Number of k-th landscapes

    dir_results : string
        Path where the plot should be saved

    file_name : string
        Name of the saved file
    """
    name_str = dir_results + file_name + '_layer_pvalue.csv'
    for r in range(len(landscapes)):
        net1 = landscapes[r]
        for n in range(r+1,len(landscapes)):
            net2 = landscapes[n]
            p_value = np.zeros(numLayers)
            for m in range(numLayers):
                xx = net1[:,m*resol*numkthlands:(m+1)*resol*numkthlands]
                y = net2[:,m*resol*numkthlands:(m+1)*resol*numkthlands]
                res = scipy.stats.permutation_test((xx, y), statistic, vectorized=True, axis=0)
                p_value[m] = res.pvalue
            base.write_data(name_str, p_value)

def perform_pca(landscapes, numLayers, batch, dir_results, file_name, plot_pca=True):
    """PCA of the landscapes and display the 2d projection of all the layers
    
    Parameters
    ----------
    landscapes : array-like of shape (numLayers*batch, resol*numkthlands)
        Persistence landscapes of the studied layers

    numLayers : int
        Number of layers studied

    batch : int
        Batch size

    dir_results : string
        Path where the plot should be saved

    file_name : string
        Name of the saved file

    plot_pca : bool, default=True
        Whether or not to display the 2d projection

    Returns
    -------
    landSPCA : array-like of shape (numLayers*numImages, numLayers*numImages)
    """
    print('PCA...')
    pca = PCA()
    path_pca = dir_results + file_name + '_pca.png' 
    landSPCA = pca.fit(landscapes).transform(landscapes)

    if plot_pca:
        plt.figure(figsize=(5,5))
        for l in range(numLayers):
            plt.scatter(landSPCA[batch*l:batch*(l+1),0], landSPCA[batch*l:batch*(l+1),1]) 
        plt.savefig(path_pca)
        plt.show()
        plt.close()

    return landSPCA

def perform_permutation_tests(landscapes, file_name, dir_results, resol, numLayers, numkthlands, nets=True, layers=True, same_net=False):
    """Perform the different permutation tests
    
    Parameters
    ----------
    landscapes : list where each element is array-like of shape (m, 
    numLayers*resol*numkthlands)
        Persistence landscapes of each CNN model for a batch of images

    file_name : string
        Name of the saved file

    dir_results : string
        Path where the plot should be saved

    resol : int
        Resolution of landscapes

    numLayers : int
        Number of layers studied

    numkthlands : int
        Number of k-th landscapes

    nets : bool, default=True
        Whether or not to compute permutation test between CNN models

    layers : boo, default=True
        Whether or not to compute permutation test between CNN models layer-wise

    same_net : bool, default=False
        Whether or not to compute permutation test between layers of each CNN 
        model
    """
    # Compute permutation test of all the network
    if nets:
        print('Nets...')
        p_value = permutation_nets(landscapes, dir_results, file_name)
    #Compute permutation test by layers
    if layers:
        print('Layers...')
        permutation_layer_nets(landscapes, numLayers, resol, numkthlands, dir_results, file_name)
    #Compute permutation of layers within network
    if same_net:
        print('Within net...')
        for n in range(len(landscapes)):
            Landscapes_bb = landscapes[n]
            name_within = file_name + str(n)
            p_value = permutation(Landscapes_bb, numLayers, resol, numkthlands, dir_results, name_within)

def compute_kmeans(landscapes, resol, numkthlands, num_clusters, l, dir_results, file_name, models_name, plot_cm=True):
    """Compute K-means with the landscapes of a certain layer and display the 
    2d PCA projections, the K-means clusters and the confusion matrix
    
    Parameters
    ----------
    landscapes : list where each element is array-like of shape (m, 
    numLayers*resol*numkthlands)
        Persistence landscapes of each CNN model for a batch of images

    resol : int
        Resolution of landscapes

    numLayers : int
        Number of layers studied

    num_clusters : int
        The number of clusters to form as well as the number of centroids to 
        generate

    l : int
        The number of the layer studied

    dir_results : string
        Path where the plot should be saved

    file_name : string
        Name of the saved file

    models_name : list of strings
        Names of the CNN models

    plot_cm : bool, default=True
        Whether or not to plot the confusion matrix

    Results
    -------
    LandSPCA : array-like
        PCA transformed values

    LandS_labels : array-like
        K-means labels of each point

    cm : array-like
        Confusion matrix
    """
    # Get the right layer for all the batch of each sample
    num_samples = len(landscapes)
    print(num_samples)
    batch = landscapes[0].shape[0]
    landscapes_data = np.empty((batch*num_samples, resol*numkthlands))
    for n in range(num_samples):
        landscapes_data[n*batch:(n+1)*batch] = landscapes[n][:,l*resol*numkthlands:(l+1)*resol*numkthlands]

    #PCA
    print('PCA...')
    pca = PCA(n_components=.9)
    LandSPCA = pca.fit_transform(landscapes_data)
    print('Explained variance : ', pca.explained_variance_ratio_)

    #Two-dimensional PCA projection
    path_pca = dir_results + file_name + '_layer' + str(l) + '_clusters' + str(num_clusters)
    _plot_2dpca(LandSPCA, num_samples, models_name, path_pca)
    
    #K-means
    print('K-Means...')
    kmeans = KMeans(n_clusters=num_clusters, n_init='auto')
    LandS_labels = kmeans.fit(LandSPCA).labels_
    for i in range(num_samples):
        print(LandS_labels[i*batch:(i+1)*batch])

    #Visualization of K-means
    path_kmeans = dir_results + file_name + '_layer' + str(l) + '_clusters' + str(num_clusters)
    _plot_kmeans(LandSPCA, LandS_labels, num_clusters, path_kmeans)
    
    #Confusion matrix
    save_path = dir_results + file_name + '_layer' + str(l) + '_clusters' + str(num_clusters)
    cm = _confusion_matrix(batch, num_samples, LandS_labels, save_path, plot_cm=plot_cm)

    return LandSPCA, LandS_labels, cm

def _plot_2dpca(data, num_samples, data_legend, save_path):
    """Plot 2d projections and save the result (as pca.png)"""
    batch = data.shape[0]//num_samples
    path_pca = save_path + '_2dpca.png'
    matplotlib.rcParams['mathtext.fontset'] = 'stix'
    matplotlib.rcParams['font.family'] = 'STIXGeneral'
    marker = itertools.cycle(('o', 'v','*', 'D', '<', '+', '>', 'x', 's'))
    plt.figure(figsize=(8,7))
    for m in range(num_samples):
        if data.shape[1] == 1:
            plt.scatter(data[batch*m:batch*(m+1)], m*np.ones(batch), marker=next(marker))
        else:
             plt.scatter(data[batch*m:batch*(m+1),0], data[batch*m:batch*(m+1),1], marker=next(marker))

    plt.title('PCA projections', fontsize=44)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlabel('PC1', fontsize=25)
    plt.ylabel('PC2', fontsize=25)
    plt.legend(data_legend, loc='best', fontsize=25)
    plt.savefig(path_pca, dpi=300)
    plt.show()
    plt.close()

def _plot_kmeans(data, labels, num_clusters, save_path):
    """Plot k-means clusters and save the result (as kmeans.png)"""
    plt.figure(figsize=(9,7))
    title = 'K-means, k=' + str(num_clusters)
    path_kmeans = save_path + 'kmeans.png'
    if data.shape[1] == 1:
        plt.scatter(data, np.zeros(data.shape[0]), c=labels)
        plt.title(title, fontsize=44)
    else:
        plt.scatter(data[:,0], data[:,1], c=labels)
        plt.title(title, fontsize=44)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlabel('PC1', fontsize=25)
    plt.ylabel('PC2', fontsize=25)
    plt.savefig(path_kmeans, dpi=300)
    plt.show()
    plt.close()

def _confusion_matrix(batch, num_samples, labels, save_path, plot_cm=True):
    """Compute confusion matrix from k-means results"""
    truth = np.zeros(batch*num_samples)
    if num_samples == 4:
        for b in range(batch):
            truth[batch*3 + b] = 1
    else:
        for n in range(num_samples):
            for b in range(batch):
                truth[batch*n + b] = n

    k_labels = labels  # Get cluster labels
    k_labels_matched = np.empty_like(k_labels)

    # For each cluster label...
    for k in np.unique(k_labels):
        # ...find and assign the best-matching truth label
        match_nums = [np.sum((k_labels==k)*(truth==t)) for t in np.unique(truth)]
        k_labels_matched[k_labels==k] = np.unique(truth)[np.argmax(match_nums)]
    
    # Compute confusion matrix
    cm = confusion_matrix(truth, k_labels_matched)

    if plot_cm:
        # Plot confusion matrix
        plot_confusion_matrix(cm, save_path)

    return cm

def plot_confusion_matrix(cm, save_path):
    """Plot the given matrix and save results (as confusion.png)"""
    # Plot confusion matrix
    path_cm = save_path + '_confusion.png'
    plt.imshow(cm,interpolation='none',cmap='Blues')
    for (i, j), z in np.ndenumerate(cm):
        plt.text(j, i, z, ha='center', va='center', fontsize=40)
    plt.xlabel("kmeans label", fontsize=25)
    plt.ylabel("truth label", fontsize=25)
    plt.title("Confusion matrix", fontsize=30)
    plt.yticks([])
    plt.xticks([])

    plt.savefig(path_cm, dpi=300)
    plt.show()
    plt.close()