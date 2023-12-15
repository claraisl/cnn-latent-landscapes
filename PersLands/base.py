import numpy as np
import gudhi as gd
import gudhi.representations
import csv
import pickle
import matplotlib
import matplotlib.pyplot as plt

def persistenceDiagram(data, radius=100, max_dim=2):
    """Compute persistence diagrams associated to H0 and H1

    Parameters
    ----------
    data : array-like
        Collection of channel vectors of a layer

    radius : int, default=100
        alpha_max

    max_dim : int, default=2
        Maximum simplex dimension

    Returns
    -------
    I0 : list of pairs (dimension, pair(birth, death))
        persistence intervals of the simplicial complex in dim=0

    I1 : list of pairs (dimension, pair(birth, death))
        persistence intervals of the simplicial complex in dim=1
    """
    skeleton = gd.RipsComplex(points = data, max_edge_length = radius);
    Rips_simplex_tree= skeleton.create_simplex_tree(max_dimension = max_dim);
    BarCodes_Rips =Rips_simplex_tree.persistence();
    I0 = Rips_simplex_tree.persistence_intervals_in_dimension(0)
    I1 = Rips_simplex_tree.persistence_intervals_in_dimension(1)

    return I0, I1

def bottleneck(dgm1, dgm2):
    """Compute the bottleneck distance between two persistence diagrams
    
    Parameters
    ----------
    dgm1 : array-like of shape (m,2)
        The first diagram

    dgm2 : array-like of shape (n,2)
        The second diagram
    
    Returns
    -------
    dist : float
    """
    Ess = gd.representations.preprocessing.DiagramSelector(use = True, point_type='finite')
    dgm1_NoEss = Ess.fit_transform([dgm1])
    dgm2_NoEss = Ess.fit_transform([dgm2])
    dist = gd.bottleneck_distance(dgm1_NoEss[0], dgm2_NoEss[0])

    return dist

def load_data(file_name):
    """Load data (deserialize)"""
    print('Loading...')
    name_str = file_name + '.pickle'
    with open(name_str, 'rb') as handle:
        data = pickle.load(handle)

    return data

def store_data(file_name, data):
    """Store data (serialize)"""
    name_str = file_name + '.pickle'
    with open(name_str, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

def write_data(file_path, data):
    """Write data on a csv file"""
    with open(file_path, mode='a', newline='') as file:
        compare_writer = csv.writer(file, delimiter=',')
        compare_writer.writerow(data)

def average_landscape(landscapes):
    """Compute average landscape"""
    avLands = np.mean(landscapes, axis=0)
    return avLands

def plot_landscapes(landscapes, numkthlands, resol, layer_names, dir_results, file_name):
    """Display landscapes of each layer while computing them for a single CNN 
    model
    
    Parameters
    ----------
    landscapes : array-like of shape (numLayers, resol*numkthlands)
        Landscapes

    numkthlands : int
        Number of k-th landscapes

    resol : int
        Resolution of landscapes

    layer_names : list of strings
        Names of the layers studied

    dir_results : string
        Path where the plot should be saved

    file_name : string
        Name of the saved file
    """
    num_diagrams = landscapes.shape[0]
    x = np.linspace(0,resol,num=resol)
    num_rows = np.ceil(num_diagrams/3).astype('int')
    fig, axs = plt.subplots(num_rows,3, figsize=(17,14),sharex=True, sharey=True)
    axsFlat = axs.flat
    for l in range(num_diagrams):
        for ll in range(numkthlands):
            axsFlat[l].plot(x,landscapes[l][ll*resol:(ll+1)*resol]) #Blue, Orange, Green, Red
        axsFlat[l].set_title(layer_names[l],fontsize=40)
        plt.tight_layout()

    path_lands = dir_results + file_name + '_plot.png' 
    plt.savefig(path_lands)
    plt.show()
    plt.close()

def plot_avLandscapes_rowlayers_columnnets(landscapes, file_name, resol, name_layers, numkthlands, dir_results, models):
    """Display average latent landscapes of each layer for the different CNN 
    models
    
    Parameters
    ----------
    landscapes : list where each element is array-like of shape (numImages, 
    numLayers*resol*numkthlands)
        Persistence landscapes of each CNN model

    file_name : string
        Name of the saved file

    resol : int
        Resolution of landscapes

    name_layers : list of strings
        Names of the layers studied (for the graphic)

    numkthlands : int
        Number of k-th landscapes

    dir_results : string
        Path where the plot should be saved

    models : list of strings
        Names of the CNN models
    """
    matplotlib.rcParams['mathtext.fontset'] = 'stix'
    matplotlib.rcParams['font.family'] = 'STIXGeneral'

    num_nets = len(landscapes)
    numLayers = len(name_layers)
    x = np.linspace(0,resol,num=resol)
    fig, axs = plt.subplots(numLayers, num_nets, figsize=(18,35),sharex=True, sharey='row')
    axsFlat = axs.flat
    
    l = 0
    for n in range(numLayers*num_nets):
        m = np.remainder(n,num_nets)
        if m == 0 and not n == 0:
            l = l + 1
        
        avLandscape_net = average_landscape(landscapes[m])
        avLandscape_layer = avLandscape_net[l*resol*numkthlands:(l+1)*resol*numkthlands]
        for t in range(numkthlands):
            axsFlat[n].plot(x,avLandscape_layer[t*resol:(t+1)*resol]) #Blue, Orange, Green, Red
        axsFlat[num_nets*l].set_ylabel(name_layers[l],fontsize=30)
        axsFlat[m].set_title(models[m], fontsize=57)
        plt.tight_layout()

    path_lands = dir_results + file_name + '_vplot.png' 
    plt.savefig(path_lands, dpi=500)
    plt.show()
    plt.close()

def plot_avLandscapes_rownets_columnlayers(landscapes, layers, file_name, resol, name_layers, numkthlands, dir_results, models):
    """Display average latent landscapes of each specified layer for the different CNN 
    models
    
    Parameters
    ----------
    landscapes : list where each element is array-like of shape (numImages, 
    numLayers*resol*numkthlands)
        Persistence landscapes of each CNN model

    layers : list of int
        Layers studied

    file_name : string
        Name of the saved file

    resol : int
        Resolution of landscapes

    name_layers : list of strings
        Names of the layers studied (for the graphic)

    numkthlands : int
        Number of k-th landscapes

    dir_results : string
        Path where the plot should be saved

    models : list of strings
        Names of the CNN models
    """
    matplotlib.rcParams['mathtext.fontset'] = 'stix'
    matplotlib.rcParams['font.family'] = 'STIXGeneral'

    num_nets = len(landscapes)
    numLayers = len(name_layers)
    x = np.linspace(0,resol,num=resol)
    fig, axs = plt.subplots(num_nets, numLayers, figsize=(30, 10),sharex=True, sharey='row')
    axsFlat = axs.flat
    
    l = 0
    for n in range(numLayers*num_nets):
        m = np.remainder(n,numLayers)
        if m == 0 and not n == 0:
            l = l + 1
        
        avLandscape_net = average_landscape(landscapes[l])
        mm = layers[m]
        avLandscape_layer = avLandscape_net[mm*resol*numkthlands:(mm+1)*resol*numkthlands]
        for t in range(numkthlands):
            axsFlat[n].plot(x,avLandscape_layer[t*resol:(t+1)*resol]) #Blue, Orange, Green, Red
        axsFlat[m].set_title(name_layers[m],fontsize=40)
        axsFlat[numLayers*l].set_ylabel(models[l], fontsize=40)
        plt.tight_layout()

    path_lands = dir_results + file_name + '_hplot.png' 
    plt.savefig(path_lands, dpi=500)
    plt.show()
    plt.close()