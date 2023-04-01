import numpy as np
import scipy
import csv
import pickle
import PersLands as PL
import matplotlib
import matplotlib.pyplot as plt
import scipy.stats as ss
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix

def load_permutation(name, serv_results, resol, numLayers, numkthlands, nets = True, layers = True, same_net=False):
    # Load data (deserialize)
    print('Loading...')
    name_str = name + '.pickle'
    with open(name_str, 'rb') as handle:
        landscapes = pickle.load(handle)
    
    # Compute permutation test of all the network
    if nets:
        print('Nets...')
        name_nets = name + '_pvalue'
        p_value = PL.permutation_nets(landscapes, serv_results, name_nets)
        print(p_value)
    #Compute permutation test by layers
    if layers:
        print('Layers...')
        name_layers = name + '_layer_pvalue'
        PL.permutation_layer_nets(landscapes, numLayers, resol, numkthlands, serv_results, name_layers)
    #Compute permutation of layers within network
    if same_net:
        print('Within net...')
        for n in range(len(landscapes)):
            Landscapes_bb = landscapes[n]
            name_within = name + str(n) + 'samenet_pvalue'
            p_value = PL.permutation(Landscapes_bb, numLayers, resol, numkthlands, serv_results, name_within)

def distance(name, classes, numLayers, serv_results):
    # Load data (deserialize)
    print('Loading...')
    name_str = name + '.pickle'
    with open(name_str, 'rb') as handle:
        landscapes = pickle.load(handle)

    print('Distance...')
    numClasses = len(classes)
    for st in range(numClasses):
        net01 = landscapes[st]
        for stt in range(st+1, numClasses):
            net02 = landscapes[stt]
            distance0 = np.zeros(numLayers+1)
            for m in range(numLayers):
                xx0 = net01[m,:]
                y0 = net02[m,:]
                distance0[m] = np.linalg.norm(xx0-y0)
            distance0[m+1] = np.linalg.norm(net01.flatten() - net02.flatten())
            
            name_str0 = serv_results + name + '_distance_' + classes[st] + classes[stt] + '.csv'
            with open(name_str0, mode='a', newline='') as dist_file:
                compare_writer = csv.writer(dist_file, delimiter=',')
                compare_writer.writerow(distance0)

def distanceMean(name, classes, numLayers, serv_results, resol, numkthlands):
    # Load data (deserialize)
    print('Loading...')
    name_str = name + '.pickle'
    with open(name_str, 'rb') as handle:
        landscapes = pickle.load(handle)

    print('Distance...')
    numClasses = len(classes)
    for st in range(numClasses):
        net01 = landscapes[st]
        for stt in range(st+1, numClasses):
            net02 = landscapes[stt]
            distance0 = np.zeros(numLayers+1) #distances mean
            distanceAv = np.zeros(numLayers+1) #distance between average landscapes
            for m in range(numLayers):
                xx0 = net01[:, m*resol*numkthlands:(m+1)*resol*numkthlands]
                y0 = net02[:, m*resol*numkthlands:(m+1)*resol*numkthlands]
                distance0[m] = np.mean(np.linalg.norm((xx0-y0), axis=-1))
                distanceAv[m] = PL.statistic(xx0,y0, axis=0)
            distance0[m+1] = np.mean(np.linalg.norm((net01-net02), axis=-1))
            distanceAv[m+1] = PL.statistic(net01, net02, axis=0)
            
            name_str0 = serv_results + name + '_distanceMean_' + classes[st] + classes[stt] + '.csv'
            with open(name_str0, mode='a', newline='') as dist_file:
                compare_writer = csv.writer(dist_file, delimiter=',')
                compare_writer.writerow(distance0)

            name_strAv = serv_results + name + '_distanceAv_' + classes[st] + classes[stt] + '.csv'
            with open(name_strAv, mode='a', newline='') as dist_file:
                compare_writer = csv.writer(dist_file, delimiter=',')
                compare_writer.writerow(distanceAv)

def complexity0(name, numLayers, serv_results):
    # Load data (deserialize)
    print('Loading...')
    name_str = name + '.pickle'
    with open(name_str, 'rb') as handle:
        landscapes = pickle.load(handle)

    print('Complexity...')
    xaxis = np.arange(0,numLayers)
    complex1 = np.zeros(numLayers)
    for n in range(len(landscapes)):
        for m in range(numLayers):
            complex1[m] = PL.complexity(landscapes[n][m,:])

        plt.plot(xaxis,complex1, '-o')

    print('Plot...')
    path_complex = serv_results + name + '_0complex.png'
    plt.savefig(path_complex)
    plt.show()
    plt.close()

def complexity(name, numLayers, serv_results, resol, numkthlands, num_net=3):
    # Load data (deserialize)
    print('Loading...')
    name_str = name + '.pickle'
    with open(name_str, 'rb') as handle:
        landscapes = pickle.load(handle)

    print('Complexity...')
    xaxis = np.arange(0,numLayers)
    complex1 = np.zeros(numLayers)
    landscape = landscapes[num_net]
    for m in range(landscape.shape[0]):
        for l in range(numLayers):
            complex1[l] = PL.complexity(landscape[m, l*resol*numkthlands: (l+1)*resol*numkthlands])

        plt.plot(xaxis,complex1, '-o')

    print('Plot...')
    #print(m)
    path_complex = serv_results + name + '_batch_complex.png'
    plt.savefig(path_complex)
    plt.show()
    plt.close()


def plot_avLandscapes_rowlayers_columnnets(name, resol, name_layers, numkthlands, serv_results, classes):
    # Load data (deserialize)
    print('Loading...')
    name_str = name + '.pickle'
    with open(name_str, 'rb') as handle:
        landscapes = pickle.load(handle)

    matplotlib.rcParams['mathtext.fontset'] = 'stix'
    matplotlib.rcParams['font.family'] = 'STIXGeneral'

    print('Plot...')
    #num_nets = len(landscapes) #cifar
    nets = [0, 1, 2, 3] #unet
    num_nets = len(nets)
    numLayers = len(name_layers)
    #layers = [0, 3, 6, 7, 8, 9, 11, 13, 16] #unet
    #layers = [0, 2, 5, 6, 9, 10, 11, 12] #cifar
    #numLayers = len(layers)
    x = np.linspace(0,resol,num=resol)
    fig, axs = plt.subplots(numLayers, num_nets, figsize=(18,30),sharex=True, sharey='row')
    #fig.suptitle('U-Net average latent landscapes', fontsize=45)
    axsFlat = axs.flat
    l = 0
    for n in range(numLayers*num_nets):
        m = np.remainder(n,num_nets)
        if m == 0 and not n == 0:
            l = l + 1
        
        avLandscape_net = PL.average_landscape(landscapes[nets[m]])
        #mm = layers[m]
        avLandscape_layer = avLandscape_net[l*resol*numkthlands:(l+1)*resol*numkthlands]
        for t in range(numkthlands):
            axsFlat[n].plot(x,avLandscape_layer[t*resol:(t+1)*resol]) #Blue, Orange, Green, Red
        axsFlat[num_nets*l].set_ylabel(name_layers[l],fontsize=37)
        axsFlat[m].set_title(classes[m], fontsize=57)
        plt.tight_layout()

    path_lands = serv_results + name + '_landscapesPlot.png' 
    plt.savefig(path_lands)
    plt.show()
    plt.close()

def compute_kmeans(name, batch, resol, numkthlands, numLayers, num_clusters, false_clus, num_samples, layer, more_name=''):
    landscapes = PL.load_data(name)
    l = layer

    # Get the right layer for all the batch of each sample
    landscapes_data = np.zeros((batch*num_samples, resol*numkthlands))
    for n in range(num_samples):
        landscapes_data[n*batch:(n+1)*batch] = landscapes[n][:,l*resol*numkthlands:(l+1)*resol*numkthlands]

    #PCA
    pca = PCA(n_components=.9)
    LandSPCA = pca.fit_transform(landscapes_data)
    print(LandSPCA.shape)
    print(pca.explained_variance_ratio_)

    mark = ['o', 'v','*', 'D']

    #Two-dimensional PCA projection
    matplotlib.rcParams['mathtext.fontset'] = 'stix'
    matplotlib.rcParams['font.family'] = 'STIXGeneral'
    path_pca = serv_results + 'cifar_' + more_name + str(layer) + '_pca.png' 
    plt.figure(figsize=(6,6))
    for m in range(num_samples):
        if LandSPCA.shape[1] == 1:
            plt.scatter(LandSPCA[batch*m:batch*(m+1)], m*np.ones(batch), marker=mark[m])
        else:
             plt.scatter(LandSPCA[batch*m:batch*(m+1),0], LandSPCA[batch*m:batch*(m+1),1], marker=mark[m])

    plt.title('PCA projections', fontsize=44)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(['Cat', 'Dog','Airplane', 'All'], loc='best', fontsize=25)
    plt.savefig(path_pca)
    plt.show()
    plt.close()

    #K-means
    kmeans = KMeans(n_clusters=num_clusters, n_init='auto', verbose=5)
    LandS_labels = kmeans.fit(LandSPCA).labels_
    print(kmeans.inertia_)
    print([LandS_labels[:batch],LandS_labels[batch:2*batch], LandS_labels[2*batch:3*batch]])

    kfalse = KMeans(n_clusters=false_clus, n_init='auto')
    LandS_labels_false = kfalse.fit(LandSPCA).labels_
    #print([LandS_labels_false[:batch],LandS_labels_false[batch:2*batch], LandS_labels_false[2*batch:3*batch]])

    #Visualization of K-means
    path_kmeans = serv_results + 'cifar_' + more_name + str(layer) + '_kmeans.png'
    #fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(6, 6))
    plt.figure(figsize=(6,6))
    if LandSPCA.shape[1] == 1:
        #axs[0].scatter(LandSPCA, np.zeros(batch*4), c=LandS_labels_false)
        #axs[0].set_title("False Clusters")
        #axs[1].scatter(LandSPCA, np.zeros(batch*4), c=LandS_labels)
        #axs[1].set_title("True Clusters")
        plt.scatter(LandSPCA, np.zeros(batch*4), c=LandS_labels)
        plt.title("K-means, k=2", fontsize=44)
    else:
        #axs[0].scatter(LandSPCA[:,0], LandSPCA[:,1], c=LandS_labels_false)
        plt.scatter(LandSPCA[:,0], LandSPCA[:,1], c=LandS_labels)
        plt.title("K-means, k=2", fontsize=44)
        #axs[0].set_title("K-means, k=3", fontsize=40)
        #axs[1].scatter(LandSPCA[:,0], LandSPCA[:,1], c=LandS_labels)
        #axs[1].set_title("K-means, k=2", fontsize=40)
        #axs[0].set_xticks([])
    plt.tight_layout()
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.savefig(path_kmeans)
    plt.show()
    plt.close()

    #Confusion matrix
    truth = np.zeros(batch*num_samples)
    if num_samples == 4:
        for b in range(batch):
            truth[batch*3 + b] = 1
    else:
        for n in range(num_samples):
            for b in range(batch):
                truth[batch*n + b] = n

    # Prep
    k_labels = LandS_labels  # Get cluster labels
    k_labels_matched = np.empty_like(k_labels)

    # For each cluster label...
    for k in np.unique(k_labels):
        # ...find and assign the best-matching truth label
        match_nums = [np.sum((k_labels==k)*(truth==t)) for t in np.unique(truth)]
        k_labels_matched[k_labels==k] = np.unique(truth)[np.argmax(match_nums)]

    # Compute confusion matrix
    cm = confusion_matrix(truth, k_labels_matched)

    # Plot confusion matrix
    path_cm = serv_results + 'cifar_' + more_name + str(layer) + '_confusion.png'
    plt.imshow(cm,interpolation='none',cmap='Blues')
    for (i, j), z in np.ndenumerate(cm):
        plt.text(j, i, z, ha='center', va='center', fontsize=40)
    plt.xlabel("kmeans label", fontsize=25)
    plt.ylabel("truth label", fontsize=25)
    plt.title("Confusion matrix", fontsize=30)
    plt.yticks([])
    plt.xticks([])

    plt.savefig(path_cm)
    plt.show()
    plt.close()

    return LandS_labels



if __name__ == '__main__':
    serv_results = '/xxx/xxx/xxxx/xxx/xxxxxx/x'

    layers_unet = ["E1-ReLU1", "E1-ReLU2", "E2-ReLU1", "E2-ReLU2", "E3-ReLU1", "E3-ReLU2", "B-ReLU1", "B-ReLU2", "D1-UpReLU", "D1-ReLU1", "D1-ReLU2", "D2-UpReLU", "D2-ReLU1", "D2-ReLU2", "D3-UpReLU", "D3-ReLU1", "D3-ReLU2"]
    layers_cifar = ["ReLU11", "ReLU12", "ReLU21", "ReLU22", "ReLU31", "ReLU32", "ReLU33", "ReLU41", "ReLU42", "ReLU43", "ReLU51", "ReLU52", "ReLU53"]
    classes = ['Cat', 'Dog', 'Airplane', 'All']
    unets = ['M1', 'M2', 'M3', 'M4', 'RGB']

    resol = 700
    numkthlands = 10
    numLayers_unet = len(layers_unet)
    numLayers_cifar = len(layers_cifar)

    #distance('Landscapes_classes0', classes, numLayers_cifar, serv_results)
    #complexity0('Landscapes_classes0', numLayers_cifar, serv_results)

    #distance('Landscapes_classes00', classes, numLayers_cifar, serv_results)
    #complexity0('Landscapes_classes00', numLayers_cifar, serv_results)

    #distanceMean('Landscapes_catdogairplanecifar', classes, numLayers_cifar, serv_results, resol, numkthlands)

    #load_permutation('Landscapes_catdogairplanecifar', serv_results, resol, numLayers_cifar, numkthlands, nets=False, same_net=True)

    #complexity('Landscapes_unet13567rgb', numLayers_unet, serv_results, resol, numkthlands, num_net=3)

    #plot_avLandscapes_rowlayers_columnnets('Landscapes_unet13567rgb', resol, layers_unet, numkthlands, serv_results, unets)

    #plot_avLandscapes_rowlayers_columnnets('Landscapes_catdogairplanecifar', resol, layers_cifar, numkthlands, serv_results, classes)

    #num_clusters = 3; false_clus = 2; num_samples = 3; layer = 5
    #lands_kmeans = compute_kmeans('Landscapes_catdogairplanecifar', 32, resol, #numkthlands, numLayers_cifar, num_clusters, false_clus, num_samples, layer)

    num_clusters = 2; false_clus = 3; num_samples = 4; layer = 5
    lands_kmeans = compute_kmeans('Landscapes_catdogairplanecifar', 32, resol, numkthlands, numLayers_cifar, num_clusters, false_clus, num_samples, layer, more_name='all')