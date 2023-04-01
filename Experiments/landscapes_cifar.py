import numpy as np
import scipy
import scipy.io as sio
import gudhi as gd
import gudhi.representations
import csv
import matplotlib.pyplot as plt


import PersLands as PL

resol = 700
numkthlands = 10
batch = 32
LS = gd.representations.Landscape(num_landscapes=numkthlands, resolution=resol)
Ess = gd.representations.preprocessing.DiagramSelector(use = True, point_type='finite')
BC = gd.representations.vector_methods.BettiCurve(resolution=resol)
x = np.linspace(0,resol,num=resol)
names = ["ReLU11", "ReLU12", "ReLU21", "ReLU22", "ReLU31", "ReLU32", "ReLU33", "ReLU41", "ReLU42", "ReLU43", "ReLU51", "ReLU52", "ReLU53"]

a0 = False
a1 = False
b = False
a00 = False
a11 = False
bb = True
a0a00 = False

serv = '/xxx/xxx/xxxx/xxx/xxxxxx/x'
serv_results = '/xxx/xxx/xxxx/xxx/xxxxxx/x'

if a0 or a1 or a00 or a11:
    net_path = serv + 'ctivationCell_cifar.mat'
    dataMatrix = sio.loadmat(net_path)
    data = dataMatrix['activationCell']
    numLayers = data.shape[1]
    numImages = data[0][0].shape[1]
#a0)
if a0:
    print('a0...')
    Landscapes_a0 = PL.a0(data, resol, numkthlands, LS, Ess, names, x, serv_results, 'a0_cifar', perform_pca=False)

#a1)
if a1:
    print('a1...')
    Landscapes_a1 = PL.a1(data, resol, numkthlands , LS, Ess, names, x, serv_results, 'a1_cifar', perform_pca=True)

    print('Test...')
    p_value_a1 = PL.permutation(Landscapes_a1, numLayers, resol, numkthlands,serv_results, 'a1_cifar_pvalue')

# Compare latent representations for a single image (a0) or enough of the (a00)
if a0a00:
    print('a0a00...')
    Landscapes_classes0 = []
    Landscapes_classes00 = []
    classes = ['cat', 'dog', 'airplane']
    for st in classes:
        net_path = serv + 'ctivationCell_cifar_' + st + '.mat'
        dataMatrix = sio.loadmat(net_path)
        data = dataMatrix['activationCell']
        numLayers = data.shape[1]
        numImages = data[0][0].shape[1]
        file_name0 = 'a0_'+st
        file_name00 = 'a00_'+st
        Landscapes_a0 = PL.a0(data, resol, numkthlands , LS, Ess, names, x, serv_results, file_name0, perform_pca=False)
        Landscapes_a00 = PL.a00(data, resol, numkthlands , LS, Ess, names, x, serv_results, file_name00, perform_pca=False)
        Landscapes_classes0.append(Landscapes_a0)
        Landscapes_classes00.append(Landscapes_a00)

    # Different classes computing the distance between each layer landscape, the distance of the whole network representation
    print('Distance...')
    for st in range(len(classes)):
        net01 = Landscapes_classes0[st]
        net001 = Landscapes_classes00[st]
        for stt in range(st+1, len(classes)):
            net02 = Landscapes_classes0[stt]
            net002 = Landscapes_classes00[stt]
            distance0 = np.zeros(numLayers+1)
            distance00 = np.zeros(numLayers+1)
            for m in range(numLayers):
                xx0 = net01[m,:]
                y0 = net02[m,:]
                distance0[m] = np.sqrt(np.sum((xx0-y0)**2))

                xx00 = net001[m,:]
                y00 = net002[m,:]
                distance00[m] = np.sqrt(np.sum((xx00-y00)**2))
            distance0[m+1] = np.sqrt(np.sum((net01.flatten() - net02.flatten())**2))
            distance00[m+1] = np.sqrt(np.sum((net001.flatten() - net002.flatten())**2))
            name_str0 = serv_results + 'a0_distance_catdogairplane' + '.csv'
            name_str00 = serv_results + 'a0_distance_catdogairplane' + '.csv'
            with open(name_str0, mode='a', newline='') as dist_file:
                compare_writer = csv.writer(dist_file, delimiter=',')
                compare_writer.writerow(distance0)
            with open(name_str00, mode='a', newline='') as dist_file:
                compare_writer = csv.writer(dist_file, delimiter=',')
                compare_writer.writerow(distance00)

#b) Comparing different classes
if b:
    print('b...')
    Landscapes_nets = []
    for st in ['', '_cat', '_dog', '_airplane']:
        net_path = serv + 'ctivationCell_cifar' + st + '.mat'
        dataMatrix = sio.loadmat(net_path)
        data = dataMatrix['activationCell']
        numLayers = data.shape[1]
        numImages = data[0][0].shape[1]
        file_name = 'b_cifar'+st
        Landscapes_b = PL.a1(data, resol, numkthlands , LS, Ess, names, x, serv_results, file_name, perform_pca=False)
        Landscapes_nets.append(Landscapes_b)
    print('Test...')
    p_value_b_classes = PL.permutation_nets(Landscapes_nets, serv_results, 'b_multicatdogairplane_pvalue')
    p_value_b_classes_layerwise = PL.permutation_layer_nets(Landscapes_nets, numLayers, resol, numkthlands, serv_results, 'b_classes_multicatdogairplane_pvalue')

#a00) CIFAR-10
if a00:
    print('a00...')
    Landscapes_a00 = PL.a00(data, resol, numkthlands, LS, Ess, names, x, serv_results, 'a00_cifar', perform_pca=False)

#a11) CIFAR-10
if a11:
    print('a11...')
    Landscapes_a11 = PL.a11(data, resol, numkthlands, batch, LS, Ess, names, x, serv_results, 'a11_cifar_cat', perform_pca=True)

    print('Test...')
    p_value_a11 = PL.permutation(Landscapes_a11, numLayers, resol, numkthlands,serv_results, 'a11_cifar_cat_pvalue')

#b) CIFAR-10 with a1
if bb:
    print('bb...')
    Landscapes_nets = []
    name_nets = ['', '_cat', '_dog', '_airplane']
    for st in name_nets:
        net_path = serv + 'ctivationCell_cifar' + st + '.mat'
        dataMatrix = sio.loadmat(net_path)
        data = dataMatrix['activationCell']
        numLayers = data.shape[1]
        numImages = data[0][0].shape[1]
        file_name = 'bb_cifar'+st
        Landscapes_bb = PL.a11(data, resol, numkthlands, batch, LS, Ess, names, x, serv_results, file_name, perform_pca=False)
        #pvalue of all layers for all nets
        print('Test...')
        pvalue_path = 'a11_cifar' + st + '_pvalue'
        p_value_a1 = PL.permutation(Landscapes_bb, numLayers, resol, numkthlands, serv_results, pvalue_path)
        Landscapes_nets.append(Landscapes_bb)
    print('Test...')
    #p_value_bb_8bands = PL.permutation_nets(Landscapes_nets, serv_results, 'bb_multicatdogairplane_pvalue')
    p_value_b_classes_layerwise = PL.permutation_layer_nets(Landscapes_nets, numLayers, resol, numkthlands, serv_results, 'bb_classes_multicatdogairplane_pvalue')