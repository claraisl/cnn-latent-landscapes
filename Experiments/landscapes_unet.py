import numpy as np
import scipy
import scipy.io as sio
import gudhi as gd
import gudhi.representations
import csv
import matplotlib.pyplot as plt
import pickle


import PersLands as PL

resol = 700
numkthlands = 10
LS = gd.representations.Landscape(num_landscapes=numkthlands, resolution=resol)
Ess = gd.representations.preprocessing.DiagramSelector(use = True, point_type='finite')
BC = gd.representations.vector_methods.BettiCurve(resolution=resol)
x = np.linspace(0,resol,num=resol)
names = ["E1-ReLU1", "E1-ReLU2", "E2-ReLU1", "E2-ReLU2", "E3-ReLU1", "E3-ReLU2", "B-ReLU1", "B-ReLU2", "D1-UpReLU", "D1-ReLU1", "D1-ReLU2", "D2-UpReLU", "D2-ReLU1", "D2-ReLU2", "D3-UpReLU", "D3-ReLU1", "D3-ReLU2"]

a0 = False
a1 = False
b = False
brgb = True
b8rgb = False

serv = '/xxx/xxx/xxxx/xxx/xxxxxx/x'
serv_results = '/xxx/xxx/xxxx/xxx/xxxxxx/x'


if a0 or a1:
    net_path = serv + 'ctivationCell_rgb.mat'
    dataMatrix = sio.loadmat(net_path)
    data = dataMatrix['activationCell']
    numLayers = data.shape[1]
    numImages = data[0][0].shape[1]
#a0
if a0:
    print('a0...')
    Landscapes_a0_2 = PL.a0(data, resol, numkthlands, LS, Ess, names, x, serv_results, 'a0_5_rgb', perform_pca=False, k=5)
    Landscapes_a0_3 = PL.a0(data, resol, numkthlands, LS, Ess, names, x, serv_results, 'a0_7_rgb', perform_pca=False, k=7)

    # Compare a0 for two different images computing the distance between each layer landscape, the distance of the whole network representation and the permutation test taking as samples each of the layer landscape
    print('Test...')
    distance = np.zeros(numLayers+1)
    for m in range(numLayers):
        xx = Landscapes_a0_2[m,:]
        y = Landscapes_a0_3[m,:]
        distance[m] = np.sqrt(np.sum((xx-y)**2))
    distance[m+1] = np.sqrt(np.sum((Landscapes_a0_2.flatten() - Landscapes_a0_3.flatten())**2))
    lands = []
    lands.append(Landscapes_a0_2)
    lands.append(Landscapes_a0_3)
    p_value = PL.permutation_nets(lands, serv_results, 'a0_rgb57_layer_pvalue_dist_layerwiseWhole')
    name_str = serv_results + 'a0_rgb57_layer_pvalue_dist_layerwiseWhole' + '.csv'
    with open(name_str, mode='a', newline='') as pvalue_file:
        compare_writer = csv.writer(pvalue_file, delimiter=',')
        compare_writer.writerow(distance)

    # Compute complexity for the latent representation of two different images
    print('Complexity...')
    complex2 = np.zeros(numLayers)
    complex3 = np.zeros(numLayers)
    for m in range(numLayers):
        complex2[m] = PL.complexity(Landscapes_a0_2[m,:])
        complex3[m] = PL.complexity(Landscapes_a0_3[m,:])
    xaxis = np.arange(0,numLayers)
    plt.plot(xaxis,complex2, '-o')
    plt.plot(xaxis,complex3, '-o')
    path_complex = serv_results + 'a0_57_rgb' + '_complex.png'
    plt.savefig(path_complex)
    plt.show()
    plt.close()

#a1
if a1:
    print('a1...')
    Landscapes_a1 = PL.a1(data, resol, numkthlands , LS, Ess, names, x, serv_results, 'a1_rgb', perform_pca=True)

    print('Test...')
    p_value_a1 = PL.permutation(Landscapes_a1, numLayers, resol, numkthlands, serv_results, 'a1_rgb_pvalue')

#b) Comparing different U-Net models. This returns the permutation test of each network with itself layer by layer (a1_unetx_pvalue), the test between whole networks ('b8Rgb_pvalue'), and the test between the layers within a network('b_8Rgb_layer_pvalue')
if b:
    print('b...')
    Landscapes_8rgb = []
    for st in ['unet1', 'unet3', 'unet5', 'unet6', 'unet7', 'rgb']:
        net_path = serv + 'ctivationCell_' + st + '.mat'
        dataMatrix = sio.loadmat(net_path)
        data = dataMatrix['activationCell']
        numLayers = data.shape[1]
        numImages = data[0][0].shape[1]
        file_name = 'b_unet'+st
        Landscapes_b = PL.a1(data, resol, numkthlands , LS, Ess, names, x, serv_results, file_name, perform_pca=False)
        #pvalue of all layers for all unetx
        #print('Test...')
        #pvalue_path = 'a1_' + st + '_pvalue'
        #p_value_a1 = PL.permutation(Landscapes_b, numLayers, resol, numkthlands, serv_results, pvalue_path)
        Landscapes_8rgb.append(Landscapes_b) 
    print('Test...')
    p_value_b_8bands = PL.permutation_nets(Landscapes_8rgb, serv_results, 'b_8bandsRgb_u1u3u5u6u7rgb_pvalue')
    PL.permutation_layer_nets(Landscapes_8rgb, numLayers, resol, numkthlands, serv_results, 'b_8bandsRgb_u1u3u5u6u7rgb_layer_pvalue')

#b) Comparing U-Net Multi-4 and RGB
if brgb:
    print('b...')
    Landscapes_8rgb_nets = []
    for st in ['unet7', 'rgb']:
        net_path = serv + 'ctivationCell_' + st + '.mat'
        dataMatrix = sio.loadmat(net_path)
        data = dataMatrix['activationCell']
        numLayers = data.shape[1]
        numImages = data[0][0].shape[1]
        file_name = 'b_'+st
        Landscapes_b = PL.a1(data, resol, numkthlands , LS, Ess, names, x, serv_results, file_name, perform_pca=False)
        Landscapes_8rgb_nets.append(Landscapes_b)
    #print('Test...')
    #p_value_b_8rgb = PL.permutation_nets(Landscapes_8rgb_nets, serv_results, 'b_unet3rgb_pvalue')
    #PL.permutation_layer_nets(Landscapes_8rgb_nets, numLayers, resol, numkthlands, serv_results, 'b_unet3rgb_layer_pvalue')

    # Store data (serialize)
    your_data = Landscapes_8rgb_nets
    name_dir = '/POOL/data/clara/lands' + 'Landscapes_8rgb_nets' + '.pickle'
    with open('Landscapes_8rgb_nets.pickle', 'wb') as handle:
        pickle.dump(your_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Load data (deserialize)
    #with open('Landscapes_8rgb_nets.pickle', 'rb') as handle:
    # unserialized_data = pickle.load(handle)

# Compare multi layer by layer between them and with rgb
if b8rgb:
    print('b...')
    Landscapes_8netsRgb = []
    name_nets = ['unet1', 'unet3', 'unet5', 'unet6', 'unet7', 'rgb']
    for st in name_nets:
        net_path = serv + 'ctivationCell_' + st + '.mat'
        dataMatrix = sio.loadmat(net_path)
        data = dataMatrix['activationCell']
        numLayers = data.shape[1]
        numImages = data[0][0].shape[1]
        file_name = 'b_'+st
        Landscapes_b = PL.a1(data, resol, numkthlands , LS, Ess, names, x, serv_results, file_name, perform_pca=False)
        Landscapes_8netsRgb.append(Landscapes_b)
    print('Test...')
    PL.permutation_layer_nets(Landscapes_8netsRgb, numLayers, resol, numkthlands, serv_results, 'b_unetRgb_layer_pvalue')