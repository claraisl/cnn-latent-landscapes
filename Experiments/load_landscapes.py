import numpy as np
import scipy
import scipy.io as sio
import gudhi as gd
import gudhi.representations
import csv
import matplotlib.pyplot as plt
import pickle
import PersLands as PL

# Compute the latent landscapes for all the activation layers and save the data for future experiments

resol = 700
numkthlands = 10
batch = 32

LS = gd.representations.Landscape(num_landscapes=numkthlands, resolution=resol)
Ess = gd.representations.preprocessing.DiagramSelector(use = True, point_type='finite')
BC = gd.representations.vector_methods.BettiCurve(resolution=resol)
x = np.linspace(0,resol,num=resol)

names_unet = ["E1-ReLU1", "E1-ReLU2", "E2-ReLU1", "E2-ReLU2", "E3-ReLU1", "E3-ReLU2", "B-ReLU1", "B-ReLU2", "D1-UpReLU", "D1-ReLU1", "D1-ReLU2", "D2-UpReLU", "D2-ReLU1", "D2-ReLU2", "D3-UpReLU", "D3-ReLU1", "D3-ReLU2"]
names_vgg = ["ReLU11", "ReLU12", "ReLU21", "ReLU22", "ReLU31", "ReLU32", "ReLU33", "ReLU41", "ReLU42", "ReLU43", "ReLU51", "ReLU52", "ReLU53"]

serv = '/xxx/xxx/xxxx/xxx/xxxxxx/x'
serv_results = '/xxx/xxx/xxxx/xxx/xxxxxx/x'

unet_a0 = False
unet_a1 = False
vgg_net = True
unet_bigB = False

if unet_a0:
    print('Unets a0...')
    Landscapes_rgb_img2357 = []
    net_path = serv + 'ctivationCell_rgb.mat'
    dataMatrix = sio.loadmat(net_path)
    data = dataMatrix['activationCell']
    numLayers = data.shape[1]
    numImages = data[0][0].shape[1]
    for st in [2, 3, 5, 7]:
        Landscapes_a0 = PL.a0(data, resol, numkthlands, LS, Ess, names_unet, x, serv_results, 'a0_5_rgb', perform_pca=False, k=st)
        Landscapes_rgb_img2357.append(Landscapes_a0)

    with open('Landscapes_rgb_img2357.pickle', 'wb') as handle:
        pickle.dump(Landscapes_rgb_img2357, handle, protocol=pickle.HIGHEST_PROTOCOL)

if unet_a1:
    print('Unets a1...')
    Landscapes_unet13567rgb = []
    for st in ['unet1', 'unet3', 'unet5', 'unet6', 'unet7', 'rgb']:
        net_path = serv + 'ctivationCell_' + st + '.mat'
        dataMatrix = sio.loadmat(net_path)
        data = dataMatrix['activationCell']
        numLayers = data.shape[1]
        numImages = data[0][0].shape[1]
        file_name = 'b_unet'+st
        Landscapes_b = PL.a1(data, resol, numkthlands , LS, Ess, names_unet, x, serv_results, file_name)
        Landscapes_unet13567rgb.append(Landscapes_b)

    # Store data (serialize)
    with open('Landscapes_unet13567rgb.pickle', 'wb') as handle:
            pickle.dump(Landscapes_unet13567rgb, handle, protocol=pickle.HIGHEST_PROTOCOL)

if unet_bigB:
    print('Unets a1 bigger batch...')
    Landscapes_unet37rgb_bigB = []
    for st in ['unet3', 'unet7', 'rgb']:
        net_path = serv + 'ctivationCell_' + st + 'B12.mat'
        dataMatrix = sio.loadmat(net_path)
        data = dataMatrix['activationCell']
        numLayers = data.shape[1]
        numImages = data[0][0].shape[1]
        file_name = 'b_'+st
        Landscapes_b = PL.a1(data, resol, numkthlands , LS, Ess, names_unet, x, serv_results, file_name)
        Landscapes_unet37rgb_bigB.append(Landscapes_b)

    # Store data (serialize)
    with open('Landscapes_unet37rgb_bigB.pickle', 'wb') as handle:
        pickle.dump(Landscapes_unet37rgb_bigB, handle, protocol=pickle.HIGHEST_PROTOCOL)

if vgg_net:
    print('vgg...')
    #Landscapes_classes0 = []
    #Landscapes_classes00 = []
    Landscapes_truckbirdauto = []
    classes = ['truck', 'bird', 'auto']
    for st in classes:
        net_path = serv + 'ctivationCell_cifar_' + st + '.mat'
        dataMatrix = sio.loadmat(net_path)
        data = dataMatrix['activationCell']
        numLayers = data.shape[1]
        numImages = data[0][0].shape[1]

        #file_name0 = 'a0_'+st
        #file_name00 = 'a00_'+st
        file_name = 'bb_cifar'+st

        #Landscapes_a0 = PL.a0(data, resol, numkthlands , LS, Ess, names_vgg, x, serv_results, file_name0, perform_pca=False)
        #Landscapes_a00 = PL.a00(data, resol, numkthlands , LS, Ess, names_vgg, x, serv_results, file_name00, perform_pca=False)
        Landscapes_bb = PL.a11(data, resol, numkthlands, batch, LS, Ess, names_vgg, x, serv_results, file_name, perform_pca=False)

        #Landscapes_classes0.append(Landscapes_a0)
        #Landscapes_classes00.append(Landscapes_a00)
        Landscapes_truckbirdauto.append(Landscapes_bb)

    #net_path = serv + 'ctivationCell_cifar.mat'
    #dataMatrix = sio.loadmat(net_path)
    #data = dataMatrix['activationCell']
    #numLayers = data.shape[1]
    #numImages = data[0][0].shape[1]
    #file_name = 'bb_cifar'
    #Landscapes_bb = PL.a11(data, resol, numkthlands, batch, LS, Ess, names_vgg, x, serv_results, file_name, perform_pca=False)
    #Landscapes_catdogairplanecifar.append(Landscapes_bb)


    # Store data (serialize)
    #with open('Landscapes_classes0.pickle', 'wb') as handle:
    #    pickle.dump(Landscapes_classes0, handle, protocol=pickle.HIGHEST_PROTOCOL)

    #with open('Landscapes_classes00.pickle', 'wb') as handle:
    #    pickle.dump(Landscapes_classes00, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('Landscapes_truckbirdauto.pickle', 'wb') as handle:
        pickle.dump(Landscapes_truckbirdauto, handle, protocol=pickle.HIGHEST_PROTOCOL)