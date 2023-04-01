import numpy as np
import gudhi as gd
import gudhi.representations
import scipy
import csv
import pickle

def persistenceDiagram(data, radius=100, max_dim=2):
    skeleton = gd.RipsComplex(points = data, max_edge_length = radius);
    Rips_simplex_tree= skeleton.create_simplex_tree(max_dimension = max_dim);
    BarCodes_Rips =Rips_simplex_tree.persistence();
    I0 = Rips_simplex_tree.persistence_intervals_in_dimension(0)
    I1 = Rips_simplex_tree.persistence_intervals_in_dimension(1)
    return I0, I1

def bottleneck(dgm1, dgm2):
    Ess = gd.representations.preprocessing.DiagramSelector(use = True, point_type='finite')
    dgm1_NoEss = Ess.fit_transform([dgm1])
    dgm2_NoEss = Ess.fit_transform([dgm2])
    dist = gd.bottleneck_distance(dgm1_NoEss[0], dgm2_NoEss[0])
    return dist

def load_data(name):
    # Load data (deserialize)
    print('Loading...')
    name_str = name + '.pickle'
    with open(name_str, 'rb') as handle:
        landscapes = pickle.load(handle)

    return landscapes

def average_landscape(landscapes):
    return np.mean(landscapes, axis=0) # dimension = 1 x (resol * num_kthlandscapes)

def _inner_product(x, y):
    if len(x.shape) == 2:
        return np.sum(x*y, axis=-1)
    else:
        return np.sum(x*y)

def statistic(*samples, axis):
    return np.linalg.norm(np.mean(samples[0], axis=axis) - np.mean(samples[1], axis=axis), axis=-1)

def statistic2(*samples, axis):
    return _inner_product(np.mean(samples[0], axis=axis),np.mean(samples[1], axis=axis))

def permutation(Landscapes, numLayers, resol, numkthlands, serv_results, name):
    p_value = np.zeros((numLayers,numLayers));
    for m in range(numLayers):
        xx = Landscapes[:,m*resol*numkthlands:(m+1)*resol*numkthlands]
        for n in range(m+1,numLayers):
            y = Landscapes[:,n*resol*numkthlands:(n+1)*resol*numkthlands]
            res = scipy.stats.permutation_test((xx, y), statistic, vectorized=True, axis=0)
            p_value[m,n] = res.pvalue
        name_str = serv_results + name + '.csv'
        with open(name_str, mode='a', newline='') as pvalue_file:
            compare_writer = csv.writer(pvalue_file, delimiter=',')
            compare_writer.writerow(p_value[m,:])
    return p_value

def permutation_nets(Landscapes_nets, serv_results, name):
    num_nets = len(Landscapes_nets)
    p_value = np.zeros((num_nets,num_nets))
    for m in range(num_nets):
        net1 = Landscapes_nets[m]
        for n in range(m+1, num_nets):
            net2 = Landscapes_nets[n]
            res = scipy.stats.permutation_test((net1, net2), statistic, vectorized=True, n_resamples=2000, axis=0)
            p_value[m,n] = res.pvalue
        name_str = serv_results + name + '.csv'
        with open(name_str, mode='a', newline='') as pvalue_file:
            compare_writer = csv.writer(pvalue_file, delimiter=',')
            compare_writer.writerow(p_value[m,:])
    return p_value

def complexity(x):
    return np.sqrt(_inner_product(x, x))

# Compare different networks layer by layer. Landscapes is a list of the landscapes of each network (the matrix output of a1)
def permutation_layer_nets(Landscapes, numLayers, resol, numkthlands, serv_results, name):
    name_str = serv_results + name + '.csv'
    for r in range(len(Landscapes)):
        net1 = Landscapes[r]
        for n in range(r+1,len(Landscapes)):
            net2 = Landscapes[n]
            p_value = np.zeros(numLayers)
            for m in range(numLayers):
                xx = net1[:,m*resol*numkthlands:(m+1)*resol*numkthlands]
                y = net2[:,m*resol*numkthlands:(m+1)*resol*numkthlands]
                res = scipy.stats.permutation_test((xx, y), statistic, vectorized=True, axis=0)
                p_value[m] = res.pvalue
            with open(name_str, mode='a', newline='') as pvalue_file:
                compare_writer = csv.writer(pvalue_file, delimiter=',')
                compare_writer.writerow(p_value)