"""Performs experiments on U-Net/CRBD models:
- Permutation tests
- Average latent landscapes plot

Parameters
----------
resol : resolution of landscapes
numkthlands : number of k-th landscapes
layer_names : names of the layers studied (for the graphic)
unets : names of the CNN models (for the graphic)
dir_results : directory path where results are saved
file_name : name of landscapes file to load
"""

import PersLands as PL
import parameters

if __name__=='__main__':

    resol = parameters.RESOL
    numkthlands = parameters.NUMKTHLANDS
    layer_names = parameters.NAMES_LAYERS_UNET
    unets = parameters.UNETS
    dir_results = parameters.DIR_RESULTS
    
    numLayers = len(layer_names)

    file_name = 'landscapes_Multi1234Rgb'

    landscapes = PL.load_data(file_name)

    PL.perform_permutation_tests(landscapes, file_name, dir_results, resol, numLayers, numkthlands, nets=False)

    PL.plot_avLandscapes_rowlayers_columnnets(landscapes, file_name, resol, layer_names, numkthlands, dir_results, unets)