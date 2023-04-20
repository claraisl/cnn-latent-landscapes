"""List of parameters and others
"""
import gudhi as gd
import gudhi.representations

RESOL = 700
NUMKTHLANDS = 10
BATCH = 32

LS = gd.representations.Landscape(num_landscapes=NUMKTHLANDS, resolution=RESOL)
Ess = gd.representations.preprocessing.DiagramSelector(use = True, point_type='finite')

DIR_DATA = '/xxx/xxx/xxxx/xxx/xxxxxx/x'
DIR_RESULTS = '/xxx/xxx/xxxx/xxx/xxxxxx/'

NAMES_LAYERS_UNET = ["E1-ReLU1", "E1-ReLU2", "E2-ReLU1", "E2-ReLU2", "E3-ReLU1", "E3-ReLU2", "B-ReLU1", "B-ReLU2", "D1-UpReLU", "D1-ReLU1", "D1-ReLU2", "D2-UpReLU", "D2-ReLU1", "D2-ReLU2", "D3-UpReLU", "D3-ReLU1", "D3-ReLU2"]
NAMES_LAYERS_VGG = ["ReLU11", "ReLU12", "ReLU21", "ReLU22", "ReLU31", "ReLU32", "ReLU33", "ReLU41", "ReLU42", "ReLU43", "ReLU51", "ReLU52", "ReLU53"]

CLASSES = ['Cat', 'Dog', 'Airplane', 'All']
UNETS = ['M1', 'M2', 'M3', 'M4', 'RGB']