import numpy as np

from .CFAR_old import CA_CFAR_2D
from .RangeCompression import RangeCompression
from .DopplerProcessing import DopplerProcessing
from .CFAR import CFAR_1D

def algorithm_cfar_1d(data,argArtifacts):
    range_cube =  RangeCompression(data,axis=1)
    linear, mag = DopplerProcessing(range_cube, axis=0,isClutterRemoval=True)
    
    linear[argArtifacts] = 1e-10
    detections_map,P_detections,detections, detections_cord,det_tuples = CFAR_1D(np.abs(linear).copy(), 4, 8, 0.01)
    return detections_cord , detections_map

def algorithm_cfar_1_v2(data,argArtifacts):
    range_cube =  RangeCompression(data,axis=1)
    linear, mag = DopplerProcessing(range_cube, axis=0,isClutterRemoval=True)
    
    linear[argArtifacts] = 1e-10
    detections_cord = CA_CFAR_2D(np.abs(linear).copy(), 4, 8)
    return detections_cord