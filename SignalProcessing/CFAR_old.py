import numpy as np
from scipy.ndimage import convolve1d
from scipy.signal import convolve2d

"""
This file cotains varius CFAR detection algorithms.

"""

def CA_CFAR_2D_(input_val, guard_cells=4, window_size=8,mode="wrap", l_bound=200000, *argv, **kwargs):
 

    kernel = np.ones(1 + (2 * guard_cells) + (2 * window_size), dtype=input_val.dtype) / (2 * window_size)
    kernel[window_size:window_size + (2 * guard_cells) + 1] = 0
    noise_floor = convolve1d(input_val, kernel, mode=mode)
    threshold = noise_floor + l_bound
    

    return threshold, noise_floor

def CA_CFAR_2D(x, *argv, **kwargs):
    
    threshold, _ = CA_CFAR_2D_(x, *argv, **kwargs)
    ret = (x > threshold)
    return ret


def P_avg(P,N):
    return P/N
def alpha(N,P_FA):
    return(P_FA**(-1/N)-1)

def estimated_teshold(alpha,P):
    return alpha*np.abs(P)
def window_estimator(x,training_cells,training_area):    
    
    P_total = np.sum(np.abs(x))
    P_center_square =np.sum(np.abs(x[training_cells:x.shape[0]-training_cells,training_cells:x.shape[1]-training_cells]))
    P_traning_cells = np.abs(P_total - P_center_square)
    
    return P_avg(P_traning_cells,training_area)

def CFAR_2D(data, guard_cells, training_cells, PFA):
    
    
    window_size = guard_cells + training_cells
    
    window_area = (2*window_size+1)**2
    training_area = window_area - (2*window_size+1-2*training_cells)**2
    a = alpha(training_area, PFA)

    kernel = np.ones((1 + (2 * guard_cells) + (2 * training_cells),1 + (2 * guard_cells) + (2 * training_cells)), dtype=data.dtype)
    kernel[training_cells:training_cells + (2 * guard_cells) + 1,training_cells:training_cells + (2 * guard_cells) + 1] = 0
    
    res = convolve2d(data.copy(), kernel, mode='same',boundary="wrap")
    
    
    ret = (np.abs(data)>estimated_teshold(a,res))

    
    

    
    
            
    return ret


def CFAR_1D(data, guard_cells, training_cells, PFA):
    
    
        window_size = guard_cells + training_cells
        
        window_area = (2*window_size+1)**2
        training_area = training_cells*2
        a = alpha(training_area, PFA)

        kernel = np.ones((1 + (2 * guard_cells) + (2 * training_cells)), dtype=data.dtype)
        kernel[training_cells:training_cells + (2 * guard_cells) + 1] = 0
        
        res = convolve1d(data.copy(), kernel, mode='wrap')
        
        
        ret = (np.abs(data)>estimated_teshold(a,res))

        
        

        
        
                
        return ret