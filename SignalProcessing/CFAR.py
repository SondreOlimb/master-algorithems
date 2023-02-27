from scipy.ndimage import convolve1d
import numpy as np

def P_avg(P,N):
    return P
def alpha(N,P_FA):
    return(P_FA**(-1/N)-1)

def estimated_teshold(alpha,P):
    return alpha*np.abs(P)


def CFAR_1D(data, guard_cells, training_cells, PFA):
    
    
    window_size = guard_cells + training_cells
    
    window_area = (2*window_size+1)**2
    training_area = training_cells*2
    a = alpha(training_area, PFA)

    kernel = np.ones((1 + (2 * guard_cells) + (2 * training_cells)), dtype=data.dtype)
    kernel[training_cells:training_cells + (2 * guard_cells) + 1] = 0
    
    res = convolve1d(data.copy(), kernel, mode='wrap')
    
    
    ret = (np.abs(data)>estimated_teshold(a,res))
    
    detections = np.argwhere(ret==True)
    #Transform cords to range and doppler
    #print(detections)
    

    detections = detections[detections[:,0] < 120,:] #delete all irelevant detections
    detections_cord =  detections.copy()
    detections[:,1] = detections[:,1]*0.785277
    detections[:,0] = (128-detections[:,0])*-0.12755
    
    det_tuples = [(i[1],i[0]) for i in detections] 

    
    
    

    
    
            
    return ret,data*ret,detections,detections_cord,det_tuples