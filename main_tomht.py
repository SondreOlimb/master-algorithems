from SignalProcessing import RangeCompression,DopplerProcessing,CFAR, Clustering
import Dataloader
import Plot
import numpy as np
from mmwave import dsp
from matplotlib import cm
import pandas as pd
import numpy as np
import seaborn as sns; 
from matplotlib.colors import  LogNorm
from scipy.signal import butter, lfilter, freqz,detrend
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
from scipy import constants
from scipy import ndimage
from PIL import Image, ImageFilter
from scipy.ndimage import convolve1d

from operator import itemgetter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from SignalProcessing.algo import algorithm_cfar_1d,algorithm_cfar_1_v2
from Tracking.Initiator import Initiator
from Tracking.TOMHT import TOMHT

no_reflector = "felttest1/Record_2022-09-27_14-34-48/Record_2022-09-27_14-34-48.bin"
best_restult = "felttest1/Record_2022-09-27_14-14-14/Record_2022-09-27_14-14-14.bin"
munkholm = "felttest2/Record_2022-09-28_15-13-15/Record_2022-09-28_15-13-15.bin"

data_IF =Dataloader.LoadData("best_restult.npy")

old=  RangeCompression.RangeCompression(data_IF[100,0],axis=1)
old, _ = DopplerProcessing.DopplerProcessing(old, axis=0,isClutterRemoval=True, removeArtifacts=False)
old= np.abs(old)
argArtifacts = np.nonzero(old >4*np.mean(old[:,50]))

cfar_arr=[]
cfar_not_arr=[]
for i in range(200,len(data_IF)):
    #not_form = algorithm_cfar_1_v2(data_IF[i,0],argArtifacts)
    form, not_form = algorithm_cfar_1d(data_IF[i,0],argArtifacts)
    cfar_arr.append(form)
    cfar_not_arr.append(not_form)

fig, ax = plt.subplots(1, 2, figsize=(10, 10))
initiat = Initiator(3,5)
MHT = TOMHT()
for i, det in enumerate(cfar_arr):
    print("\n",f"############ {i} #############")
    
    
    plt.title(f"Frame {i}")
    #rotate_img =ndimage.rotate(cfar_not_arr[i], 90)
    
    print(det.shape)
    tracks, unused_measurments = MHT.main(det)
   
    if(len(MHT.tracks)>0):
        ax[1].plot([det[0] for det in tracks], [det[1] for det in tracks], 'ro',label="Tracking")
    detections,tar = initiat.main(unused_measurments)
    rotate_img = cfar_not_arr[i]
    if(len(detections)>0):
        MHT.new_track(tar)

        ax[0].plot([det[0] for det in detections], [det[1] for det in detections], 'ro',label="Tracking")
    # display the image
    #ax.set_title(f"Frame {i}")
    ax[0].imshow(rotate_img,label="Detections")
    ax[1].imshow(rotate_img,label="Detections")
    ax[1].set_title("MHT")
    ax[0].set_title("NN tracking")
    #plt.legend()
    ax[1].set_xticks(np.linspace(0,256,9),labels=np.round(np.linspace(0,255*0.785277,9)),size =10)
    ax[1].set_yticks(np.linspace(0,256,7),labels=np.round(np.linspace(-0.127552440715*127,0.127552440715*127,7),2),size =10)
    ax[0].set_xticks(np.linspace(0,256,9),labels=np.round(np.linspace(0,255*0.785277,9)),size =10)
    ax[0].set_yticks(np.linspace(0,256,7),labels=np.round(np.linspace(-0.127552440715*127,0.127552440715*127,7),2),size =10)
    plt.draw()
    plt.pause(0.01)
    #print("displayed")
    #input("Press Enter to continue...")
    ax[0].cla() # clear axis for the next frame
    ax[1].cla() # clear axis for the next frame
    