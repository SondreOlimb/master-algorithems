import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from scipy import ndimage


def Plot(data,
vmin=-40,vmax=0,
labels = {
    "x_label":"Velocity [knots]",
    "y_label":"Range [m]",
    "title": "Radar MRDM"

}):


    #data_fft_plot = cv2.GaussianBlur(np.abs(data_ma), (3, 3),sigmaX=1,sigmaY=1)
    #data_fft_plot = ndimage.gaussian_filter(data_fft_plot,sigma=1,mode=mode)
    #sns.set(rc={'figure.figsize':(20,15)})

    #test = np.max(np.abs(MRDM[:500]),axis=0)
    plt.figure(figsize=(10,10))

    #test[test>40] =0
    rotated_img = ndimage.rotate(data,90)
    #rotated_img = radar_cube[0] 
    #rotated_img = np.flip(test,axis=0)
    rotated_img =20*np.log10(np.abs(rotated_img)) # We rotate the image so the x axis is the velocity

    max_val = np.max(rotated_img)
    rotated_img = rotated_img-max_val
    #rotated_img = cv2.GaussianBlur(rotated_img, (3, 3),sigmaX=1,sigmaY=1)
    #plt.imshow(rotated_img,cmap="plasma", vmin=plot_min_doppler,vmax=plot_max_doppler)
    plt.imshow(rotated_img,cmap="plasma", vmin=vmin,vmax=vmax)
    # rms = 10*np.log(np.sqrt(np.mean(np.abs(rotated_img[130:135,80:100])**2)))
    # peak = 10*np.log(np.abs(rotated_img[137,87]))
    # snr = peak-rms
    # print("Peak:",peak)
    # print("Side loab:", 10*np.log(np.abs(rotated_img[137,85])))
    # print("RMS:",rms)
    # print("SNR:",snr) 
    #plt.xlim(80,94)
    #plt.yticks(np.linspace(0,256,9),labels=np.round(np.linspace(255*0.785277,0,9)),size =15)



    #plt.xticks(np.linspace(0,256,7),labels=np.round(np.linspace(-0.127552440715*127,0.127552440715*127,7),2),size =15)
    cbar  = plt.colorbar()
    cbar.set_label('Mangnitude [dB]',fontdict = {'fontsize' : 20})
    cbar.ax.tick_params(labelsize=15) 
    plt.xlabel(labels["x_label"],fontdict = {'fontsize' : 20})
    plt.ylabel(labels["y_label"],fontdict = {'fontsize' : 20})
    plt.title(labels["title"],fontdict = {'fontsize' : 30})
    plt.grid(False)
    #plt.savefig("plots/results/radar_MRDM.svg",format="svg")

def PlotCFAR(data,labels = {
    "x_label":"Velocity [knots]",
    "y_label":"Range [m]",
    "title": "CFAR"

}):


    
    plt.figure(figsize=(10,10))

    
    rotated_img = ndimage.rotate(data,90)

    plt.imshow(rotated_img,cmap="plasma")
    
    plt.yticks(np.linspace(0,256,9),labels=np.round(np.linspace(255*0.785277,0,9)),size =15)



    plt.xticks(np.linspace(0,256,7),labels=np.round(np.linspace(-0.127552440715*127,0.127552440715*127,7),2),size =15)
    

    plt.xlabel(labels["x_label"],fontdict = {'fontsize' : 20})
    plt.ylabel(labels["y_label"],fontdict = {'fontsize' : 20})
    plt.title(labels["title"],fontdict = {'fontsize' : 30})
    plt.grid(False)
    