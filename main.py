from collections import deque
import time

from scipy import ndimage
import Dataloader
from SignalProcessing import RangeCompression,DopplerProcessing,CFAR, Clustering
import numpy as np
from SignalProcessing.algo import algorithm_cfar_1d,algorithm_cfar_1_v2
from Tracking.KF import KalmanFilter
from Tracking.NN import NearestNeighbor
from Tracking.Utils import Track, TrackMaintinance
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
import cv2 as cv

no_reflector = "felttest1/Record_2022-09-27_14-34-48/Record_2022-09-27_14-34-48.bin"
best_restult = "felttest1/Record_2022-09-27_14-14-14/Record_2022-09-27_14-14-14.bin"
munkholm = "felttest2/Record_2022-09-28_15-13-15/Record_2022-09-28_15-13-15.bin"



fig = plt.figure(figsize=(10,10))





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

import cv2
import matplotlib.pyplot as plt

# create a figure to display the image
fig, ax = plt.subplots(1, 1, figsize=(10, 10))


#loop to capture and display images
# for i, img in enumerate(cfar_not_arr):
#     rotate_img =ndimage.rotate(img, 90)
    
#     # display the image
#     ax.set_title(f"Frame {i}")
#     ax.imshow(rotate_img)
#     plt.draw()
#     plt.pause(0.001)
#     ax.cla() # clear axis for the next frame




# Example usage
dt = 50*1e-3 # time step (in seconds)
x_init = np.array([[200], [80]])  # initial state (position, velocity)
P_init = np.diag([10, 10])  # initial covariance matrix

Q = np.diag([0.1, 0.1])  # process noise covariance
R = np.diag([[1,1],[1, 1]])  # measurement noise covariance

track_id = 0
# Process measurement data
tracks = {}
for t,detections in enumerate( cfar_arr):
    print(f"############### {t} ###############")
    matches, unmatched_tracks, unmatched_observations = NearestNeighbor(detections, tracks, 10)
    
    
    if(len(unmatched_observations)>0):
        for obs in unmatched_observations:
            
            track_id += 1
            kf = KalmanFilter(dt, np.array([detections[obs][1],detections[obs][0]]), P_init, Q, R)
            
            trk = Track(track_id, detections[obs][1], detections[obs][0], kf, 1, deque([1]))
            tracks[track_id] = trk
   
    if(len(matches)>0):
        for trk_id, obs in matches:
            tracks[trk_id].update(detections[obs][1], detections[obs][0],1)
            tracks[trk_id].kalman_filter.predict()
            tracks[trk_id].kalman_filter.update(np.array([detections[obs][1],detections[obs][0]]))
    if(len(unmatched_tracks)>0):
        for trk_id in unmatched_tracks:
            tracks[trk_id].kalman_filter.predict()
            tracks[trk_id].update(tracks[trk_id].kalman_filter.x[0], tracks[trk_id].kalman_filter.x[1],0)
    
    tracks, detections = TrackMaintinance(tracks)
    for det in detections:
        print(det)
    rotate_img =ndimage.rotate(cfar_not_arr[t], 90)
    rotate_img = cfar_not_arr[t]
    ax.plot([det[0] for det in detections], [det[1] for det in detections], 'ro',label="Tracking")
    # display the image
    ax.set_title(f"Frame {t}")
    ax.imshow(rotate_img,label="Detections")
    ax.legend()
    
    plt.draw()
    plt.pause(0.01)
    ax.cla() # clear axis for the next frame
    print("\n")
    #time.sleep(1)