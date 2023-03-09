from collections import deque
import numpy as np
from .Utils import *
from .KF import KalmanFilter
from scipy.linalg import inv

from scipy.spatial import distance
from scipy.optimize import linear_sum_assignment
P_init = np.diag([10, 10])  # initial covariance matrix

Q = np.diag([15, 0])  # process noise covariance
#R = np.diag([1,1])  # measurement noise covariance
R = np.array([[  4.11455458 ,-16.52043271],
 [-16.52043271 , 66.33152914]])
dt = 50*1e-4
class Initiator:
    """Initiates new tracks."""

    def __init__(self,M,N ):
        self.tracks = []
        self.potential_targets = np.empty((0,2))
        self.M = M
        self.N = N
        self.id = 0
        self.treshold = 2

    def initiate(self, z):
        """Initiates a new track."""
        #self.tracks.append(Track(self.id+1, z[0], z[1], KalmanFilter(dt, z),P_init,Q,R))
        #print("meas",z.shape)
        #print("p",self.potential_targets.shape)
        self.potential_targets =np.append( self.potential_targets,z,axis=0)
        #print(self.potential_targets)
       
    def preliminary_track(self, z):
        """Initiates a new track."""
        if( len(self.tracks) == 0 ):return z
        
        if z is None:
            return None
        NIS_mat = np.empty((len(self.tracks),len(z)))
        
        for t,track in enumerate( self.tracks):
            H = np.array([[1, 0], [0, 1]])  # observation matrix

           
            
            S = H @ track.kalman_filter.P @ H.T + R  # innovation covariance
            save = track
            
            track.kalman_filter.predict()
            
            for m,mesurement in enumerate(z):
                
                
                NIS = (np.array([mesurement[1],mesurement[0]]) - H@track.kalman_filter.x).T @ inv(S) @ (np.array([mesurement[1],mesurement[0]]) - H@track.kalman_filter.x )
                
                if NIS <= self.treshold:
                    NIS_mat[t,m] = NIS
                    
                else:
                    NIS_mat[t,m] = 10000

        nis_c = NIS_mat.copy()
        unused_meas = z[np.all(NIS_mat ==10000, axis=0)]
        z = z[~np.all(NIS_mat ==10000, axis=0)]
        
        NIS_mat = NIS_mat[:,~np.all(NIS_mat==10000, axis=0)]
        row_ind, col_ind = linear_sum_assignment(NIS_mat)
        #print("##MTCHING#")
        for r,c in zip(row_ind, col_ind):
            # print(len(self.tracks))
            # print(nis_c)
            # print(NIS_mat)
            # [print(tr) for tr in self.tracks]
            # [print(det,f"Range:{det[1]*0.785277}, V:{(128-det[0])*-0.12755}") for det in z]
            # print("\n","##result##")
            # print(NIS_mat[r,c])
            # print("track:", self.tracks[r])
            # print(z[c],f"Range:{z[c][1]*0.785277}, V:{(128-z[c][0])*-0.12755},")
            self.tracks[r].update(z[c][1],z[c][0],1)
            #self.tracks[r].kalman_filter.update(np.array([z[c][1],z[c][0]]))
          #  print("matched after: ",self.tracks[r])
            #print("KF",self.tracks[r].kalman_filter)
         #   print("\n")
            
            
            
        for i in range(len(self.tracks)):
            if i not in row_ind:
                self.tracks[i].no_update()
                
        
        #print("unused",unused_meas.shape)  
        #input("Press Enter to continue...")
        #print("\n")
        return unused_meas
                    
                    

    def process_initiator(self,measurments,max_distance = 3):
        
        if( self.potential_targets.size == 0 ):return measurments
        if(measurments.size == 0):
            self.potential_targets = np.empty((0,2))
            return None
        #print("####INITIATOR####")
        euclidian_distance = distance.cdist(self.potential_targets, measurments, 'euclidean')
        euclidian_distance[euclidian_distance>max_distance] = 100000
        
        unused_meas = measurments[np.all(euclidian_distance==10000, axis=0)]
        euclidian_distance = euclidian_distance[:,~np.all(euclidian_distance ==10000, axis=0)]
        
        try:
       
            _, col_ind = linear_sum_assignment(euclidian_distance)
        except Exception as e:
            print("error",euclidian_distance)
            # print(self.potential_targets, measurments)
            print(e)
            
       
        for i in col_ind:
            
            kf =KalmanFilter(dt, np.array([measurments[i,1], measurments[i,0]]),P_init,Q,R)
            track = Track(self.id+1, measurments[i,1], measurments[i,0],kf ,track_age = 0,track_history=deque([1]),track_length=5)
            self.tracks.append(track)
         #   print("NEW: ",track)
            
            self.id += 1
        #input("Press Enter to continue...")
        #print("\n")
        self.potential_targets = np.empty((0,2))
       
        return unused_meas
    def trackMaintinance(self,max_age=10, N=3,M=5):
        print("####MAINTINANCE####")
        confirmed_tracks = []
        delete_tracks = []
        potential_tracks = []
        for track in self.tracks:
            
            if(sum(track.track_history) < N and len(track.track_history) >= M ):
                delete_tracks.append(track)
                print("deleted",track)
            elif sum(track.track_history)>=N :
                print("CONFIRMED",track)
                #track.track_history = deque([1,1,1,1,1])
                confirmed_tracks.append([track.kalman_filter.x[0],track.kalman_filter.x[1]])
                potential_tracks.append(track)
                #print([track.kalman_filter.x[0],track.kalman_filter.x[1]])
                #print(track.kalman_filter)
                #print("confirmed",track)
            else:
                potential_tracks.append(track)
                print("POTENTIAL",track)
        self.tracks = potential_tracks
        print("\n")
        return confirmed_tracks
    

    def main(self, measurments):
        """Initiates new tracks and manages track list."""
        #self.preliminary_track(measurments)
        print("measurments",measurments.shape)
        #print("\n", "##TRACKS##")
       # [print(tr) for tr in self.tracks]
        #print("\n")
        meas = self.preliminary_track(measurments)
        meas = self.process_initiator(meas)
        if(meas is not None):
            self.initiate(meas)
        

        return self.trackMaintinance()


        
           
           
            

       
