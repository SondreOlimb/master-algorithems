import numpy as np
from scipy.linalg import inv


class KalmanFilter(object):
    def __init__(self, dt, x_init, P_init, Q, R):
        self.dt = dt  # time step
        self.x = x_init  # initial state (position, velocity)
        self.P = P_init  # initial covariance matrix
        self.Q = Q  # process noise covariance
        self.R = R  # measurement noise covariance
        

    def predict(self, u=None):
        
        F = np.array([[1, -self.dt], [0, 1]])  # state transition matrix
        if u is not None:
            B = np.array([[(self.dt**2)/2], [self.dt]])  # control input matrix
            self.x = F @ self.x + B @ u
                      
        else:
            
            self.x = F @ self.x
            
        self.P = F @ self.P @ F.T + self.Q

    def update(self, z):
        H = np.array([[1, 0], [0, 1]])  # observation matrix
       
        y = z - H @ self.x  # innovation
        
        self.S = H @ self.P @ H.T + self.R  # innovation covariance
        
        K = self.P @ H.T @ inv(self.S)  # Kalman gain
        
        
        self.x = self.x + K @ y  # updated state estimate
        
        self.P = (np.eye(2) - K @ H) @ self.P  # updated covariance matrix
    def __str__(self):
        return f"Range:{self.x[0]*0.785277}, V:{(128-self.x[1])*-0.12755}"
    
    
