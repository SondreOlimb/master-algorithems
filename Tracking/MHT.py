import numpy as np




class TOMHT:
    def __init__(self, dt, x_init, P_init, Q, R, lambda_c, threshold):
            self.dt = dt
            self.x = x_init
            self.P = P_init
            self.Q = Q
            self.R = R
            self.lambda_c = lambda_c
            self.threshold = threshold
            self.tracks = []
    
    