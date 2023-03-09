import numpy as np
from scipy.stats import multivariate_normal

# Define the state transition model
F = np.array([[1, 1],
              [0, 1]])

# Define the measurement model
H = np.array([[1, 0],
              [0, 1]])

# Define the covariance matrices
Q = np.diag([0.1, 0.01])
R = np.diag([0.1, 0.1])

# Define the clutter intensity
lambda_c = 1e-5

# Define the threshold for creating new tracks
threshold = 0.5

# Define a class for a track
class Track:
    def __init__(self, x, P):
        self.x = x
        self.P = P
        self.age = 1
        self.hits = 1
        self.score = 1
    
    def predict(self):
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
    
    def update(self, z):
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (z - H @ self.x)
        self.P = (np.eye(2) - K @ H) @ self.P
        self.hits += 1
    
    def get_prediction(self):
        return self.x, self.P
    
    def get_score(self, z):
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        x_pred = F @ self.x
        P_pred = F @ self.P @ F.T + Q
        innovation = z - H @ x_pred
        mahalanobis = innovation.T @ np.linalg.inv(S) @ innovation
        score = multivariate_normal.pdf(innovation, mean=None, cov=S)
        return score
    
    def get_state(self):
        return self.x

# Define a class for the tracker
class Tracker:
    def __init__(self):
        self.tracks = []
        self.track_num = 0
    
    def update(self, Z):
        N = Z.shape[1]
        for track in self.tracks:
            track.predict()
        scores = np.zeros((len(self.tracks), N))
        for i, track in enumerate(self.tracks):
            for j in range(N):
                scores[i, j] = track.get_score(Z[:, j])
        for j in range(N):
            best_score = -np.inf
            best_track = None
            for track in self.tracks:
                score = track.get_score(Z[:, j])
                if score > best_score:
                    best_score = score
                    best_track = track
            if best_score > threshold:
                best_track.update(Z[:, j])
            else:
                new_track = Track(Z[:, j], Q)
                self.tracks.append(new_track)
                self.track_num += 1
        self.tracks = [track for track in self.tracks if track.hits > 1]
        for track in self.tracks:
            track.age += 1
            track.score = np.sum(scores[i, :])
    
    def get_tracks(self):
        tracks = []
        for track in self.tracks:
            tracks.append(track.get_state())
        return tracks