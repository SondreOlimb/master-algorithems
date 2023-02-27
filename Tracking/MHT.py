import numpy as np
from itertools import product

class Track:
    """Class representing a track"""
    
    def __init__(self, track_id, x, y, vx, vy):
        self.track_id = track_id
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        
    def predict(self, dt):
        """Predicts the state of the track after time 'dt' using a constant velocity model."""
        self.x += self.vx * dt
        self.y += self.vy * dt
    
    def update(self, x, y):
        """Updates the state of the track with a new observation."""
        self.vx = x - self.x
        self.vy = y - self.y
        self.x = x
        self.y = y

class Hypothesis:
    """Class representing a hypothesis"""
    
    def __init__(self, tracks, observation_indices, probability):
        self.tracks = tracks
        self.observation_indices = observation_indices
        self.probability = probability
        
    def copy(self):
        return Hypothesis([t for t in self.tracks], [i for i in self.observation_indices], self.probability)
    
    def update_probability(self, likelihood):
        self.probability *= likelihood

def distance(track, observation):
    """Computes the distance between a track and an observation."""
    return np.sqrt((track.x - observation[0])**2 + (track.y - observation[1])**2)

def mht(observations, max_distance, dt, p_detection, p_false_alarm):
    """Performs multiple-hypothesis tracking between measurements and tracks.
    
    Parameters:
        observations (list): A list of observations in the form of (x, y) tuples.
        max_distance (float): The maximum distance between a measurement and a track for them to be associated.
        dt (float): The time interval between observations.
        p_detection (float): The probability of detecting an object in the scene.
        p_false_alarm (float): The probability of a false alarm per unit area.
    
    Returns:
        tracks (list): A list of tracks in the form of Track objects.
    """
    next_track_id = 0
    hypotheses = [Hypothesis([], [], 1)]
    tracks = []
    
    for observation_index, observation in enumerate(observations):
        new_hypotheses = []
        
        # Generate new hypotheses by adding each observation to each track
        for hypothesis in hypotheses:
            # Create a new track for the observation
            new_track = Track(next_track_id, observation[0], observation[1], 0, 0)
            next_track_id += 1
            
            # Predict the existing tracks and add them to the new hypothesis
            for track in hypothesis.tracks:
                track_copy = Track(track.track_id, track.x, track.y, track.vx, track.vy)
                track_copy.predict(dt)
                new_hypothesis = hypothesis.copy()
                new_hypothesis.tracks.append(track_copy)
                new_hypothesis.observation_indices.append(None)
                new_hypotheses.append(new_hypothesis)
            
            # Add the new track to the new hypothesis
            new_hypothesis = hypothesis.copy()
            new_hypothesis.tracks.append(new_track)
            new_hypothesis.observation_indices.append(observation_index)
            new_hypotheses.append(new_hypothesis)
        
        # Generate new hypotheses by not associating the observation with any existing track
        for hypothesis in hypotheses:
            new_hypothesis = hypothesis.copy()
            new_hypothesis.tracks.append