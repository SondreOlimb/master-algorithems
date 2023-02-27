import numpy as np
def NearestNeighbor(observations, tracks, max_distance):
    """Performs nearest-neighbor data association between measurements and tracks.
    
    Parameters:
        observations (list): A list of observations in the form of (x, y) tuples.
        tracks (list): A list of tracks in the form of dictionaries, where each dictionary contains the following keys:
            - 'id': A unique identifier for the track.
            - 'x': The current x-coordinate of the track.
            - 'y': The current y-coordinate of the track.
        max_distance (float): The maximum distance between a measurement and a track for them to be associated.
    
    Returns:
        matches (list): A list of tuples in the form of (track_id, observation_index), where 'track_id' is the identifier
            of the associated track and 'observation_index' is the index of the associated observation in the 'observations' list.
        unmatched_tracks (list): A list of identifiers for tracks that did not have any associated observations.
        unmatched_observations (list): A list of indices for observations that did not have any associated tracks.
    """
    matches = []
    unmatched_tracks = []
    unmatched_observations = list(range(len(observations)))
    
    for track in tracks:
        
        min_distance = float('inf')
        min_index = -1
        
        for i, observation in enumerate(observations):
            min_index = -1
            
            distance = np.sqrt((tracks[track].x - observation[1])**2 + (tracks[track].vx - observation[0])**2)
            
            if distance < max_distance :
                min_distance = distance
                min_index = i
                
        
        if min_index >= 0:
            
            try:
                matches.append((tracks[track].track_id, min_index))
                unmatched_observations.remove(min_index)
            except:
                pass
            
            min_index = -1
        else:
            unmatched_tracks.append(tracks[track].track_id)
            min_index = -1
    
    return matches, unmatched_tracks, unmatched_observations