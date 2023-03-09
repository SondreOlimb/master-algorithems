import numpy as np
def TrackMaintinance(tracks, max_age=10, min_hits=3,del_tresh=0.3):
    detections = []
    delete_tracks = []
    for track in tracks:
        
        if(sum(tracks[track].track_history)/len(tracks[track].track_history) < del_tresh and len(tracks[track].track_history) > 3 or sum(tracks[track].track_history) == 0 ):
            delete_tracks.append(tracks[track].track_id)
        
        elif sum(tracks[track].track_history)>min_hits :
            detections.append([tracks[track].x,tracks[track].vx])
    for track in delete_tracks:
        del tracks[track]
    return tracks,detections    




class Track:
    """Class representing a track"""
    
    def __init__(self, track_id, x,  vx, kalman_filter,track_age, track_history,track_length=5 ):
        self.track_id = track_id
        self.track_age = track_age
        self.x = kalman_filter.x[0]
        self.vx = kalman_filter.x[1]
        self.kalman_filter = kalman_filter
        self.track_history = track_history
        self.track_length = track_length

        
        
   
    
    def update(self, x, vx,hist):
        """Updates the state of the track with a new observation."""
        self.kalman_filter.update(np.array([x,vx]))
        self.vx = self.kalman_filter.x[1]
        self.x = self.kalman_filter.x[0]
        self.track_age += 1
        self.track_history.append(hist)
        if(len(self.track_history)>self.track_length):
            print("##################################hei")
            self.track_history.popleft()
    def no_update(self):
        """Updates the state of the track with a new observation."""
        self.track_age += 1
        self.track_history.append(0)
        if(len(self.track_history)>self.track_length):
            self.track_history.popleft()

    def __str__(self):
        return f"ID:{self.track_id}, Range:{round(self.x*0.785277,2)}, V:{round((128-self.vx)*-0.12755,2)}, History:{self.track_history}"