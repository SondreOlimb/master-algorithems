def TrackMaintinance(tracks):
    detections = []
    delete_tracks = []
    for track in tracks:
        
        if(sum(tracks[track].track_history)/len(tracks[track].track_history) < 0.5 and len(tracks[track].track_history) > 3 ):
            delete_tracks.append(tracks[track].track_id)
        
        elif sum(tracks[track].track_history)>3 :
            detections.append([tracks[track].x,tracks[track].vx])
    for track in delete_tracks:
        del tracks[track]
    return tracks,detections    




class Track:
    """Class representing a track"""
    
    def __init__(self, track_id, x,  vx, kalman_filter,track_age, track_history ):
        self.track_id = track_id
        self.track_age = track_age
        self.x = x
        self.vx = vx
        self.kalman_filter = kalman_filter
        self.track_history = track_history

        
        
   
    
    def update(self, x, vx,hist):
        """Updates the state of the track with a new observation."""
        self.vx = vx
        self.x = x
        self.track_age += 1
        self.track_history.append(hist)
        if(len(self.track_history)>5):
            self.track_history.popleft()

    def __str__(self):
        return f"ID:{self.track_id}, Range:{self.x*0.785277}, V:{(128-self.vx)*-0.12755}"