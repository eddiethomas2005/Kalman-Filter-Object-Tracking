"""Where the detector and filter meet. All the occlusion handling is here."""

from dataclasses import dataclass
from enum import Enum, auto

import numpy as np


class TrackState(Enum):
    SEARCHING = auto()   # no track, waiting for a detection
    TRACKING = auto()    # used a detection this frame
    COASTING = auto()    # no (good) detection, prediction only
    LOST = auto()        # coasted too long, goes back to SEARCHING next frame


@dataclass
class TrackResult:
    frame_idx: int
    detection: tuple | None        # straight from the detector
    position: np.ndarray | None    # filter's estimate
    position_cov: np.ndarray | None
    state: TrackState




class Tracker:
    def __init__(self, detector, kf, max_coast_frames, gate_threshold):
        self.detector = detector
        self.kf = kf
        self.max_coast_frames = max_coast_frames
        self.gate_threshold = gate_threshold
        self.state = TrackState.SEARCHING
        self.missed = 0
        self.frame_idx = 0 #frame index

    def inovCheck(self ,z): #gate checking how likely the ball would be in 99% of time
        if z is None: return False
        in_dist = self.kf.innovation_distance(z)
        if in_dist>self.gate_threshold: return False
        return True
        

    def step(self, frame, force_miss=False):
        """One frame. force_miss fakes an occlusion (--fake-occlusion)."""

        if self.state == TrackState.LOST: 
            self.state=TrackState.SEARCHING
            
        z = None if force_miss else self.detector.detect(frame)


        if self.state == TrackState.SEARCHING:
            if z is None:
                self.frame_idx += 1
                return TrackResult(self.frame_idx-1,z,None, None,self.state)
            else:
                self.kf.initialize(z)
                self.state= TrackState.TRACKING
                self.missed=0
        elif self.state == TrackState.TRACKING  :
            self.kf.predict()
            if self.inovCheck(z):
                self.kf.update(z)
                self.missed=0
            else :
                self.missed+=1
                self.state = TrackState.COASTING
        elif self.state == TrackState.COASTING :
            self.kf.predict()
            if z is None or not self.inovCheck(z):
                self.missed+=1
                if self.missed> self.max_coast_frames: self.state=TrackState.LOST
            else :
                self.kf.update(z)
                self.missed=0
                self.state=TrackState.TRACKING
        


        self.frame_idx += 1
        return TrackResult(self.frame_idx-1,z,self.kf.position, self.kf.position_cov,self.state)
        
