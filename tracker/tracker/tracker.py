"""Glues detector and filter together. All occlusion handling lives here."""

from dataclasses import dataclass
from enum import Enum, auto

import numpy as np


class TrackState(Enum):
    SEARCHING = auto()   # no track yet, waiting for a first detection
    TRACKING = auto()    # detection accepted this frame
    COASTING = auto()    # no usable detection, running on prediction alone
    LOST = auto()        # coasted too long, will reset to SEARCHING


@dataclass
class TrackResult:
    frame_idx: int
    detection: tuple | None        # raw detector output
    position: np.ndarray | None    # filtered estimate
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
        """Process one frame. force_miss simulates occlusion for demos/tests."""

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
        
