"""Detectors. Frame in, (x, y) of the ball or None out. Don't know about the filter at all."""

import cv2
import numpy as np


def _largest_blob_centroid(mask, min_area):
    """Centre of the biggest blob in the mask, or None."""
    
    contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #ret tuple
    #retr_ext ignores holes in blobs aka only gives external contour
    #Chainaproxsimp controls size of outline saving memory 
    refinedCon = [ c for c in contours if cv2.contourArea(c)>= min_area]
    if not refinedCon :
        return None
    biggest = max(refinedCon,key=cv2.contourArea)
    m = cv2.moments(biggest)
    if m["m00"] == 0: 
        return None #guard
    return (m["m10"]/m["m00"],m["m01"]/m["m00"])




class ColorDetector:
    def __init__(self, hsv_lower, hsv_upper, min_area):
        self.lower = np.array(hsv_lower)
        self.upper = np.array(hsv_upper)
        self.min_area = min_area
        self.kernel = np.ones((5, 5), np.uint8)
        self.last_mask = None   # kept for debugging

    def detect(self, frame):
        hsvFrame = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV) #convert to hue sat value

        mask = cv2.inRange(hsvFrame,self.lower,self.upper) # convert frame to a mask 


        mask =cv2.erode(mask,self.kernel) # erode removes noise 
        mask = cv2.dilate(mask,self.kernel) #dialte restores lost data from noise removal

        self.last_mask = mask #store mask 
        return _largest_blob_centroid(mask,self.min_area) #find centre


class BackgroundSubtractorDetector: 
    def __init__(self, min_area, history=500, var_threshold=16):
        self.subtractor = cv2.createBackgroundSubtractorMOG2(
            history=history, varThreshold=var_threshold, detectShadows=True)
        self.min_area = min_area
        self.kernel = np.ones((5, 5), np.uint8)
        self.last_mask = None

    def detect(self, frame):# locates moving pixels
        mask = self.subtractor.apply(frame) #returns a mask of moving, shadow and background pixels
        _ , mask = cv2.threshold(mask,200,255,cv2.THRESH_BINARY) #returns mask with removed shadows and threshhold used
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel) #removes noise
        self.last_mask=mask
        return _largest_blob_centroid(mask,self.min_area)