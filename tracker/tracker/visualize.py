"""Drawing on frames during the run, and plots after it."""

import config
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tracker import TrackState
import os





STATE_COLOURS ={ TrackState.TRACKING: (0,0,0),
                TrackState.COASTING: (0,0,0),
                 TrackState.SEARCHING: (0,0,0),
                  TrackState.LOST: (0,0,255) }

STATE_LABELS = {TrackState.TRACKING: "Tracking",
                TrackState.COASTING: "Coasting",
                TrackState.SEARCHING: "Searching",
                TrackState.LOST: "Lost"}




def draw_uncertainty_ellipse(frame, center, cov, n_sigma=2.0,thickness=2, colour=(0, 255, 0)): #an elipse is drawn from the degrees of freedom all possible places ball can go
    """Ellipse from the 2x2 position covariance."""
    vals, vecs = np.linalg.eigh(cov) #gets eigen vectors and eigenvalues of covarience matrix
    axes = n_sigma * np.sqrt(vals) # converts variance into standard deviation this will be half length of eclipse
    angle = np.degrees(np.arctan2(vecs[1,0],vecs[0,0])) #finds angle of first vector as matrix is symetric then other angle will be perpendicular
    axes = tuple([int(round(a)) for a in axes ]) #formatting
    center =tuple([int(round(a)) for a in center ])
    cv2.ellipse(frame, center, axes, angle, 0, 360, colour, thickness,lineType=cv2.LINE_AA) #draws eclipse on frame

    
def intTuple (floatArray):
    return (int(floatArray[0]),int(floatArray[1]))

def draw_frame(frame, result, history):
    """Red dot = raw detection, green circle + ellipse = filtered, trail, state label."""
    state = result.state
    detection = result.detection 
    
    filt_det = result.position
    
    cov = result.position_cov
    trails = history[-config.TRAIL_LENGTH:][::-1]


    if detection is not None:
        det_cen = intTuple(detection)
        cv2.circle(frame, det_cen, 4, (0,0,255), -1,lineType=cv2.LINE_AA)

    if filt_det is not None:
            filt_cen = intTuple(filt_det)
            cv2.circle(frame, filt_cen, 8, (0,255,0), 2,lineType=cv2.LINE_AA)
            draw_uncertainty_ellipse(frame,filt_cen,cov,)
    if len(trails)>1:
        for i in range(len(trails)-1) :
            t1 = trails[i].position
            t2=trails[i+1].position
            if t1 is None or t2 is None:
                break
            cv2.line(frame, intTuple(t1),intTuple(t2), (0,0,0),2,lineType=cv2.LINE_AA)

    cv2.putText(frame, STATE_LABELS[state], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75,STATE_COLOURS[state], thickness=2,lineType = cv2.LINE_AA)
    return frame


def plot_trajectories(results, out_path):
    """Matplotlib plot of raw detections vs. filtered path; shade coasting frames."""
    x_raw=[]
    y_raw=[]
    x_filt=[]
    y_filt=[]
    x_coast=[]
    y_coast=[]
    for r in results: #filter and catergorise ball history
        raw =r.detection
        if raw is not None:
            x_raw.append(raw[0])
            y_raw.append(raw[1])
        state =r.state
        filtered=r.position
        if filtered is not None:
            x_filt.append(filtered[0])
            y_filt.append(filtered[1])
            if state == TrackState.COASTING:
                x_coast.append(x_filt[-1])
                y_coast.append(y_filt[-1])
        else:
            x_filt.append(np.nan)
            y_filt.append(np.nan)
    fig , ax =plt.subplots(figsize=(8,6))
    ax.scatter(x_raw,y_raw,s=8, alpha=0.5,color='r',label='Raw Detection')
    ax.plot(x_filt,y_filt,color='g',linestyle='-',label='Kalman Filter')
    ax.scatter(x_coast,y_coast,color='orange',label='kalman prediction')
    ax.set_aspect('equal')
    ax.set_xlabel("x pixels")
    ax.set_ylabel("y pixels")
    ax.set_title("Trajectories plot")
    ax.legend()
    ax.invert_yaxis()
    
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

        
         
