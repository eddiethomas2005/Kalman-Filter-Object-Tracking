"""Every number I might want to tweak, so I'm not hunting through files."""


# Kalman filter
DT = 1.0                 # working in frames, 1/fps if I want real units
PROCESS_VAR = 10        # Q scale, bumped up from 1.0
MEAS_VAR = 25.0          # R, detector jitter in px^2 (~5 px std)
INIT_POS_VAR = 25.0
INIT_VEL_VAR = 1000.0    # huge on purpose, no clue about velocity on frame 1

# Tracker
MAX_COAST_FRAMES = 30    # ~1s at 30fps then give up
GATE_THRESHOLD = 9.21    # chi^2, 2 dof, 99%

# HSV range for the green ball. orange video doesn't work with these, use bgsub for it
HSV_LOWER = (35, 80, 80)
HSV_UPPER = (85, 255, 255)
MIN_AREA = 100           # px, anything smaller is noise




#visulalize

TRAIL_LENGTH = 40