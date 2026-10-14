"""All tunable numbers in one place."""


# Kalman filter
DT = 1.0                 # time step in frames (use 1/fps for seconds)
PROCESS_VAR = 1.0        # Q scale: unmodeled acceleration, pixels/frame^2
MEAS_VAR = 25.0          # R: detector jitter, pixels^2 (std ~5 px)
INIT_POS_VAR = 25.0
INIT_VEL_VAR = 1000.0    # we know nothing about velocity at first

# Tracker
MAX_COAST_FRAMES = 30    # frames to predict through before declaring the track lost
GATE_THRESHOLD = 9.21    # chi-squared, 2 DOF, 99%

# Colour detector (HSV). Example values for a bright green ball; tune for yours.
HSV_LOWER = (35, 80, 80)
HSV_UPPER = (85, 255, 255)
MIN_AREA = 100           # ignore blobs smaller than this, pixels




#visulalize

TRAIL_LENGTH = 40