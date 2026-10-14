# Single-object tracker: code structure

A starting skeleton for the OpenCV detector + from-scratch NumPy Kalman filter project.
Stub files sit next to this document in `/mnt/project-files/tracker/`; every function has
its signature and docstring, with `TODO` where you write the real code.

## Folder layout

```
tracker/
├── README.md              # write last: approach, math, results GIF
├── requirements.txt       # numpy, opencv-python, matplotlib
├── config.py              # every tunable number in one place
├── detector.py            # frame -> (x, y) or None
├── kalman.py              # KalmanFilter class, pure NumPy, no OpenCV
├── tracker.py             # glues detector + filter, owns occlusion logic
├── visualize.py           # drawing on frames + post-run plots
├── main.py                # CLI entry point: open video, run loop, save output
├── tests/
│   └── test_kalman.py     # filter tested on synthetic data, no video needed
└── data/                  # your test videos (gitignored if large)
```

The key design rule: **`kalman.py` knows nothing about video, and `detector.py` knows
nothing about Kalman filters.** `tracker.py` is the only place they meet. That keeps the
filter testable on fake data and makes the README easy to explain.

## Module by module

### `kalman.py`: the core (build and test this first)

Constant-velocity model. State is `x = [px, py, vx, vy]ᵀ`, measurement is `z = [px, py]ᵀ`.

| Matrix | Shape | Meaning |
|---|---|---|
| `F` | 4×4 | state transition: `px += vx·dt`, `py += vy·dt` |
| `H` | 2×4 | measurement picks out position: `[[1,0,0,0],[0,1,0,0]]` |
| `Q` | 4×4 | process noise: how much the motion model can be wrong (unmodeled acceleration) |
| `R` | 2×2 | measurement noise: how jittery the detector is, in pixels² |
| `P` | 4×4 | state covariance: the filter's current uncertainty |

```python
class KalmanFilter:
    def __init__(self, dt, process_var, meas_var, init_pos_var, init_vel_var): ...
    def initialize(self, z): ...          # set state from first detection, velocity = 0
    def predict(self) -> np.ndarray: ...   # x = F x ;  P = F P Fᵀ + Q
    def update(self, z) -> np.ndarray: ... # K = P Hᵀ (H P Hᵀ + R)⁻¹ ; x += K(z − Hx) ; P = (I − KH) P
    def innovation_distance(self, z) -> float: ...  # Mahalanobis distance, used for gating
    @property
    def position(self): ...
```

Build `Q` with the standard discrete white-noise-acceleration form (in the stub). Tuning
intuition for the README: large `R` means the filter trusts the detector less and smooths more;
large `Q` means it trusts the motion model less and follows detections more closely.

### `detector.py`: frame to measurement

```python
class ColorDetector:
    def __init__(self, hsv_lower, hsv_upper, min_area): ...
    def detect(self, frame) -> tuple[float, float] | None: ...

class BackgroundSubtractorDetector:
    def __init__(self, min_area, history=500, var_threshold=16): ...
    def detect(self, frame) -> tuple[float, float] | None: ...
```

Both share one interface: return the centroid of the largest blob, or `None`. Pipeline for
colour: BGR to HSV, `cv2.inRange`, erode/dilate to remove speckle, `findContours`, keep the
largest contour above `min_area`, centroid from `cv2.moments`. Start with `ColorDetector` and a
brightly coloured ball; it's the easiest to debug. A small `tune_hsv.py` with trackbars is a
worthwhile 20-minute detour if thresholds are fiddly.

### `tracker.py`: where occlusion is handled

```python
class TrackState(Enum): SEARCHING, TRACKING, COASTING, LOST

class Tracker:
    def __init__(self, detector, kf, max_coast_frames, gate_threshold): ...
    def step(self, frame) -> TrackResult: ...
```

`TrackResult` carries the raw detection (or `None`), the filtered position, the covariance,
and the state, so `visualize.py` can draw everything without re-computing.

### `visualize.py`

- `draw_frame(frame, result, history)`: raw detection as a red dot, filtered estimate as a
  green circle, a 2-sigma uncertainty ellipse from `P[:2,:2]`, trail of past positions, state
  label (TRACKING / COASTING) in the corner. The ellipse visibly growing during occlusion is
  the single best visual for the demo.
- `plot_trajectories(log)`: matplotlib plot of raw vs. filtered path after the run.
- Optional: side-by-side output video of raw detection vs. filtered tracking.

### `main.py`

```
python main.py data/ball.mp4 --detector color --out output/ball_tracked.mp4
```

Opens the video, builds detector + filter + tracker from `config.py`, loops over frames,
calls `tracker.step`, draws, writes the output video, saves a CSV log and the trajectory plot.
`cv2.VideoCapture(0)` instead of a file path is all live webcam needs later.

## The main loop and missed detections

This is the logic in `Tracker.step`, and the actual point of the project:

```
z = detector.detect(frame)

if state == SEARCHING:
    if z is not None: kf.initialize(z); state = TRACKING
    return

kf.predict()                          # ALWAYS predict, every frame

if z is not None and kf.innovation_distance(z) < gate_threshold:
    kf.update(z)                      # good detection: correct the prediction
    missed = 0
    state = TRACKING
else:
    missed += 1                       # no detection (occluded) or an outlier: skip update
    state = COASTING                  # the filter runs on prediction alone, P grows
    if missed > max_coast_frames:
        state = LOST -> back to SEARCHING
```

Three ideas worth explaining in the README:

1. **Predict always, update only when you have a measurement.** Skipping `update` is the
   whole occlusion handler. The state keeps moving at its last estimated velocity and `P`
   grows each frame, so the filter "knows" it is less sure.
2. **Gating.** The Mahalanobis distance `yᵀ S⁻¹ y` (with innovation `y = z − Hx` and
   `S = H P Hᵀ + R`) rejects detections that are implausibly far from the prediction, e.g. a
   red shirt that appears while the red ball is hidden. A chi-squared threshold of ~9.21
   (99%, 2 DOF) is a sensible start. Because `S` grows during coasting, the gate widens
   automatically and the object can be reacquired after occlusion.
3. **Give up eventually.** After `max_coast_frames` without a match, the prediction is
   unreliable; go back to searching and re-initialise on the next detection.

## Suggested build order

1. `kalman.py` + `tests/test_kalman.py`: generate a straight-line trajectory with Gaussian
   noise, check the filter output is smoother than the input, and that it keeps going sensibly
   if you drop measurements 30–40.
2. `detector.py` on a recorded video, drawing only raw detections.
3. `tracker.py` without gating, then add gating.
4. Record a test video where the ball passes behind something. Optionally add a
   `--fake-occlusion START END` flag to blank detections for a frame range, so you can demo
   occlusion reproducibly.
5. `visualize.py`, output video, plot, README.

## Stretch goals (only after the above works)

Multi-object (Hungarian assignment on the same gating distance), a particle filter swap-in
behind the same `predict`/`update` interface, constant-acceleration model, live webcam.
