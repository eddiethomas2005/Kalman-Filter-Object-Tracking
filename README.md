# Ball tracker (OpenCV + Kalman filter from scratch)

Tracks a single ball in a video. OpenCV finds the ball in each frame, and a Kalman filter I
wrote in plain NumPy smooths the detections and keeps predicting where the ball is when the
detector loses it.

I didn't use `cv2.KalmanFilter`, the point was to actually understand predict/update and how
the filter copes with missing measurements.

## How it works

Every frame:

1. **Detect.** Find the ball either by colour (HSV threshold) or by background subtraction
   (MOG2). Gives the centre `(x, y)` or `None`.
2. **Predict.** Move the estimate forward with a constant velocity model. Uncertainty grows.
3. **Gate.** If there's a detection, check its Mahalanobis distance from the prediction.
   Anything past the 99% chi^2 cutoff (9.21) is treated as a false detection.
4. **Update or coast.** Good detection → correct the prediction, uncertainty shrinks. No
   detection → skip the update and carry on with the prediction ("coasting").
5. **Give up.** After 30 frames of coasting the track is LOST and it goes back to searching.

The state is `[px, py, vx, vy]` even though the detector only gives position. The filter
works out velocity from how the position changes, which is what lets it keep going through
an occlusion.

```
SEARCHING ──first detection──► TRACKING ◄──good detection──┐
    ▲                             │                        │
    │                     no / bad detection               │
    │                             ▼                        │
    └──── > 30 frames ──── LOST ◄─ COASTING ───────────────┘
```

In the output video: red dot = raw detection, green circle + ellipse = filter estimate with
its 2 sigma uncertainty, black line = trail, and the current state in the top left. The
ellipse visibly grows while it's coasting.

## Setup

Python 3.10+ (uses `X | None` type hints).

```bash
cd tracker/tracker
python -m venv .venv
.venv\Scripts\activate          # mac/linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Running

Run from inside `tracker/tracker/`, the imports assume that's the working directory.
The videos aren't in the repo (`*.mp4` and `data/` are gitignored), so put your own clip in
`data/`.

```bash
# colour detector (default)
python main.py data/ball_long.mp4

# background subtraction, works for any colour ball as long as the camera doesn't move
python main.py data/ball_long_orange.mp4 --detector bgsub

# blank out detections for frames 60-90 to show it coasting
python main.py data/ball_long.mp4 --fake-occlusion 60 90

# webcam
python main.py 0
```

| Flag | What it does |
|---|---|
| `source` | video path, or `0` for webcam |
| `--detector` | `color` (default) or `bgsub` |
| `--out` | output video (default `output/tracked.mp4`) |
| `--fake-occlusion START END` | drops detections for that frame range |
| `--no-display` | don't open a window, just write the outputs |

Press `q` to stop early. Every run writes:

- the annotated video to `--out`
- `output/log.csv`: frame, raw detection, filter estimate and state for every frame
- `output/trajectory.png`: raw detections vs the filtered path, coasting frames in orange

The HSV range in `config.py` is for a green ball. For a different colour either change
`HSV_LOWER` / `HSV_UPPER` or use `--detector bgsub`.

## Tests

The filter is tested on a fake noisy straight line, so no video needed:

```bash
pytest tests/
```

They check that:
- the filtered path is closer to the truth than the raw measurements
- after 15 dropped frames it's still within 15 px of the truth
- uncertainty grows while coasting
- re-initialising after a lost track resets the covariance

## Layout

```
tracker/
├── config.py          all the tunable numbers
├── detector.py        ColorDetector, BackgroundSubtractorDetector
├── kalman.py          KalmanFilter (NumPy only)
├── tracker.py         Tracker + state machine, handles occlusion
├── visualize.py       drawing on frames + trajectory plot
├── main.py            CLI / main loop, writes video, csv and plot
├── tests/
│   └── test_kalman.py
├── NOTES.md           my dev notes
└── requirements.txt
```

`kalman.py` doesn't know about video and `detector.py` doesn't know about the filter.
`tracker.py` is the only place they meet, which is why the filter can be tested on its own.

## Tuning

All in `config.py`:

- `MEAS_VAR` (R): higher = trust the detector less, smoother but more lag
- `PROCESS_VAR` (Q): higher = trust the motion model less, follows detections more closely
- `MAX_COAST_FRAMES`: how long to predict through an occlusion before giving up
- `GATE_THRESHOLD`: how far a detection can be from the prediction and still count
- `TRAIL_LENGTH`: how many past positions to draw

## Ideas for later

- constant acceleration model so bounces don't throw it off as much
- multiple balls with Hungarian matching on the same gating distance
- swap in a particle filter behind the same predict/update interface
