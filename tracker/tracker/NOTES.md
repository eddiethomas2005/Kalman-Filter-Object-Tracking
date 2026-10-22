# notes

Stuff I want to remember about this. Not proper docs.

## why it's split up like this

- kalman.py has no OpenCV in it. means I can test the filter on fake data, and if the tracker
  does something weird I know it's the glue and not the maths
- detectors only return (x, y) or None, the tracker doesn't care which one it's using
- tracker.py is the only file that touches both
- detector + filter get passed into Tracker rather than made inside it, so a test could give it
  a fake detector

## decisions

- **constant velocity.** fine for a ball rolling/flying in a straight-ish line. can't predict
  bounces though (see below)
- **4D state from 2D measurements.** velocity never gets measured, the filter infers it. that's
  the whole reason it can keep going when the ball disappears
- **INIT_VEL_VAR = 1000.** no idea about velocity on the first frame so P starts huge there and
  drops after a few updates
- **Mahalanobis gate at 9.21** (chi^2, 2 dof, 99%). S grows while coasting so the gate widens by
  itself and it can grab the ball again afterwards
- **give up after 30 frames**, about a second
- **PROCESS_VAR is 10 not 1.** tests still use 0.01 because the fake data is a perfect straight
  line, real footage needs way more
- **K via np.linalg.solve.** `solve(S, H @ P).T` gives `P H^T S^-1` without inverting S, works
  because P and S are symmetric
- **LOST lasts one frame**, then step() flips it back to SEARCHING at the start of the next frame
  so it reinitialises on the next detection

## gotchas I hit / need to remember

- MOG2 marks shadows as 127, threshold at 200 or the shadow counts as the ball
- image y goes down, need invert_yaxis() on the plot
- frame_idx: every path through step() has to increment it or --fake-occlusion breaks.
  main.py reads it *before* step so the result uses the pre-increment value
- reinitialising after LOST has to reset P too, otherwise the new track starts with the huge
  coasted uncertainty (that's what test_reinitialize_resets_uncertainty checks)
- the HSV range only works for the green ball. orange video gets 0 detections with colour, use
  bgsub for it
- **bounces**: when the ball bounces or turns sharply the detection jumps outside the gate, it
  coasts off in a straight line and you get those long orange lines in trajectory.png. the gate
  is doing its job, it's the motion model that's too simple. constant acceleration or a wider
  gate would help
- cv2.imshow/destroyAllWindows need the normal opencv-python, the headless build crashes on them
- the csv + plot always go to output/, only the video follows --out
- STATE_COLOURS are BGR, so TRACKING (255,0,0) is blue not red

## todo

- [ ] small tune_hsv.py with trackbars to pick colour ranges
- [ ] gif of the ellipse growing during occlusion for the readme
- [ ] try constant acceleration (6D state) and compare on the bouncy bits
- [ ] maybe: multiple objects, particle filter
