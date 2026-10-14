 # How the Kalman filter works (for this tracker)

You already have the core idea right: every frame you have two guesses about where the ball is, a **prediction** from physics and a **measurement** from OpenCV. Both are uncertain, both are modelled as Gaussians, and the filter blends them into one better guess. This document fills in the two gaps:

1. What "update" actually does.
2. Why the state is 4D `[px, py, vx, vy]` when OpenCV only gives you 2D `(px, py)`.

It starts with a 1D example using plain numbers, then shows the matrix version that `kalman.py` uses. It does not write your TODOs for you, but by the end you should be able to.

---

## 1. The loop: predict, then update

Every frame the filter does two steps:

| Step | Question it answers | What happens to uncertainty |
|---|---|---|
| **Predict** | "Given where I thought the ball was and how it was moving, where should it be now?" | **Grows.** Time passing makes you less sure. |
| **Update** (also called *correct*) | "OpenCV says the ball is here. How should I adjust my prediction?" | **Shrinks.** New evidence makes you more sure. |

So "update" just means **correcting the prediction using the measurement**. The output of the update is the filter's best estimate for this frame, and it becomes the starting point for next frame's prediction. That's the "repeated" part.

The state is always stored as two things:

- `x`: the mean (best guess).
- `P`: the covariance (how unsure we are, and how the uncertainties are linked; more on that later).

A Gaussian is fully described by a mean and a variance, which is why these two are all the filter needs to remember.

---

## 2. The update in 1D, with numbers

Forget velocity for a moment. Say we only track the ball's x position.

- Prediction: x = **100 px**, variance P = **4** (std 2 px). We're fairly confident.
- Measurement: z = **110 px**, variance R = **16** (std 4 px). The detector is noisier.

### Step 1: the innovation

How far off was the prediction?

```
y = z - x = 110 - 100 = 10
```

This is the **innovation** (or residual). It's the "surprise" in the measurement.

### Step 2: the Kalman gain

How much should we trust that surprise?

```
K = P / (P + R) = 4 / (4 + 16) = 0.2
```

K is always between 0 and 1. It's the fraction of the way we move from the prediction toward the measurement.

- If the prediction is very uncertain (P large), K → 1 and we basically take the measurement.
- If the detector is very noisy (R large), K → 0 and we basically ignore it.

### Step 3: the new estimate

```
x_new = x + K * y = 100 + 0.2 * 10 = 102
```

We moved 20% of the way from 100 toward 110, because the prediction was 4 times more trustworthy than the measurement.

### Step 4: the new uncertainty

```
P_new = (1 - K) * P = 0.8 * 4 = 3.2
```

Notice **3.2 is smaller than both 4 and 16**. Combining two independent noisy estimates always gives you something more certain than either one alone. This is the real payoff of the filter, and it's exactly what you'd get by multiplying the two Gaussian curves together: the product is another Gaussian, narrower than both, centred at the weighted average.

```
       prediction (100, σ=2)
            ▁▃█▃▁
                      measurement (110, σ=4)
                         ▁▂▃▅▃▂▁
              ▁▄█▄▁   ← combined (102, σ≈1.8)
  ──────────┼─────────┼──────────
           100       110
```

That's the entire update step. Everything in the matrix version is this same idea with vectors.

### And the predict step in 1D?

If the ball were stationary, predict would be:

```
x_pred = x            (best guess doesn't move)
P_pred = P + Q        (uncertainty grows by Q, the "process noise")
```

Q represents everything the motion model doesn't capture: the ball getting nudged, bouncing, slowing down. Without Q, P would keep shrinking forever and the filter would eventually ignore all measurements.

---

## 3. Why 4D? Because velocity is hidden but essential

### The problem with tracking position alone

Suppose the ball is rolling right at 5 px/frame. If your state is only position, your prediction for next frame is "same place as last frame". That's wrong by 5 px every single frame. The filter would lag behind the ball, and when the ball goes behind something (occlusion) the prediction would just sit still where it disappeared.

To predict properly you need to know **where the ball is AND how fast it's moving**. Hence, per axis:

```
state = [position, velocity]
```

Two axes (x and y) × two quantities (position, velocity) = **4 numbers**:

```
x = [px, py, vx, vy]
```

That's all the "4 dimensional vector" is. It's not a 4D space in any geometric sense, it's just a list of the four numbers you need to describe a ball moving in 2D.

### But OpenCV never measures velocity

Right. Velocity is a **hidden state**: you never observe it directly. The filter infers it by watching how position changes over time. The two matrices that make this work are F and H.

### F: the motion model (used in predict)

F says how the state at one frame becomes the state at the next. For a single axis with constant velocity:

```
position_next = position + velocity * dt
velocity_next = velocity
```

Written as a matrix acting on `[p, v]`:

```
F_1D = [[1, dt],
        [0,  1]]
```

Check it: `F_1D @ [p, v] = [p + v*dt, v]`. ✓

Your 4×4 `F` in `kalman.py` is the same thing done for both axes at once. **Exercise for your TODO:** with the state ordered `[px, py, vx, vy]`, figure out which entries of `np.eye(4)` need a `dt` so that `px` picks up `vx*dt` and `py` picks up `vy*dt`. (Look at how `Q` is built just below it in `kalman.py`; it uses the same index pairing: `[0, 2]` for x, `[1, 3]` for y.)

### H: the measurement model (used in update)

H says "given a state, what would the detector see?" The detector sees position only, so H just picks out px and py and throws away velocity:

```
H = [[1, 0, 0, 0],
     [0, 1, 0, 0]]

H @ [px, py, vx, vy] = [px, py]
```

This is already written for you in `kalman.py`. H is how the filter compares a 4D state with a 2D measurement: the innovation is `y = z - H x`, a 2D vector.

### So how does a position measurement ever change velocity?

This is the clever bit, and it's all in **P**, the covariance matrix.

P isn't just "how unsure am I about each number". Its off-diagonal entries say **how the errors are linked**. After a predict step, F mixes velocity into position (`p += v*dt`), so if your velocity guess is too low, your position guess will also be too low. P records that link as a positive covariance between position and velocity.

Then in update, the Kalman gain is

```
K = P Hᵀ S⁻¹       where S = H P Hᵀ + R
```

`P Hᵀ` grabs the column of P that relates **every** state variable to position. So K has a row for velocity too, and that row is nonzero whenever position and velocity are correlated. In words:

> "The ball is further right than I predicted. Since my position error is correlated with my velocity error, I was probably underestimating the speed too. Nudge velocity up."

That is how velocity gets learned from position-only measurements.

---

## 4. Watching velocity appear: a worked run

Here's the actual filter (1 axis, `[p, v]`) run with **your config values**: `DT = 1`, `MEAS_VAR = 25`, `INIT_POS_VAR = 25`, `INIT_VEL_VAR = 1000`, `PROCESS_VAR = 1`. The ball moves at a true 5 px/frame, and to keep it readable the detections are perfect: 0, 5, 10, 15, 20, 25.

Initialise on the first detection (z = 0) with zero velocity, as your `initialize()` docstring says:

```
x = [0, 0]          P = [[25,    0],
                         [ 0, 1000]]    ← "no idea how fast it's going"
```

| Frame | Predicted `[p, v]` | Measurement z | Gain K `[for p, for v]` | Updated `[p, v]` | Var(p), Var(v) after update |
|---|---|---|---|---|---|
| 1 | [0.00, 0.00] | 5 | [0.976, 0.953] | [4.88, 4.76] | 24.4, 47.9 |
| 2 | [9.64, 4.76] | 10 | [0.828, 0.497] | [9.94, 4.94] | 20.7, 13.0 |
| 3 | [14.88, 4.94] | 15 | [0.702, 0.309] | [14.96, 4.98] | 17.5, 6.0 |
| 4 | [19.94, 4.98] | 20 | [0.611, 0.221] | [19.98, 4.99] | 15.3, 3.8 |
| 5 | [24.97, 4.99] | 25 | [0.549, 0.178] | [24.99, 5.00] | 13.7, 3.1 |

Things to notice:

- **Frame 1, velocity jumps from 0 to 4.76 from a single position reading.** Why? The predict step turned the huge velocity variance (1000) into a huge position variance (≈1025) *and* a huge position–velocity covariance (≈1000). The predicted P was:
  ```
  [[1025.2, 1000.5],
   [1000.5, 1001.0]]
  ```
  That 1000.5 says "if position is off, velocity is off by about the same amount". So when the ball turned up 5 px further than predicted, the filter put almost all of that into velocity. This is the hidden-velocity mechanism in action.
- **K shrinks every frame.** As the filter gets more confident, it leans more on its own prediction and less on each new detection. That's what smooths out detector jitter.
- **Var(p) settles below 25**, the detector's own variance. The filtered position is more accurate than any single detection.
- By frame 5 the velocity estimate is essentially exact, and the innovation `y` is tiny (0.03 px) because the predictions are now almost perfect.

With real, noisy detections the numbers wobble, but the pattern is the same.

---

## 5. Occlusion: predict with no update

When the ball disappears, there's no measurement, so you **skip the update** and only predict. Continuing from frame 5 above:

| Frames hidden | Predicted p | Std of p (px) |
|---|---|---|
| 1 | 29.99 | 5.1 |
| 2 | 34.99 | 6.8 |
| 3 | 39.99 | 8.8 |
| 4 | 44.99 | 10.9 |
| 5 | 49.99 | 13.3 |

The prediction keeps moving the ball at 5 px/frame (that's the velocity state earning its keep), and the uncertainty grows each frame because Q keeps being added. In your visualisation this is the uncertainty ellipse ballooning while the ball is hidden. When the ball reappears, P is large, so K is large, and the filter snaps quickly back onto the detection.

This is the `COASTING` state in `tracker.py`: always predict, only update when a detection exists and passes the gate.

---

## 6. The full matrix version (what `kalman.py` does)

Same steps as the 1D example, with matrices instead of numbers:

**Predict**
```
x = F x                 move the state forward one frame
P = F P Fᵀ + Q          push uncertainty forward, then add process noise
```

`F P Fᵀ` is the matrix way of saying "if p = p + v·dt, then the uncertainty in v leaks into p". That's where the position–velocity covariance comes from.

**Update**
```
y = z - H x             innovation: measured minus predicted position (2D)
S = H P Hᵀ + R          innovation covariance: predicted position uncertainty + detector noise (2×2)
K = P Hᵀ S⁻¹            Kalman gain (4×2): how much each of the 4 states should move per pixel of surprise
x = x + K y             correct the state, including velocity
P = (I - K H) P         shrink the uncertainty
```

Side by side with 1D:

| 1D | Matrix | Meaning |
|---|---|---|
| `y = z - x` | `y = z - H x` | surprise |
| `P + R` | `S = H P Hᵀ + R` | total uncertainty in that surprise |
| `K = P / (P + R)` | `K = P Hᵀ S⁻¹` | how much to trust the surprise |
| `x + K y` | `x + K y` | correct |
| `(1 - K) P` | `(I - K H) P` | shrink uncertainty |

The only genuinely new things in the matrix version are H (to bridge 4D state and 2D measurement) and the off-diagonal entries of P (which let a position surprise correct velocity).

### Where each config number lands

| Config | Matrix | Effect if you raise it |
|---|---|---|
| `MEAS_VAR` | R | Trust detections less: smoother but laggier track |
| `PROCESS_VAR` | Q | Expect more unmodelled motion: follows sudden turns faster but jitters more |
| `INIT_POS_VAR`, `INIT_VEL_VAR` | initial P | How fast the filter "locks on" at the start |

The ratio of Q to R is what really matters: it decides how much the filter believes physics versus the detector.

### The gate (`innovation_distance`)

`y` tells you how surprised you are, and `S` tells you how surprised you *should* expect to be. The squared Mahalanobis distance `yᵀ S⁻¹ y` combines them into one number: "how many standard deviations off is this detection?" If it's above `GATE_THRESHOLD` (9.21, the 99% cutoff for 2 degrees of freedom), the detection is probably a false positive (some other green thing) and you skip the update for that frame.

---

## 7. Suggested order for your TODOs

1. **`F`**: use the 1D `[[1, dt], [0, 1]]` pattern, spread across the 4×4 using the `[0, 2]` / `[1, 3]` pairing.
2. **`initialize`**: position from z, velocity zero, P diagonal with the init variances.
3. **`predict`**: two lines, straight from section 6.
4. **`update`**: five lines from section 6. Use `np.linalg.solve` instead of `np.linalg.inv(S)` as the docstring suggests.
5. **Check against this document**: run your filter in 1D-style with perfect detections 0, 5, 10, 15, … along x (and y fixed) and you should reproduce the table in section 4 for the x components.
6. **`innovation_distance`**: reuses `y` and `S` from update.

Then `tests/test_kalman.py` will tell you if it's right.
