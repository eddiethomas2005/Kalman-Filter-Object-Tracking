"""Test the filter on synthetic data. No video, no OpenCV."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kalman import KalmanFilter  # noqa: E402


def make_track(n=100, v=(3.0, 1.5), noise_std=5.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n)[:, None]
    truth = np.array([50.0, 100.0]) + t * np.array(v)
    return truth, truth + rng.normal(0, noise_std, truth.shape)


def run(kf, meas, drop=()):
    out = []
    kf.initialize(meas[0])
    out.append(kf.position)
    for i in range(1, len(meas)):
        kf.predict()
        if i not in drop:
            kf.update(meas[i])
        out.append(kf.position)
    return np.array(out)


def new_kf():
    return KalmanFilter(dt=1.0, process_var=0.01, meas_var=25.0,
                        init_pos_var=25.0, init_vel_var=100.0)


def test_filter_smoother_than_measurements():
    truth, meas = make_track()
    est = run(new_kf(), meas)
    # skip the first 10 frames while velocity converges
    assert np.abs(est[10:] - truth[10:]).mean() < np.abs(meas[10:] - truth[10:]).mean()


def test_coasts_through_occlusion():
    truth, meas = make_track()
    est = run(new_kf(), meas, drop=range(40, 55))
    assert np.linalg.norm(est[54] - truth[54]) < 15.0


def test_uncertainty_grows_while_coasting():
    _, meas = make_track()
    kf = new_kf()
    run(kf, meas[:30])
    before = np.trace(kf.position_cov)
    for _ in range(10):
        kf.predict()
    assert np.trace(kf.position_cov) > before
