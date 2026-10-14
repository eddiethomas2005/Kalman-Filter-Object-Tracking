"""Entry point: python main.py data/ball.mp4 --detector color --out output/tracked.mp4"""

import argparse
import csv
import cv2
import os
import config
from detector import BackgroundSubtractorDetector, ColorDetector
from kalman import KalmanFilter
from tracker import Tracker
from visualize import draw_frame, plot_trajectories


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("source", help="video path, or 0 for webcam")
    p.add_argument("--detector", choices=["color", "bgsub"], default="color")
    p.add_argument("--out", default="output/tracked.mp4")
    p.add_argument("--fake-occlusion", nargs=2, type=int, metavar=("START", "END"),
                   help="drop detections for this frame range to demo coasting")
    p.add_argument("--no-display", action="store_true")
    return p.parse_args()


def build_tracker(kind):
    if kind == "color":
        det = ColorDetector(config.HSV_LOWER, config.HSV_UPPER, config.MIN_AREA)
    else:
        det = BackgroundSubtractorDetector(config.MIN_AREA)
    kf = KalmanFilter(config.DT, config.PROCESS_VAR, config.MEAS_VAR,
                      config.INIT_POS_VAR, config.INIT_VEL_VAR)
    return Tracker(det, kf, config.MAX_COAST_FRAMES, config.GATE_THRESHOLD)


def main():
    args = parse_args()
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    tracker = build_tracker(args.detector)
    results, writer = [], None

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx = tracker.frame_idx
        occluded = bool(args.fake_occlusion) and args.fake_occlusion[0] <= idx <= args.fake_occlusion[1]
        result = tracker.step(frame, force_miss=occluded)
        results.append(result)
        vis = draw_frame(frame, result, results)

        if writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps==0:
                fps=30
            (height , width)=vis.shape[:2]
            os.makedirs(os.path.dirname(args.out), exist_ok=True)
            writer = cv2.VideoWriter(args.out, fourcc, fps, (width, height))
        
        writer.write(vis)




        if not args.no_display:
            cv2.imshow("tracker", vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    os.makedirs("output", exist_ok=True)
    with open("output/log.csv", "w", newline="") as f:
        log = csv.writer(f)
        log.writerow(["frame", "det_x", "det_y", "est_x", "est_y", "state"])
        for r in results:
            det_x, det_y = r.detection if r.detection is not None else ("", "")
            est_x, est_y = r.position if r.position is not None else ("", "")
            log.writerow([r.frame_idx, det_x, det_y, est_x, est_y, r.state.name])
    plot_trajectories(results, "output/trajectory.png")


if __name__ == "__main__":
    main()
