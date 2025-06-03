# uav/perception.py
import cv2
import numpy as np
import time

class FlowHistory:
    def __init__(self, size=5, alpha=0.5):
        self.size = size
        self.alpha = alpha
        self.smooth = None

    def update(self, left, center, right):
        new = np.array([left, center, right])
        if self.smooth is None:
            self.smooth = new
        else:
            self.smooth = self.alpha * new + (1 - self.alpha) * self.smooth

    def average(self):
        if self.smooth is None:
            return 0.0, 0.0, 0.0
        return tuple(self.smooth)

class OpticalFlowTracker:
    def __init__(self, lk_params, feature_params):
        self.lk_params = lk_params
        self.feature_params = feature_params
        self.prev_gray = None
        self.prev_pts = None
        self.prev_time = time.time()

    def initialize(self, gray_frame):
        self.prev_gray = gray_frame
        self.prev_pts = cv2.goodFeaturesToTrack(gray_frame, mask=None, **self.feature_params)
        self.prev_time = time.time()

    def process_frame(self, gray, _unused_start_time):  # ignore external time
        """Process a new grayscale frame.

        Returns
        -------
        tuple
            A tuple ``(points, flow, flow_std, dt)`` where ``points`` are the
            feature locations from the previous frame, ``flow`` is the raw pixel
            displacement for those features, ``flow_std`` is the standard
            deviation of the per-second flow magnitudes and ``dt`` is the time
            difference to the previous frame in seconds.  Returning ``dt`` allows
            callers to correctly scale the flow vectors.
        """

        if self.prev_gray is None or self.prev_pts is None:
            self.initialize(gray)
            return np.array([]), np.array([]), 0.0, 0.0

        next_pts, status, err = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, self.prev_pts, None, **self.lk_params)

        if next_pts is None or status is None:
            self.initialize(gray)
            return np.array([]), np.array([]), 0.0, 0.0

        good_old = self.prev_pts[status.flatten() == 1]
        good_new = next_pts[status.flatten() == 1]

        current_time = time.time()
        dt = max(current_time - self.prev_time, 1e-6)  # avoid div by zero
        self.prev_time = current_time

        self.prev_gray = gray
        self.prev_pts = cv2.goodFeaturesToTrack(gray, mask=None, **self.feature_params)

        if len(good_old) == 0:
            return np.array([]), np.array([]), 0.0, dt

        flow_vectors = good_new - good_old
        magnitudes = np.linalg.norm(flow_vectors, axis=1) / dt  # pixels/sec
        flow_std = np.std(magnitudes)

        return good_old, flow_vectors, flow_std, dt


