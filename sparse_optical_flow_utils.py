import cv2
import numpy as np

# Parameters
# Tune Shi-Tomasi parameters so that more features are detected from the
# very first frame.  A lower quality level allows weaker corners to be
# returned while a higher maxCorners value ensures we don't hit the
# feature limit too early.
shitomasi_params = dict(maxCorners=200, qualityLevel=0.02, minDistance=7)
lk_params = dict(winSize=(15, 15), maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))


def initialize_sparse_features(gray_frame):
    """
    Run Shi-Tomasi corner detection on the first grayscale frame.
    """
    return cv2.goodFeaturesToTrack(gray_frame, mask=None, **shitomasi_params)


def track_and_detect_obstacle(prev_gray, curr_gray, prev_pts, roi, displacement_threshold=5):
    """
    Performs Lucas-Kanade tracking and returns:
    - obstacle_detected (bool)
    - new_points for continued tracking
    """
    new_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None, **lk_params)

    # Filter only good points
    good_old = prev_pts[status == 1]
    good_new = new_pts[status == 1]

    if len(good_new) < 2:
        return False, new_pts  # Not enough points to analyze

    # Filter points inside ROI
    roi_corners = good_new[
        (good_new[:, 0] >= roi[0]) &
        (good_new[:, 1] >= roi[1]) &
        (good_new[:, 0] <= roi[2]) &
        (good_new[:, 1] <= roi[3])
    ]

    if len(roi_corners) < 2:
        return False, new_pts

    # Mean motion of ROI features
    dx = np.mean(roi_corners[:, 0]) - roi_corners[0, 0]
    dy = np.mean(roi_corners[:, 1]) - roi_corners[0, 1]

    if abs(dx) > displacement_threshold or abs(dy) > displacement_threshold:
        return True, new_pts

    return False, new_pts
