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


def track_and_detect_obstacle(prev_gray, curr_gray, prev_pts, roi,
                              dt=1.0, drone_speed=0.0, displacement_threshold=10):
    """
    Performs Lucas-Kanade tracking and returns:
    - obstacle_detected (bool)
    - new_points for continued tracking
    """
    new_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None, **lk_params)

    # Filter only good points
    good_old = prev_pts[status == 1]
    good_new = new_pts[status == 1]

    if len(good_new) < 5:
        return False, new_pts

    # Filter points in ROI
    roi_mask = (
        (good_new[:, 0] >= roi[0]) &
        (good_new[:, 1] >= roi[1]) &
        (good_new[:, 0] <= roi[2]) &
        (good_new[:, 1] <= roi[3])
    )

    roi_old = good_old[roi_mask]
    roi_new = good_new[roi_mask]

    if len(roi_new) < 5:
        return False, new_pts

    # Compute flow magnitude
    disp = roi_new - roi_old
    magnitudes = np.linalg.norm(disp, axis=1)
    avg_mag = np.mean(magnitudes)

    # Normalize by frame time
    if dt > 0:
        avg_mag /= dt

    # Clamp speed to avoid instability
    effective_speed = max(drone_speed, 0.2)
    threshold = displacement_threshold * effective_speed

    print(f"[DEBUG] ROI avg flow: {avg_mag:.2f}, Threshold: {threshold:.2f}, Speed: {drone_speed:.2f}")

    if avg_mag > threshold:
        return True, new_pts

    return False, new_pts

