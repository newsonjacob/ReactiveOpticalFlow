import cv2
import numpy as np
from uav.utils import apply_clahe, partition_roi

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
    gray_frame = apply_clahe(gray_frame)
    return cv2.goodFeaturesToTrack(gray_frame, mask=None, **shitomasi_params)


def track_and_detect_obstacle(prev_gray, curr_gray, prev_pts, roi,
                              partitions=1, dt=1.0, drone_speed=0.0,
                              displacement_threshold=10):
    """
    Performs Lucas-Kanade tracking and returns:
    - obstacle_detected (bool)
    - new_points for continued tracking
    - good_old: valid points from the previous frame
    - good_new: corresponding new positions of valid points
    """
    prev_gray = apply_clahe(prev_gray)
    curr_gray = apply_clahe(curr_gray)

    new_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None, **lk_params)

    # Filter only good points
    good_old = prev_pts[status == 1]
    good_new = new_pts[status == 1]

    if len(good_new) < 5:
        return False, new_pts, good_old, good_new, [0.0] * partitions

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
        return False, new_pts, good_old, good_new, [0.0] * partitions

    # Compute flow magnitude
    disp = roi_new - roi_old
    magnitudes = np.linalg.norm(disp, axis=1)
    avg_mag = np.mean(magnitudes)

    # Flow by partition
    partition_avgs = []
    for px1, py1, px2, py2 in partition_roi(roi, partitions):
        mask = (
            (roi_new[:, 0] >= px1) &
            (roi_new[:, 0] < px2) &
            (roi_new[:, 1] >= py1) &
            (roi_new[:, 1] <= py2)
        )
        part_old = roi_old[mask]
        part_new = roi_new[mask]
        if len(part_new) == 0:
            partition_avgs.append(0.0)
            continue
        part_disp = part_new - part_old
        part_mag = np.linalg.norm(part_disp, axis=1)
        part_avg = np.mean(part_mag)
        partition_avgs.append(part_avg)

    # Normalize by frame time
    if dt > 0:
        avg_mag /= dt
        partition_avgs = [p / dt for p in partition_avgs]

    # Clamp speed to avoid instability
    effective_speed = max(drone_speed, 0.2)
    threshold = displacement_threshold * effective_speed

    print(f"[DEBUG] ROI avg flow: {avg_mag:.2f}, Threshold: {threshold:.2f}, Speed: {drone_speed:.2f}")

    obstacle_detected = False
    if partitions >= 3:
        center_flow = partition_avgs[partitions // 2]
        obstacle_detected = center_flow > threshold
    else:
        obstacle_detected = avg_mag > threshold

    return obstacle_detected, new_pts, good_old, good_new, partition_avgs
