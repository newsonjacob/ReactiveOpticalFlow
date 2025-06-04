import airsim
import cv2
import numpy as np
import time
from datetime import datetime
import os
import subprocess
from airsim import ImageRequest, ImageType

from uav.interface import exit_flag, start_gui
from uav.navigation import Navigator
from uav.utils import get_drone_state, partition_roi
from uav.perception import FlowHistory
from uav.logging import debug_print
from sparse_optical_flow_utils import initialize_sparse_features, track_and_detect_obstacle

# GUI state holder
param_refs = {
    'state': [''],
    'reset_flag': [False]
}
start_gui(param_refs)

# Display debug images if environment variable is set
DEBUG_DISPLAY = os.environ.get("DEBUG_DISPLAY", "0") == "1"

# === Launch Unreal Engine simulation ===
# Path to the Blocks executable. This can be overridden by setting the
# BLOCKS_EXE_PATH environment variable.
ue4_exe = os.environ.get(
    "BLOCKS_EXE_PATH",
    r"C:\Users\newso\Documents\AirSimExperiments\BlocksBuild\WindowsNoEditor\Blocks\Binaries\Win64\Blocks.exe",
)
try:
    sim_process = subprocess.Popen([ue4_exe, "-windowed", "-ResX=1280", "-ResY=720"])
    print("Launching Unreal Engine simulation...")
    time.sleep(5)
except Exception as e:
    print("Failed to launch UE4:", e)

client = airsim.MultirotorClient()
client.confirmConnection()
print("Connected!")
client.enableApiControl(True)
client.armDisarm(True)
client.takeoffAsync().join()
client.moveToPositionAsync(0, 0, -2, 2).join()

navigator = Navigator(client)

GRACE_FRAMES = 30  # ignore obstacle logic for startup period
NO_FEATURE_LIMIT = 10
no_feature_frames = 0
PARTITIONS = 3
SAFE_FRAMES = 5  # frames without obstacles before resuming after a brake
safe_counter = 0
flow_history = FlowHistory(alpha=0.5)
smooth_L = smooth_C = smooth_R = 0.0

frame_count = 0
start_time = time.time()
prev_time = None
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
os.makedirs("flow_logs", exist_ok=True)
log_file = open(f"flow_logs/sparse_log_{timestamp}.csv", 'w')
log_file.write(
    "frame,time,speed,obstacle_detected,features_detected,flow_left,flow_center,flow_right,state,safe_counter\n"
)

# Sparse optical flow state
prev_gray_sparse = None
prev_pts = None
roi = [60, 60, 580, 420]  # wider and more forgiving ROI
roi_parts = partition_roi(roi, PARTITIONS)

# Video writer
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter('sparse_flow_output.avi', fourcc, 8.0, (640, 480))

try:
    while not exit_flag[0]:
        frame_count += 1
        time_now = time.time()
        dt = 0.0 if prev_time is None else time_now - prev_time
        prev_time = time_now
        pos, yaw, speed = get_drone_state(client)

        responses = client.simGetImages([
            ImageRequest("oakd_camera", ImageType.Scene, False, True)
        ])
        response = responses[0]
        if response.width == 0 or len(response.image_data_uint8) == 0:
            print("⚠️ Empty image response")
            continue

        img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
        img = cv2.imdecode(img1d, cv2.IMREAD_COLOR)
        if img is None:
            print("❌ Failed to decode image")
            continue

        debug_print(f"🖼 Frame {frame_count} captured and decoded")
        img = cv2.resize(img, (640, 480))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        vis_img = img.copy()
        good_old = np.empty((0, 2), dtype=np.float32)
        good_new = np.empty((0, 2), dtype=np.float32)
        part_flows = [0.0] * PARTITIONS

        # Sparse flow detection
        obstacle_sparse = False
        features_detected = 0
        if prev_gray_sparse is None:
            prev_gray_sparse = gray
            prev_pts = initialize_sparse_features(prev_gray_sparse)
            if prev_pts is not None:
                features_detected = len(prev_pts)
                debug_print(f"🔍 Initialized {features_detected} features")
            debug_print("🔧 First grayscale frame set")
        else:
            obstacle_sparse, prev_pts, good_old, good_new, part_flows = track_and_detect_obstacle(
                prev_gray_sparse,
                gray,
                prev_pts,
                roi,
                partitions=PARTITIONS,
                dt=dt,
                drone_speed=speed,
                displacement_threshold=2.5,
            )

            prev_gray_sparse = gray.copy()
            if prev_pts is not None:
                features_detected = len(prev_pts)
                if features_detected < 10:
                    debug_print("🔁 Too few features — reinitializing")
                    prev_pts = initialize_sparse_features(prev_gray_sparse)
                    if prev_pts is not None:
                        features_detected = len(prev_pts)

        debug_print(f"📈 Features detected: {features_detected}")
        if features_detected == 0:
            no_feature_frames += 1
        else:
            no_feature_frames = 0

        if part_flows:
            flow_history.update(*part_flows)
        smooth_L, smooth_C, smooth_R = flow_history.average()
        debug_print(
            f"[DEBUG] smoothed flows L/C/R: {smooth_L:.2f}, "
            f"{smooth_C:.2f}, {smooth_R:.2f}"
        )

        if no_feature_frames >= NO_FEATURE_LIMIT:
            debug_print("❌ No features for several frames — resetting tracker")
            prev_gray_sparse = gray
            prev_pts = initialize_sparse_features(prev_gray_sparse)
            no_feature_frames = 0

        threshold = 2.5 * max(speed, 0.2)
        corridor = (
            smooth_C <= threshold
            and smooth_L > threshold
            and smooth_R > threshold
        )

        if frame_count < GRACE_FRAMES:
            obstacle_sparse = False
        else:
            obstacle_sparse = smooth_C > threshold
            if corridor:
                obstacle_sparse = False

        threshold = 2.5 * max(speed, 0.2)
        obstacle_sparse = smooth_C > threshold
        corridor = (
            smooth_C <= threshold
            and smooth_L > threshold
            and smooth_R > threshold
        )
        if corridor:
            obstacle_sparse = False

        # Navigation
        state_str = "forward"
        if obstacle_sparse:
            safe_counter = 0
            state_str = navigator.dodge(smooth_L, smooth_C, smooth_R)
        else:
            if navigator.braked or navigator.dodging:
                safe_counter += 1
                debug_print(f"[DEBUG] clear frames: {safe_counter}/{SAFE_FRAMES}")

                if safe_counter >= SAFE_FRAMES:
                    state_str = navigator.resume_forward()
                    safe_counter = 0
                else:
                    if navigator.braked:
                        state_str = navigator.brake()
                    else:
                        state_str = "dodge"
            else:
                state_str = navigator.blind_forward()

        param_refs['state'][0] = state_str

        # Overlay
        debug_print(
            f"🛰️ Pos({pos.x_val:.2f}, {pos.y_val:.2f}, {pos.z_val:.2f}) "
            f"Speed: {speed:.2f} m/s State: {state_str}"
        )
        if state_str == "blind_forward" and speed < 0.1:
            debug_print("⚠️ Blind forward but speed is low — possible premature brake")
        cv2.rectangle(vis_img, (roi[0], roi[1]), (roi[2], roi[3]), (255, 0, 0), 1)
        for part in roi_parts:
            cv2.rectangle(vis_img, (part[0], part[1]), (part[2], part[3]), (0, 0, 255), 1)
        if obstacle_sparse:
            cv2.putText(vis_img, "Obstacle!", (400, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.putText(vis_img, f"Frame: {frame_count}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.putText(vis_img, f"Speed: {speed:.2f}", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.putText(vis_img, f"State: {state_str}", (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.putText(vis_img, f"Sim Time: {time_now-start_time:.2f}s", (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.putText(vis_img, f"Features: {features_detected}", (10, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.putText(vis_img, f"Flow L: {smooth_L:.2f}", (10, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.putText(vis_img, f"Flow C: {smooth_C:.2f}", (10, 205), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.putText(vis_img, f"Flow R: {smooth_R:.2f}", (10, 235), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        # Draw flow vectors
        for pt_old, pt_new in zip(good_old, good_new):
            x1, y1 = pt_old.ravel()
            x2, y2 = pt_new.ravel()
            cv2.arrowedLine(vis_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1, tipLength=0.3)
            cv2.circle(vis_img, (int(x2), int(y2)), 2, (0, 255, 0), -1)

        if DEBUG_DISPLAY and prev_pts is not None:
            for p in prev_pts:
                x, y = p.ravel()
                cv2.circle(vis_img, (int(x), int(y)), 2, (0, 255, 0), -1)
            cv2.imshow("debug", vis_img)
            cv2.waitKey(1)

        out.write(vis_img)
        log_file.write(
            f"{frame_count},{time_now:.2f},{speed:.2f},{obstacle_sparse},{features_detected},{smooth_L:.2f},{smooth_C:.2f},{smooth_R:.2f},{state_str},{safe_counter}\n"
        )

        if param_refs['reset_flag'][0]:
            print("🔄 Resetting simulation...")
            client.landAsync().join()
            client.reset()
            client.enableApiControl(True)
            client.armDisarm(True)
            client.takeoffAsync().join()
            client.moveToPositionAsync(0, 0, -2, 2).join()
            prev_gray_sparse = None
            prev_pts = None
            frame_count = 0
            param_refs['reset_flag'][0] = False
            log_file.close()
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = open(f"flow_logs/sparse_log_{timestamp}.csv", 'w')
            log_file.write(
                "frame,time,speed,obstacle_detected,features_detected,flow_left,flow_center,flow_right,state,safe_counter\n"
            )
            out.release()
            out = cv2.VideoWriter('sparse_flow_output.avi', fourcc, 8.0, (640, 480))

except KeyboardInterrupt:
    print("Interrupted.")

finally:
    print("Landing...")
    log_file.close()
    out.release()
    try:
        client.landAsync().join()
        client.armDisarm(False)
        client.enableApiControl(False)
    except Exception as e:
        print("Landing error:", e)
    if sim_process:
        sim_process.terminate()
        print("UE4 simulation closed.")
    if DEBUG_DISPLAY:
        cv2.destroyAllWindows()
