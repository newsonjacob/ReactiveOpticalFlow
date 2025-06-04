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
from uav.utils import get_drone_state
from sparse_optical_flow_utils import initialize_sparse_features, track_and_detect_obstacle

# GUI state holder
param_refs = {
    'state': [''],
    'reset_flag': [False]
}
start_gui(param_refs)

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

frame_count = 0
start_time = time.time()
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
os.makedirs("flow_logs", exist_ok=True)
log_file = open(f"flow_logs/sparse_log_{timestamp}.csv", 'w')
log_file.write("frame,time,speed,obstacle_detected,features_detected\n")

# Sparse optical flow state
prev_gray_sparse = None
prev_pts = None
roi = [60, 60, 580, 420]  # wider and more forgiving ROI

# Video writer
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter('sparse_flow_output.avi', fourcc, 8.0, (640, 480))

try:
    while not exit_flag[0]:
        frame_count += 1
        time_now = time.time()

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

        print(f"🖼 Frame {frame_count} captured and decoded")
        img = cv2.resize(img, (640, 480))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        vis_img = img.copy()

        # Sparse flow detection
        obstacle_sparse = False
        features_detected = 0
        if prev_gray_sparse is None:
            prev_gray_sparse = gray
            prev_pts = initialize_sparse_features(prev_gray_sparse)
            if prev_pts is not None:
                features_detected = len(prev_pts)
                print(f"🔍 Initialized {features_detected} features")
            print("🔧 First grayscale frame set")
        else:
            obstacle_sparse, prev_pts = track_and_detect_obstacle(prev_gray_sparse, gray, prev_pts, roi, displacement_threshold=2.5)

            prev_gray_sparse = gray.copy()
            if prev_pts is not None:
                features_detected = len(prev_pts)
                if features_detected < 10:
                    print("🔁 Too few features — reinitializing")
                    prev_pts = initialize_sparse_features(prev_gray_sparse)
                    if prev_pts is not None:
                        features_detected = len(prev_pts)

        # Skip obstacle reaction for early frames
        if frame_count < 20:
            obstacle_sparse = False

        # Navigation
        state_str = "forward"
        if obstacle_sparse:
            state_str = navigator.brake()
        else:
            state_str = navigator.blind_forward()

        param_refs['state'][0] = state_str

        # Overlay
        pos, yaw, speed = get_drone_state(client)
        print(
            f"🛰️ Pos({pos.x_val:.2f}, {pos.y_val:.2f}, {pos.z_val:.2f}) "
            f"Speed: {speed:.2f} m/s State: {state_str}"
        )
        if state_str == "blind_forward" and speed < 0.1:
            print("⚠️ Blind forward but speed is low — possible premature brake")
        cv2.rectangle(vis_img, (roi[0], roi[1]), (roi[2], roi[3]), (255, 0, 0), 1)
        if obstacle_sparse:
            cv2.putText(vis_img, "Obstacle!", (400, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.putText(vis_img, f"Frame: {frame_count}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.putText(vis_img, f"Speed: {speed:.2f}", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.putText(vis_img, f"State: {state_str}", (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.putText(vis_img, f"Sim Time: {time_now-start_time:.2f}s", (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        out.write(vis_img)
        log_file.write(f"{frame_count},{time_now:.2f},{speed:.2f},{obstacle_sparse},{features_detected}\n")

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
            log_file.write("frame,time,speed,obstacle_detected,features_detected\n")
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
