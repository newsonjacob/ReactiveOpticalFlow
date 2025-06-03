import airsim
import cv2
import numpy as np
import time
from datetime import datetime
import os
import subprocess
import math
from airsim import ImageRequest, ImageType

from uav.interface import exit_flag, start_gui
from uav.perception import OpticalFlowTracker, FlowHistory
from uav.navigation import Navigator
from uav.utils import get_drone_state

# GUI parameter and status holders
brake_base = [35.0]
dodge_base = [5.0]
param_refs = {
    'brake': [brake_base],
    'dodge': [dodge_base],
    'L': [0.0],
    'C': [0.0],
    'R': [0.0],
    'state': [''],
    'reset_flag': [False]
}

start_gui(param_refs)

# === LAUNCH UE4 SIMULATION ===
ue4_exe = r"C:\Users\newso\Documents\AirSimExperiments\BlocksBuild\WindowsNoEditor\Blocks\Binaries\Win64\Blocks.exe"
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

# After takeoff
client.takeoffAsync().join()
client.moveToPositionAsync(0, 0, -2, 2).join()

feature_params = dict(maxCorners=75, qualityLevel=0.1, minDistance=5, blockSize=5)
lk_params = dict(winSize=(15, 15), maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

tracker = OpticalFlowTracker(lk_params, feature_params)
flow_history = FlowHistory(size=5)
navigator = Navigator(client)

frame_count = 0
start_time = time.time()
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
os.makedirs("flow_logs", exist_ok=True)
log_file = open(f"flow_logs/full_log_{timestamp}.csv", 'w')
log_file.write("frame,time,features,flow_left,flow_center,flow_right,speed,simgetimage_s,decode_s,processing_s,loop_s\n")

# Video writer setup
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter('flow_output.avi', fourcc, 8.0, (640, 480))

last_vis_img = np.zeros((480, 640, 3), dtype=np.uint8)  # Black frame as fallback

target_fps = 20
frame_duration = 1.0 / target_fps

# Add this before your main loop
fps_list = []
img = None  # Add this before your main loop

try:
    while not exit_flag[0]:
        frame_count += 1
        loop_start = time.time()
        time_now = time.time()  # <-- Add this line

        # --- Get image from AirSim ---
        t0 = time.time()
        responses = client.simGetImages([
            ImageRequest("oakd_camera", ImageType.Scene, False, True)  # compress=True for JPEG
        ])
        response = responses[0]
        if response.width == 0 or response.height == 0 or len(response.image_data_uint8) == 0:
            out.write(last_vis_img)
            continue

        img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
        img = cv2.imdecode(img1d, cv2.IMREAD_COLOR)
        t1 = time.time()
        if img is None:
            out.write(last_vis_img)
            continue

        img = cv2.resize(img, (640, 480))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        vis_img = img.copy()
        last_vis_img = vis_img  # Update after a valid frame

        if frame_count == 1:
            tracker.initialize(gray)
            print("Initialized optical flow tracker.")
            continue

        # Manual nudge at frame 5 to test flow responsiveness
        if frame_count == 5:
            print("🔧 Manual nudge forward for test")
            client.moveByVelocityAsync(2, 0, 0, 2).join()

        # --- Optical flow processing ---
        good_old, flow_vectors, flow_std, dt = tracker.process_frame(gray, start_time)
        # Convert raw pixel displacements to velocities using the returned dt
        magnitudes = np.linalg.norm(flow_vectors, axis=1) / max(dt, 1e-6)
        h, w = gray.shape
        good_old = good_old.reshape(-1, 2)  # Ensure proper shape
        x_coords = good_old[:, 0]
        y_coords = good_old[:, 1]

        left_mag = np.mean(magnitudes[x_coords < w // 3]) if np.any(x_coords < w // 3) else 0
        center_band = (x_coords >= w // 3) & (x_coords < 2 * w // 3)
        right_mag = np.mean(magnitudes[x_coords >= 2 * w // 3]) if np.any(x_coords >= 2 * w // 3) else 0
        probe_band = y_coords < h // 3
        probe_mag = np.mean(magnitudes[center_band & probe_band]) if np.any(center_band & probe_band) else 0
        center_mag = np.mean(magnitudes[center_band]) if np.any(center_band) else 0

        flow_history.update(left_mag, center_mag, right_mag)
        smooth_L, smooth_C, smooth_R = flow_history.average()
        param_refs['L'][0] = smooth_L
        param_refs['C'][0] = smooth_C
        param_refs['R'][0] = smooth_R

        # Draw flow arrows for visualization
        for i, (p1, vec) in enumerate(zip(good_old, flow_vectors)):
            if i > 50:
                break
            x1, y1 = int(p1[0]), int(p1[1])
            vec = np.ravel(vec)  # Ensure vec is 1D
            if vec.shape[0] >= 2:
                dx, dy = float(vec[0]), float(vec[1])
            else:
                dx, dy = 0.0, 0.0
            x2, y2 = int(x1 + dx), int(y1 + dy)
            cv2.arrowedLine(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 1, tipLength=0.3)

        # Overlay info
        pos, yaw, speed = get_drone_state(client)
        cv2.putText(vis_img, f"Frame: {frame_count}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.putText(vis_img, f"Speed: {speed:.2f}", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.putText(vis_img, f"State: {param_refs['state'][0]}", (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.putText(vis_img, f"Sim Time: {time_now-start_time:.2f}s", (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        # === Navigation logic ===
        state_str = "none"
        if len(good_old) < 5:
            if smooth_L > 1.5 and smooth_R > 1.5 and smooth_C < 0.2:
                state_str = navigator.brake()
            else:
                state_str = navigator.blind_forward()
        else:
            pos, yaw, speed = get_drone_state(client)
            brake_thres = 35 + 15 * speed
            dodge_thres = 5 + 3 * speed
            probe_req = 15 + 3 * speed

            center_high = smooth_C > dodge_thres
            side_diff = abs(smooth_L - smooth_R)
            side_safe = side_diff > 10 and (smooth_L < 100 or smooth_R < 100)

            if smooth_C > brake_thres:
                state_str = navigator.brake()
            elif probe_mag < 0.1 and center_mag > 0.7:
                print("⚠️ No probe flow but high center flow — possible wall ahead")
                state_str = navigator.brake()
            # Try regular dodge if side difference exists
            if center_high and side_safe:
                state_str = navigator.dodge(smooth_L, smooth_C, smooth_R)
            # Fallback: high center flow and low probe => flat wall straight ahead
            elif probe_mag < 0.5 and center_mag > 0.7:
                print("🟥 Flat wall detected — attempting fallback dodge")
                state_str = navigator.dodge(smooth_L, smooth_C, smooth_R)
            elif (navigator.braked or navigator.dodging) and smooth_C < 10 and smooth_L < 10 and smooth_R < 10:
                state_str = navigator.resume_forward()
            elif not navigator.braked and not navigator.dodging and time_now - navigator.last_movement_time > 2:
                state_str = navigator.reinforce()
            elif (navigator.braked or navigator.dodging) and speed < 0.2 and smooth_C < 5 and smooth_L < 5 and smooth_R < 5:
                state_str = navigator.nudge()
            elif time_now - navigator.last_movement_time > 4:
                state_str = navigator.timeout_recover()

        param_refs['state'][0] = state_str

        # === Reset logic from GUI ===
        if param_refs['reset_flag'][0]:
            print("🔄 Resetting simulation...")
            try:
                client.landAsync().join()
                client.reset()
                client.enableApiControl(True)
                client.armDisarm(True)
                client.takeoffAsync().join()
                client.moveToPositionAsync(0, 0, -2, 2).join()
            except Exception as e:
                print("Reset error:", e)

            tracker.initialize(gray)
            flow_history = FlowHistory(size=5)
            navigator = Navigator(client)
            frame_count = 0
            param_refs['reset_flag'][0] = False

            # === Reset log file ===
            log_file.close()
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = open(f"flow_logs/full_log_{timestamp}.csv", 'w')
            log_file.write("frame,time,features,flow_left,flow_center,flow_right,speed,simgetimage_s,decode_s,processing_s,loop_s\n")

            # === Reset video writer ===
            out.release()
            out = cv2.VideoWriter('flow_output.avi', fourcc, 8.0, (640, 480))
            continue

        out.write(vis_img)

        # === Logging ===
        log_file.write(
            f"{frame_count},{time_now:.2f},{len(good_old)},{smooth_L:.3f},{smooth_C:.3f},{smooth_R:.3f},{speed:.2f},"
            f"{t1-t0:.3f},{t1-t0:.3f},0.0,{time.time()-loop_start:.3f}\n"
        )

        # Throttle loop to target FPS
        elapsed = time.time() - time_now
        if elapsed < frame_duration:
            time.sleep(frame_duration - elapsed)

        # After processing each frame, before throttling
        actual_fps = 1 / (time.time() - loop_start)
        fps_list.append(actual_fps)
        print(f"Actual FPS: {actual_fps:.2f}")
        print(f"Features detected: {len(good_old)}")

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

    # Re-encode video at median FPS using OpenCV
    import statistics
    if len(fps_list) > 0:
        median_fps = statistics.median(fps_list)
        print(f"Median FPS: {median_fps:.2f}")

        input_video = 'flow_output.avi'
        output_video = 'flow_output_fixed.avi'

        cap = cv2.VideoCapture(input_video)
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out_fixed = cv2.VideoWriter(output_video, fourcc, median_fps, (640, 480))

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            out_fixed.write(frame)
            frame_count += 1

        cap.release()
        out_fixed.release()
        print(f"Re-encoded {frame_count} frames at {median_fps:.2f} FPS to {output_video}")

