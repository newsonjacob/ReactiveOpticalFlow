# UAV Optical Flow Navigation System

This project implements a real-time **sparse** optical flow-based navigation system for an unmanned aerial vehicle (UAV) in the AirSim simulation environment. The UAV uses computer vision to detect motion and brake when an obstacle is detected in a region of interest (ROI).

## Features

* 🧠 Sparse Lucas-Kanade optical flow with CLAHE enhancement
* ✈️ Basic navigation logic: brake when an obstacle is detected, otherwise continue forward
* 🪟 GUI controls to reset the simulation or stop the UAV
* 📁 Structured modular code with reusable components
* ▶️ Automatically launches the Unreal Engine Blocks environment

## Project Structure

```
ReactiveOpticalFlow/
├── main.py               # Entry point
├── airsim/               # Minimal AirSim Python client
├── uav/
│   ├── perception.py     # Optical flow tracking utilities
│   ├── navigation.py     # Motion commands
│   ├── interface.py      # GUI controls
│   └── utils.py          # Helper functions
├── flow_logs/            # CSV logs of each run
└── README.txt            # You're here!
```

## How It Works

1. **Startup phase**: The UAV takes off and a few initial frames are ignored.
2. **Tracking**: Shi–Tomasi features are tracked frame to frame using Lucas–Kanade optical flow.
3. **Obstacle detection**: Feature displacement inside an ROI is checked against a threshold.
4. **Navigation**: The Navigator brakes if an obstacle is detected, otherwise it moves forward.
5. **Logging**: Frame number, time, speed, obstacle flag and feature count are written to a CSV file.

## Requirements

* Python 3.8+
* AirSim installed and configured
* OpenCV (`pip install opencv-python`)
* NumPy (`pip install numpy`)

## Running the Simulation

1. Launch the AirSim environment or let `main.py` start the Blocks executable configured inside the script.
2. Run the program:

   ```bash
   python main.py
   ```
3. Use the GUI window to reset or stop the simulation.

## Example Log Format

```
frame,time,speed,obstacle_detected,features_detected
42,13.23,1.50,True,63
```

## Future Improvements

* Add SLAM integration
* Visualization of flow vectors in real time
* Command line argument parsing for tuning thresholds
* ROS bridge for physical deployment
