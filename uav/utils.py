# uav/utils.py
import math
import cv2
import numpy as np
import airsim

def apply_clahe(gray_image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray_image)

def get_yaw(orientation):
    return math.degrees(airsim.to_eularian_angles(orientation)[2])

def get_speed(velocity):
    return np.linalg.norm([velocity.x_val, velocity.y_val, velocity.z_val])

def get_drone_state(client):
    state = client.getMultirotorState()
    pos = state.kinematics_estimated.position
    ori = state.kinematics_estimated.orientation
    yaw = get_yaw(ori)
    vel = state.kinematics_estimated.linear_velocity
    speed = get_speed(vel)
    return pos, yaw, speed, vel


def partition_roi(roi, parts):
    """Split a ROI into equal vertical partitions.

    Parameters
    ----------
    roi : sequence
        ``(x1, y1, x2, y2)`` coordinates of the region of interest.
    parts : int
        Number of vertical partitions to create.

    Returns
    -------
    list of tuple
        List of partition ROIs ``[(x1, y1, x2, y2), ...]``.
    """
    x1, y1, x2, y2 = roi
    width = x2 - x1
    part_w = width // parts
    partitions = []
    for i in range(parts):
        px1 = x1 + i * part_w
        px2 = x1 + (i + 1) * part_w if i < parts - 1 else x2
        partitions.append((px1, y1, px2, y2))
    return partitions

