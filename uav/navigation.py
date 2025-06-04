# uav/navigation.py
import time
import math
import airsim

class Navigator:
    def __init__(self, client):
        self.client = client
        self.braked = False
        self.dodging = False
        self.last_movement_time = time.time()

    def get_state(self):
        state = self.client.getMultirotorState()
        pos = state.kinematics_estimated.position
        ori = state.kinematics_estimated.orientation
        yaw = math.degrees(airsim.to_eularian_angles(ori)[2])
        vel = state.kinematics_estimated.linear_velocity
        speed = math.sqrt(vel.x_val ** 2 + vel.y_val ** 2 + vel.z_val ** 2)
        return pos, yaw, speed

    def brake(self):
        print("🛑 Braking")
        self.client.moveByVelocityAsync(0, 0, 0, 1).join()
        self.braked = True
        return "brake"

    def dodge(self, smooth_L, smooth_C, smooth_R):
        if smooth_L < smooth_R - 10:
            direction = "left"
        elif smooth_R < smooth_L - 10:
            direction = "right"
        else:
            print("❌ Dodge ambiguous — skipping")
            return "no_dodge"

        lateral = 1.0 if direction == "right" else -1.0
        strength = 0.5 if max(smooth_L, smooth_R) > 100 else 1.0

        # Cut existing motion before dodge
        self.client.moveByVelocityBodyFrameAsync(0, 0, 0, 0.2).join()  # brief stop

        # Decide forward speed
        forward_speed = 0.0 if smooth_C > 1.0 else 0.3

        print(f"🔀 Dodging {direction} (strength {strength:.1f}, forward {forward_speed:.1f})")
        self.client.moveByVelocityBodyFrameAsync(
            forward_speed,
            lateral * strength,
            0,
            2.0
        ).join()

        self.dodging = True
        self.braked = False
        self.last_movement_time = time.time()
        return f"dodge_{direction}"


    def resume_forward(self):
        print("✅ Resuming forward motion")
        self.client.moveByVelocityAsync(2, 0, 0, duration=3,
            drivetrain=airsim.DrivetrainType.ForwardOnly,
            yaw_mode=airsim.YawMode(False, 0))
        self.braked = False
        self.dodging = False
        self.last_movement_time = time.time()
        return "resume"

    def blind_forward(self):
        pos, _, speed = self.get_state()
        print(
            f"⚠️ No features — continuing blind forward motion \n"
            f"   Start pos ({pos.x_val:.2f}, {pos.y_val:.2f}, {pos.z_val:.2f})"
            f" speed {speed:.2f}"
        )
        self.client.moveByVelocityAsync(
            4,
            0,
            0,
            duration=2,
            drivetrain=airsim.DrivetrainType.ForwardOnly,
            yaw_mode=airsim.YawMode(False, 0),
        )
        pos_after, _, speed_after = self.get_state()
        print(
            f"   After command pos ({pos_after.x_val:.2f}, {pos_after.y_val:.2f},"
            f" {pos_after.z_val:.2f}) speed {speed_after:.2f}"
        )
        self.last_movement_time = time.time()
        if speed_after < 0.1:
            print("   ⚠️ Blind forward issued but UAV not moving")
        return "blind_forward"

    def nudge(self):
        print("⚠️ Low flow + zero velocity — nudging forward")
        self.client.moveByVelocityAsync(0.5, 0, 0, 1).join()
        self.last_movement_time = time.time()
        return "nudge"

    def reinforce(self):
        print("🔁 Reinforcing forward motion")
        self.client.moveByVelocityAsync(2, 0, 0, duration=3,
            drivetrain=airsim.DrivetrainType.ForwardOnly,
            yaw_mode=airsim.YawMode(False, 0))
        self.last_movement_time = time.time()
        return "resume_reinforce"

    def timeout_recover(self):
        print("⏳ Timeout — forcing recovery motion")
        self.client.moveByVelocityAsync(0.5, 0, 0, 1).join()
        self.last_movement_time = time.time()
        return "timeout_nudge"

