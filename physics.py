import numpy as np

class PhysicsEngine:
    """
    Simple 3D physics engine for AR objects.
    Handles smooth translation, rotation, and scaling with damping and max speed.
    """

    def __init__(self,
                 translation_damping=0.2,
                 rotation_damping=0.1,
                 scale_damping=0.05,
                 max_translation_speed=0.1,
                 max_rotation_speed=5.0,
                 max_scale_speed=0.05):
        self.translation_damping = translation_damping
        self.rotation_damping = rotation_damping
        self.scale_damping = scale_damping
        self.max_translation_speed = max_translation_speed
        self.max_rotation_speed = max_rotation_speed
        self.max_scale_speed = max_scale_speed

        # Internal velocity tracking per object
        self.velocities = {}

    # --------------------------
    # Apply physics to an object
    # --------------------------
    def apply(self, obj, delta):
        obj_id = id(obj)
        if obj_id not in self.velocities:
            self.velocities[obj_id] = {
                "translate": np.zeros(3, dtype=np.float32),
                "rotate": np.zeros(3, dtype=np.float32),
                "scale": np.zeros(3, dtype=np.float32)
            }

        # --------------------------
        # Translation
        # --------------------------
        vel_t = self.velocities[obj_id]["translate"]
        target_t = np.array(delta.get("translate", np.zeros(3)), dtype=np.float32).flatten()
        vel_t = vel_t * (1 - self.translation_damping) + target_t
        speed_t = np.linalg.norm(vel_t)
        if speed_t > self.max_translation_speed:
            vel_t = (vel_t / speed_t) * self.max_translation_speed
        obj.translate(vel_t)
        self.velocities[obj_id]["translate"] = vel_t

        # --------------------------
        # Rotation
        # --------------------------
        vel_r = self.velocities[obj_id]["rotate"]
        target_r = np.array(delta.get("rotate", np.zeros(3)), dtype=np.float32).flatten()
        vel_r = vel_r * (1 - self.rotation_damping) + target_r
        vel_r = np.clip(vel_r, -self.max_rotation_speed, self.max_rotation_speed)
        obj.rotate(vel_r)
        self.velocities[obj_id]["rotate"] = vel_r

        # --------------------------
        # Scaling
        # --------------------------
        vel_s = self.velocities[obj_id]["scale"]
        target_s = np.array(delta.get("scale", np.ones(3)), dtype=np.float32).flatten() - 1.0  # relative delta
        vel_s = vel_s * (1 - self.scale_damping) + target_s

        # Clamp scale change per frame
        vel_s = np.clip(vel_s, -self.max_scale_speed, self.max_scale_speed)

        # Apply scale
        obj.scale_object(1.0 + vel_s)  # safe 1D multiplication
        self.velocities[obj_id]["scale"] = vel_s
