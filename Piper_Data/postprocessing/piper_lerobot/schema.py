"""Native episode format; keep consistent with recording/piper_capture/schema.py."""
CAMERAS = ("head", "left_wrist", "right_wrist")
ARMS = ("left", "right")
VECTOR_NAMES = [f"{side}.{field}" for side in ARMS
                for field in ("x", "y", "z", "qx", "qy", "qz", "qw", "gripper_m")]
ACTION_SEMANTICS = "next-grid measured TCP pose + gripper width; demonstration target, not a sent command"
