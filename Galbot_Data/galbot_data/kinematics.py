"""Original Galbot joint-to-EEF convention, with explicit paths and validation."""

import csv
from pathlib import Path

LEFT_JOINTS = [f"left_arm/left_arm_joint{i}" for i in range(1, 8)]
RIGHT_JOINTS = [f"right_arm/right_arm_joint{i}" for i in range(1, 8)]
GRIPPERS = ["left_gripper/left_gripper_joint1", "right_gripper/right_gripper_joint1"]


def convert_eef(source, urdf, output):
    try:
        import kinpy
        from scipy.spatial.transform import Rotation
    except ImportError as exc:
        raise RuntimeError("末端位姿转换需要: pip install -r requirements-eef.txt") from exc
    data = Path(urdf).read_bytes()
    chains = [kinpy.build_serial_chain_from_urdf(
        data, f"{side}_arm_end_effector_mount_link", root_link_name=f"{side}_arm_base_link"
    ) for side in ("left", "right")]
    header = ["timestamp"]
    for side in ("left", "right"):
        header.extend(f"{side}_eef_{axis}" for axis in ("x", "y", "z", "roll", "pitch", "yaw"))
        header.append(f"{side}_gripper")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with Path(source).open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        missing = set(["timestamp", *LEFT_JOINTS, *RIGHT_JOINTS, *GRIPPERS]) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV 缺少关节列: {sorted(missing)}")
        created = False
        try:
            with output.open("x", newline="", encoding="utf-8") as target:
                created = True
                writer = csv.writer(target)
                writer.writerow(header)
                count = 0
                for row in reader:
                    result = [row["timestamp"]]
                    for chain, names, gripper in zip(chains, (LEFT_JOINTS, RIGHT_JOINTS), GRIPPERS):
                        transform = chain.forward_kinematics([float(row[name]) for name in names])
                        qw, qx, qy, qz = transform.rot
                        angles = Rotation.from_quat([qx, qy, qz, qw]).as_euler("xyz")
                        result.extend([*transform.pos, *angles, float(row[gripper])])
                    writer.writerow(result)
                    count += 1
        except Exception:
            if created:
                output.unlink(missing_ok=True)
            raise
    return count
