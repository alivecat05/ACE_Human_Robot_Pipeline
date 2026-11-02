import pyrealsense2 as rs
import numpy as np
import argparse

import pyzed.sl as sl
def get_all_realsense_intrinsics():
    ctx = rs.context()
    devices = ctx.query_devices()
    
    if len(devices) == 0:
        print("未检测到任何 RealSense 相机")
        return []

    # 先收集序列号，避免在遍历设备对象时反复 start/stop 导致底层句柄失效。
    serial_numbers = []
    for i in range(2):
        try:
            dev = devices[i]
            if not dev.supports(rs.camera_info.serial_number):
                continue
            serial_numbers.append(dev.get_info(rs.camera_info.serial_number))
        except Exception as e:
            print(f"读取设备序列号失败(index={i}): {e}")

    all_params = []

    for sn in serial_numbers:
        print(f"正在读取设备: {sn}")

        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(sn)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        started = False
        try:
            profile = pipeline.start(config)
            started = True

            color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
            depth_stream = profile.get_stream(rs.stream.depth).as_video_stream_profile()

            c_int = color_stream.get_intrinsics()
            d_int = depth_stream.get_intrinsics()

            # depth -> color 外参
            extr = depth_stream.get_extrinsics_to(color_stream)
            R = np.array(extr.rotation).reshape(3, 3)
            t = np.array(extr.translation).reshape(3, 1)
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3:] = t

            def intr_to_mat(i):
                return np.array([
                    [i.fx, 0,    i.ppx],
                    [0,    i.fy, i.ppy],
                    [0,    0,    1    ]
                ])

            all_params.append({
                'serial_number': sn,
                'K_color': intr_to_mat(c_int),
                'K_depth': intr_to_mat(d_int),
                'T_depth_to_color': T
            })

        except Exception as e:
            print(f"无法读取相机 {sn}: {e}")
        finally:
            if started:
                try:
                    pipeline.stop()
                except Exception:
                    pass

    return all_params





def print_device_info(devs):
    """Print information about detected devices"""
    for dev in devs:
        print(f"ID: {dev.id}, model: {dev.camera_model}, S/N: {dev.serial_number}, state: {dev.camera_state}")


def pick_serial(requested_sn=None):
    # Get the list of available ZED cameras
    dev_stereo_list = sl.Camera.get_device_list()
    print("Stereo cameras:")
    print_device_info(dev_stereo_list)

    dev_mono_list = sl.CameraOne.get_device_list()
    print("Mono cameras:")
    print_device_info(dev_mono_list)

    # Create combined device list with type information
    all_devices = []
    for dev in dev_stereo_list:
        all_devices.append(("stereo", dev))
    for dev in dev_mono_list:
        all_devices.append(("mono", dev))

    nb_cameras = len(all_devices)
    if nb_cameras == 0:
        print("No ZED Detected, exit program")
        return 1

    all_sn = [int(dev.serial_number) for _, dev in all_devices]
    if requested_sn not in all_sn:
        print(f"Requested S/N {requested_sn} not found in detected devices")
        return None
    
    return requested_sn


def main():
    parser = argparse.ArgumentParser(description="Read ZED calibration parameters")
    parser.add_argument("--sn", type=int, default=308678632, help="Camera serial number")
    args = parser.parse_args()

    sn = pick_serial(args.sn)
    if sn is None:
        return

    zed = sl.CameraOne()

    init_params = sl.InitParametersOne()
    init_params.set_from_serial_number(sn)
    init_params.camera_resolution = sl.RESOLUTION.HD1080


    status = zed.open(init_params)
    if status == sl.ERROR_CODE.SUCCESS:
        cam_info = zed.get_camera_information()
        calib = cam_info.camera_configuration.calibration_parameters

        print(f"Opened SN={cam_info.serial_number}, model={cam_info.camera_model}")
        print("fx, fy, cx, cy, k1")
        print(
            f"{calib.fx}, {calib.fy}, "
            f"{calib.cx}, {calib.cy}, "
            f"{calib.disto[1]}"
        )
    else:
        print(f"Failed to open ZED camera SN={sn}: {status}")

    zed.close()


if __name__ == "__main__":
    get_all_realsense_intrinsics()