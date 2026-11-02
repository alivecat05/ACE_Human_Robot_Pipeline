import argparse

import pyzed.sl as sl

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
    main()