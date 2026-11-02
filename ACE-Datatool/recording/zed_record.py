########################################################################
#
# Copyright (c) 2022, STEREOLABS.
#
# All rights reserved.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
########################################################################

"""
This sample shows how to record video in Stereolabs SVO format.
SVO video files can be played with the ZED API and used with its different modules.
"""

import pyzed.sl as sl
import threading
import signal
import time
import sys
import os
from datetime import datetime
import argparse
import csv

def parse_args():
    parser = argparse.ArgumentParser(description="ZED Camera SVO Recorder")
    parser.add_argument(
        "--record_name",
        type=str,
        default="kkb",
        help="Recording name. If not set, error"
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="raw",
        help="Directory to save SVO files."
    )
    return parser.parse_args()


# Global variable to handle exit
exit_app = False


def signal_handler(signal, frame):
    """Handle Ctrl+C to properly exit the program"""
    global exit_app
    exit_app = True
    print("\nCtrl+C pressed. Exiting...")


def acquisition(zed, imu_csv_path=None):
    """Acquisition thread function to continuously grab frames and IMU data"""
    if isinstance(zed, sl.Camera):
        infos = zed.get_camera_information()
        print(f"serial number: {infos.serial_number} start recording at {datetime.now()}")
        sn = infos.serial_number
    else:
        infos = zed.get_camera_information()
        print(f"CameraOne SN{infos.serial_number} recording start at {datetime.now()}")
        sn = infos.serial_number

    # Initialize IMU CSV file if path is provided
    csv_file = None
    csv_writer = None
    imu_count = 0
    if imu_csv_path:
        try:
            csv_file = open(imu_csv_path, 'w', newline='', buffering=1)  # Line buffering
            csv_writer = csv.writer(csv_file)
            # Write CSV header
            csv_writer.writerow([
                'timestamp', 'accel_x', 'accel_y', 'accel_z',
                'gyro_x', 'gyro_y', 'gyro_z'
            ])
            csv_file.flush()
            print(f"IMU CSV file created: {imu_csv_path}")
        except Exception as e:
            print(f"Failed to create IMU CSV file: {e}")
            csv_file = None

    sensors_data = sl.SensorsData()

    while not exit_app:
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            # Record IMU data if available
            if csv_writer:
                if zed.get_sensors_data(sensors_data, sl.TIME_REFERENCE.CURRENT) == sl.ERROR_CODE.SUCCESS:
                    imu_data = sensors_data.get_imu_data()
                    # Check if IMU data is valid
                    if imu_data.is_available:
                        csv_writer.writerow([
                            imu_data.timestamp.get_nanoseconds(),
                            imu_data.get_linear_acceleration()[0],
                            imu_data.get_linear_acceleration()[1],
                            imu_data.get_linear_acceleration()[2],
                            imu_data.get_angular_velocity()[0],
                            imu_data.get_angular_velocity()[1],
                            imu_data.get_angular_velocity()[2]
                        ])
                        imu_count += 1
                        # Flush every 100 samples
                        if imu_count % 100 == 0:
                            csv_file.flush()
                    elif imu_count == 0:
                        # Only print once at the start
                        print(f"[SN{sn}] Waiting for IMU data...")
        else:
            print(f"grab failed")

    # Close CSV file
    if csv_file:
        csv_file.flush()
        csv_file.close()
        print(f"IMU data saved: {imu_csv_path} ({imu_count} samples)")
    elif imu_csv_path:
        print(f"No IMU data recorded for SN{sn}")

    if isinstance(zed, sl.Camera):
        print(f"{infos.camera_model}[{infos.serial_number}] QUIT")
    else:
        print(f"CameraOne[{infos.serial_number}] QUIT")

    # disable Recording
    zed.disable_recording()
    # close the Camera
    zed.close()


def open_camera(zed, sn, save_path, camera_fps=30.0):
    """Open a camera with given serial number and enable streaming with given port"""
    if isinstance(zed, sl.Camera):
        init_params = sl.InitParameters()
        init_params.depth_mode = sl.DEPTH_MODE.NONE  # No depth mode for this example
        init_params.set_from_serial_number(sn)
        init_params.camera_resolution = sl.RESOLUTION.HD1080
        init_params.camera_fps = camera_fps
        # IMU is only available on stereo cameras (ZED, ZED2, ZED X, etc.)
        init_params.sensors_required = True  # Enable sensors for IMU data
    elif isinstance(zed, sl.CameraOne):
        init_params = sl.InitParametersOne()
        # init_params = sl.CameraOne.InitParametersOne()
        init_params.camera_resolution = sl.RESOLUTION.HD1080
        init_params.set_from_serial_number(sn)
        init_params.camera_fps = camera_fps
    else:
        print(f"Unsupported camera type: {type(zed)}")
        return False

    # Open the camera
    open_err = zed.open(init_params)
    if open_err == sl.ERROR_CODE.SUCCESS:
        if isinstance(zed, sl.Camera):
            info = zed.get_camera_information()
            print(f"{info.camera_model}_SN{sn} Opened")
        else:
            print(f"CameraOne_SN{sn} Opened")
    else:
        print(f"ZED SN{sn} Error: {open_err}")
        zed.close()
        return False

    # zed.set_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE, 30)
    # zed.set_camera_settings(sl.VIDEO_SETTINGS.GAIN, 16)
    zed.set_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE, 20)
    zed.set_camera_settings(sl.VIDEO_SETTINGS.GAIN, 8)

    output_svo_file = f"{save_path}/{sn}.svo2"
    recording_param = sl.RecordingParameters(output_svo_file.replace(" ", ""), sl.SVO_COMPRESSION_MODE.H265)
    # h264 will lose frame
    # Enable sensors data recording for stereo cameras
    if isinstance(zed, sl.Camera):
        recording_param.transcode_streaming_input = False
    record_err = zed.enable_recording(recording_param)
    if record_err == sl.ERROR_CODE.SUCCESS:
        print(f"SN{sn} Enabled recording")
    else:
        print(f"ZED SN{sn} Recording initialization error: {record_err}")
        zed.close()
        return False

    print(f"Recording SVO file: {recording_param.video_filename}")

    return True


def print_device_info(devs):
    """Print information about detected devices"""
    for dev in devs:
        print(f"ID: {dev.id}, model: {dev.camera_model}, S/N: {dev.serial_number}, state: {dev.camera_state}")


def main():
    global exit_app

    # Get the list of available ZED cameras
    dev_stereo_list = sl.Camera.get_device_list()
    print("Stereo cameras:")
    print_device_info(dev_stereo_list)

    if sys.platform != "win32":
        dev_mono_list = sl.CameraOne.get_device_list()
        print("Mono cameras:")
        print_device_info(dev_mono_list)
    else:
        dev_mono_list = []

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

    zed_open = False

    # Create camera objects based on type
    zeds = []
    for camera_type, dev in all_devices:
        if camera_type == "stereo":
            zeds.append(sl.Camera())
        else:  # mono
            zeds.append(sl.CameraOne())
    
    args = parse_args()
    if args.record_name:
        if os.path.exists(os.path.join(args.save_path, args.record_name, "zed")) :
            print("Record already exists!!")
            return 1
        save_path = os.path.join(args.save_path, args.record_name, "zed")
    else:
        print("Error, ZED no record name")
        return 1

    os.makedirs(save_path, exist_ok=True)
    print(f"Saving SVO files to: {save_path}")

    # Open all cameras
    for i, (camera_type, dev) in enumerate(all_devices):
        zed_open = open_camera(zeds[i], dev.serial_number, save_path)
        print(f"Camera {i} ({camera_type}) opened: {zed_open}")

    if not zed_open:
        print("No ZED opened, exit program")
        return 1

    # Create a grab thread for each opened camera
    threads = []
    for z in range(nb_cameras):
        if zeds[z].is_opened():
            # Create IMU CSV path for all cameras (both stereo and mono)
            sn = zeds[z].get_camera_information().serial_number
            imu_csv_path = f"{save_path}/{sn}_imu.csv"
            threads.append(threading.Thread(target=acquisition, args=(zeds[z], imu_csv_path)))
            threads[-1].start()

    # Set up signal handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)
    print("Press Ctrl+C to exit")

    # Main loop - wait for exit signal
    while not exit_app:
        time.sleep(0.1)

    # Wait for all threads to finish
    print("Exit signal received, closing ZEDs...")
    time.sleep(0.1)

    for thread in threads:
        thread.join()

    print("Program exited")
    return 0


if __name__ == "__main__":
    sys.exit(main())
