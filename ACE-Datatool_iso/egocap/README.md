## 📦 Installation Guide

### Environment Requirements
Any Linux system with CMake and a C++ compiler, we have tested on Ubuntu, Debian and Raspberry Pi OS.
```bash
sudo apt install libusb-1.0-0-dev
# install if necessary
sudo apt install build-essential cmake
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

### Installation Steps

## 🔌 Hardware Connection

1. Use USB3.0 Type-C cable to connect the device
2. Set device permissions `sudo vim /etc/udev/rules.d/99-seeker.rules`, add udev rules

```plain
SUBSYSTEM=="usb", ATTR{idVendor}=="2207", ATTR{idProduct}=="0000", MODE="0666"
```

‌**Reload udev rules**‌

```bash
sudo udevadm control --reload && sudo udevadm trigger
```

**Then unplug the device**

## 🚀 Quick Start
Run the following command to start the standalone application, press Ctrl+C to exit:
```bash
cd build
./egocap [output_directory]
```

**Arguments:**
- `output_directory` (optional): Base directory for recordings. Defaults to `recordings/` if not specified. A timestamped subdirectory will be created automatically.

**Examples:**
```bash
./egocap                    # Uses default: recordings/YYYYMMDD_HHMMSS/
./egocap /path/to/recordings # Uses: /path/to/recordings/YYYYMMDD_HHMMSS/
./egocap --help             # Show usage information
```

The application will output the following files in a timestamped subdirectory:
- `mjpeg.bin`: The JPEG images with device and host timestamps.
- `imu.csv`: The IMU data with device and host timestamps.
- `metadata.json`: Device and host identification information.

**Data Format:**
- Each IMU entry contains: `device_timestamp_ns,host_timestamp_ns,gyro_x,gyro_y,gyro_z,acc_x,acc_y,acc_z`
- Each MJPEG frame contains: device timestamp, host timestamp, frame size, and JPEG data

You can use the following command to extract the JPEG images and IMU data:
```bash
python3 parse_databag.py ./build/recordings
```