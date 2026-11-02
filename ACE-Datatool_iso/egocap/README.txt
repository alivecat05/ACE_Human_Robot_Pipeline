========================================
egocap v1.0.0 - Release Package
Architecture: aarch64
========================================

INSTALLATION:
-------------
1. Install system dependency (if not already installed):
   Ubuntu/Debian: sudo apt install libusb-1.0-0
   CentOS/RHEL:   sudo yum install libusb

2. Run the installation script:
   ./install.sh [install_directory]
   
   Default install directory: /opt/egocap

3. Or manually copy files:
   - Copy bin/egocap to your desired location
   - Copy bin/libegocap_dep.so to the same directory or system library path
   - Copy bin/parse_databag.py to your desired location

USAGE:
------
egocap [output_directory]

See share/egocap/README.txt for detailed usage instructions.

FILES:
------
- bin/egocap          : Main application binary
- bin/libegocap_dep.so: Internal dependency library
- bin/parse_databag.py : Python script for parsing recordings
- share/egocap/README.txt: Detailed documentation

DEPENDENCIES:
-------------
- libusb-1.0 (system library, install via package manager)
- Python 3 (for parse_databag.py, optional)

LICENSE:
--------
This is proprietary software. All rights reserved.
