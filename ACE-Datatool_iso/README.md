# ACE Robotics Camera Recording

## Usage
Python
```
python zed_record.py --record_name test001
python rs_record.py --record_name test001
./egocap/egocap ./datasets/Egocap/test001
```

or Gradio
```
python app.py
```

ZED Restart
```
sudo systemctl restart zed_x_daemon.service
```

Data convert
```
python zed_convert.py --input_directory datasets/ZED/test001
python rs_convert.py --input_directory datasets/Realsense/test001
python egocap_parse.py datasets/Egocap/test001
```

Timestamp analyze
```
python timestamp_analysis.py --zed_data_path datasets/ZED/test001 --rs_data_path datasets/Realsense/test001 
```

New !!! Data post process!
```
python egocap_parse.py datasets/Egocap/test001
python sync_color.py ./datasets --color_log_path ./datasets/color_sync_log_20260108_162507.json
python sync_timestamp_savets.py --zed_data_path datasets/ZED/test001 --rs_data_path datasets/Realsense/test001 --egocap_data_path datasets/Egocap/test001
```
裁剪片头
```
python cut.py -i F:\processed\20260122\4653804456\000002 (input) -o F:\upload\20260122\4653804456\000002 (output)
```


crontab
```
crontab -e
# add follow line
@reboot nohup bash /home/daxiao/Project/ace_record/app_crontab.sh &
```

## Reference
 - [Realsense SDK](https://github.com/realsenseai/librealsense)
 - [ZED SDK](https://github.com/stereolabs/zed-sdk)
 - [Jetson Install Guide](https://www.stereolabs.com/docs/embedded/zed-link/quad-jetson-orin-agx-devkit-setup)
