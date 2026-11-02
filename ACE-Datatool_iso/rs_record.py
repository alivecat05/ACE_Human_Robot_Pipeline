#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3 台 RealSense 全部保存 RGB + Depth + IMU（acc/gyro 分文件）
"""
import pyrealsense2 as rs
import numpy as np
import imageio
import time
import os
import sys
import threading
import signal
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="3 cam all RGBD+IMU")
    parser.add_argument("--record_name", type=str, default="",
                        help="Recording name. If not set, uses datasets/<timestamp>")
    parser.add_argument("--save_path", type=str, default="",
                        help="Directory to save result files.")
    return parser.parse_args()


stop_flag = threading.Event()


def signal_handler(sig, frame):
    global stop_flag
    stop_flag.set()
    print("\nCtrl+C pressed. Exiting...")


def lock_imu(dev):
    """返回 (acc_profile, gyro_profile) 或 (None,None)"""
    sn = dev.get_info(rs.camera_info.serial_number)
    motion = None
    for s in dev.query_sensors():
        if "motion" in s.get_info(rs.camera_info.name).lower():
            motion = s
            break
    if motion is None:
        return None, None

    avail_a = {p for p in motion.get_stream_profiles()
               if p.stream_type() == rs.stream.accel}
    avail_g = {p for p in motion.get_stream_profiles()
               if p.stream_type() == rs.stream.gyro}

    # 934222070042 强制 63/200
    if sn == "934222070042":
        wanted_a = next((p for p in avail_a if p.fps() == 63), None)
        wanted_g = next((p for p in avail_g if p.fps() == 200), None)
        return wanted_a, wanted_g

    # 其余 D435i 优先 200/200，再退 63/200
    for a_fps, g_fps in [(200, 200), (63, 200)]:
        wanted_a = next((p for p in avail_a if p.fps() == a_fps), None)
        wanted_g = next((p for p in avail_g if p.fps() == g_fps), None)
        if wanted_a and wanted_g:
            return wanted_a, wanted_g
    return None, None


def main():
    ctx = rs.context()
    devices = ctx.query_devices()[:2]
    camera_sn = []
    camera_model = []
    model_map = {"D405": "D405", "D435I": "D435i"}

    print("=== Detected Cameras ===")
    for dev in devices:
        if not dev.supports(rs.camera_info.serial_number):
            continue
        sn = dev.get_info(rs.camera_info.serial_number)
        name = dev.get_info(rs.camera_info.name).upper()
        model = next((v for k, v in model_map.items() if k in name), "D435i")
        print("Camera:", sn, model)
        camera_sn.append(sn)
        camera_model.append(model)

    # 启动 pipeline
    pipes = []
    for idx, sn in enumerate(camera_sn):
        pipe = rs.pipeline(ctx)
        cfg = rs.config()
        cfg.enable_device(sn)
        model = camera_model[idx]

        # 统一用 848×480@15 减轻带宽压力
        cfg.enable_stream(rs.stream.color, 848, 480, rs.format.rgb8, 15)
        cfg.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 15)

        # IMU：能开就开，不能开就跳过
        if model == "D435i":
            pa, pg = lock_imu(devices[idx])
            if pa and pg:
                cfg.enable_stream(pa.stream_type(), pa.format(), pa.fps())
                cfg.enable_stream(pg.stream_type(), pg.format(), pg.fps())
                print(f"  {sn} IMU locked {pa.fps()}/{pg.fps()} Hz")
            else:
                print(f"  {sn} 无可用 IMU，跳过")

        pipe.start(cfg)
        pipes.append(pipe)

 
        # 曝光控制
        # device = pipe.get_active_profile().get_device()
        # model = device.get_info(rs.camera_info.name)

        # # D435i 特殊处理：只操作第 2 个 sensor（彩色）
        # if model == "D435i":
        #     sensor = device.query_sensors()[1]          # 0=深度, 1=彩色
        #     sensor.set_option(rs.option.enable_auto_exposure, 0)
        #     sensor.set_option(rs.option.auto_exposure_priority, True)  # 允许在带宽不足时自动降曝光
        #     sensor.set_option(rs.option.exposure, 30.0)                # 30 ms = 30000 µs
        #     sensor.set_option(rs.option.enable_auto_white_balance, 1)
        # else:
        #     # 其它型号保持你原来的逻辑
        #     for sensor in device.query_sensors():
        #         if sensor.is_color_sensor():
        #             sensor.set_option(rs.option.enable_auto_exposure, 0)
        #             sensor.set_option(rs.option.enable_auto_white_balance, 1)
        #             if model == "D405":
        #                 sensor.set_option(rs.option.exposure, 6000)
        #                 sensor.set_option(rs.option.gain, 40)
        #             elif model == "D435i":
        #                 # 这里不会再走到，上面已经提前拦截
        #                 pass
        


        for sensor in pipe.get_active_profile().get_device().query_sensors():
            if sensor.is_color_sensor():
                sensor.set_option(rs.option.enable_auto_exposure, 1)
                sensor.set_option(rs.option.enable_auto_white_balance, 1)
                # if model == "D405":
                #     sensor.set_option(rs.option.exposure, 6000)
                #     sensor.set_option(rs.option.gain, 40)
                # elif model == "D435i":
                #     sensor.set_option(rs.option.exposure, 900)   # ← 从 1800 拉到 6000
                #     sensor.set_option(rs.option.gain, 16)         # ← 从 20 降到 16
                #     pass
        print(f"Started camera {sn} ({model})", flush=True)
    # 保存路径
    args = parse_args()
    if args.record_name:
        if os.path.exists(os.path.join(args.save_path, args.record_name, "realsense")) :
            print("Record already exists!!")
            return 1
        save_path = os.path.join(args.save_path, args.record_name, "realsense")
    else:
        print("Error, realsense no record name")
        return 1

    os.makedirs(save_path, exist_ok=True)
    print(f"Saving mp4 files to: {save_path}", flush=True)


    # 初始化 writer
    writer_cams = []
    writer_times = []
    writer_imu = []
    colorizer = rs.colorizer()
    for i in range(len(camera_sn)):
        writers = {}
        writers["rgb"] = imageio.get_writer(
            os.path.join(save_path, f"{camera_sn[i]}_rgb.mp4"),
            fps=15, mode='I', codec='libx264', format='FFMPEG', macro_block_size=1)
        writers["depth"] = imageio.get_writer(
            os.path.join(save_path, f"{camera_sn[i]}_depth.mp4"),
            fps=15, mode='I', codec='libx264', format='FFMPEG', macro_block_size=1)
        writer_cams.append(writers)
        ts_file = open(os.path.join(save_path, f"{camera_sn[i]}_ts.csv"), 'w')
        writer_times.append(ts_file)

        # IMU 文件（无 IMU 时句柄为 None）
        if camera_model[i] == "D435i":
            writer_imu.append(open(os.path.join(save_path, f"{camera_sn[i]}_imu.csv"), 'w'))
        else:
            writer_imu.append(None)

    # 稳帧
    for p in pipes:
        p.wait_for_frames()
    time.sleep(0.1)

    # 录制线程
    def grab_cam(i, barrier):
        barrier.wait()
        while not stop_flag.is_set():
            frames = pipes[i].wait_for_frames()
            color = frames.get_color_frame()
            depth = frames.get_depth_frame()
            if not color or not depth:
                continue

            img = np.asanyarray(color.get_data())
            depthmap = np.asanyarray(colorizer.colorize(depth).get_data())
            ts_us = frames.get_frame_metadata(rs.frame_metadata_value.backend_timestamp)
            
            writer_cams[i]["rgb"].append_data(img)
            writer_cams[i]["depth"].append_data(depthmap)
            writer_times[i].write(f"{ts_us}\n")

            # IMU：按实际采样时刻分别写 acc / gyro
            if camera_model[i] == "D435i":
                accel = frames.first_or_default(rs.stream.accel, rs.format.motion_xyz32f)
                gyro  = frames.first_or_default(rs.stream.gyro,  rs.format.motion_xyz32f)
                if accel and gyro:
                    a = accel.as_motion_frame().get_motion_data()
                    g = gyro.as_motion_frame().get_motion_data()
                    writer_imu[i].write(f"{ts_us},{g.x},{g.y},{g.z},{a.x},{a.y},{a.z}\n")


        # 清理
        writer_cams[i]["rgb"].close()
        writer_cams[i]["depth"].close()
        writer_times[i].close()
        if writer_imu[i]:
            writer_imu[i].close()

    # 启动
    barrier = threading.Barrier(len(pipes))
    threads = [threading.Thread(target=grab_cam, args=(i, barrier), daemon=True)
               for i in range(len(pipes))]
    signal.signal(signal.SIGINT, signal_handler)

    print("\n=== Recording RGBD+IMU... Press Ctrl+C to stop ===")
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    for p in pipes:
        p.stop()
    print('>>> All camera pipelines stopped.')
    return 0


if __name__ == '__main__':
    sys.exit(main())