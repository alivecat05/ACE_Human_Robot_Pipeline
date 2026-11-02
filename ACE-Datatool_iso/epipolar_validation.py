import argparse
import os
import json
import cv2
import numpy as np





IMG_WIDTH_NEW = 700
IMG_HEIGHT_NEW = 700

KNEW_FX = 200
KNEW_FY = 200
KNEW_CX = IMG_WIDTH_NEW / 2
KNEW_CY = IMG_HEIGHT_NEW / 2

RECTIFY_FLAG = cv2.omnidir.RECTIFY_PERSPECTIVE
VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def load_json(filepath):
    """加载 JSON 文件"""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

def rotation_matrix_to_axis_angle(R):
    """
    将 3x3 旋转矩阵转换为轴角向量 (Rodrigues 向量)
    使用 numpy 实现，避免依赖 cv2
    """
    # 确保是 float 类型
    R = R.astype(np.float64)

    # 计算迹
    trace = np.trace(R)

    # 处理数值误差，确保 acos 输入在 [-1, 1] 之间
    cos_theta = (trace - 1.0) / 2.0
    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    theta = np.arccos(cos_theta)

    # 如果角度很小，返回零向量，避免除以零
    if theta < 1e-6:
        return np.zeros(3)

    # 计算轴角向量 r = theta * u
    # u_x = (R32 - R23) / (2 * sin(theta))
    # u_y = (R13 - R31) / (2 * sin(theta))
    # u_z = (R21 - R12) / (2 * sin(theta))
    sin_theta = np.sin(theta)

    rx = (R[2, 1] - R[1, 2]) * theta / (2 * sin_theta)
    ry = (R[0, 2] - R[2, 0]) * theta / (2 * sin_theta)
    rz = (R[1, 0] - R[0, 1]) * theta / (2 * sin_theta)

    return np.array([rx, ry, rz])


def process_head_calibration(json_data):
    """
    处理 head 部分的相机标定数据
    """
    # 1. 定位到 head 配置
    try:
        # head_config = json_data['device']['ego']['head']
        # cam_calibration = head_config['cam_calibration']
        cam_calibration = json_data
    except KeyError as e:
        print(f"错误：JSON 结构中未找到指定路径：{e}")
        return None, None

    # 2. 准备相机列表 (根据提供的 JSON，key 为 cam_0, cam_1...), 目前就使用cam_0和cam_1
    cam_ids = ["cam0", "cam1"]

    # 验证所有相机是否存在
    for cid in cam_ids:
        if cid not in cam_calibration:
            print(f"警告：未找到 {cid} 的配置")
            return None, None

    # ---------------------------------------------------------
    # 生成 InitCams.txt 数据
    # 格式：相机序号 (1-4) fx fy cx cy xi k1 k2 p1 p2
    # ---------------------------------------------------------
    init_cams_lines = []

    for idx, cid in enumerate(cam_ids):
        cam_data = cam_calibration[cid]
        cam_idx = idx + 1  # 1-based index

        # 提取内参
        # K: [fx, fy, cx, cy]
        fx, fy, cx, cy = cam_data["K"]
        # xi: [value]
        xi = cam_data["xi"][0]
        # D: [k1, k2, p1, p2] (假设 D 的顺序对应请求的输出顺序)
        k1, k2, p1, p2 = cam_data["D"]

        line = f"{cam_idx} {fx:.10f} {fy:.10f} {cx:.10f} {cy:.10f} {xi:.10f} {k1:.10f} {k2:.10f} {p1:.10f} {p2:.10f}"
        init_cams_lines.append(line)

    # ---------------------------------------------------------
    # 生成 generalizedParams.txt 数据
    # 逻辑：计算每个相机在 cam0 坐标系下的位姿 (T_c0_cn)
    # 输出：rx ry rz tx ty tz
    # rx,ry,rz = (R_c0_cn.T) 的轴角形式
    # tx,ty,tz = T_c0_cn 的平移分量
    # ---------------------------------------------------------

    # 步骤 1: 计算从 CamN 到 Cam0 的变换链 (T_cn_c0)
    # 参考脚本逻辑：T_c1_c0 直接取自 cam1; T_c2_c0 = T_c2_c1 @ T_c1_c0 ...
    # 注意：提供的 JSON 中 cam_0 是 T_c_b，但作为相对外参基准，我们视 cam_0 为原点 (I)
    # cam_1, cam_2, cam_3 使用 T_cn_cnm1

    T_c0_c0 = np.eye(4)

    # 获取变换矩阵
    # cam_1: T_c1_c0 (cam1 -> cam0)
    T_c1_c0 = np.array(cam_calibration["cam1"]["T_cn_cnm1"])
    # cam_2: T_c2_c1 (cam2 -> cam1)
    T_c2_c1 = np.array(cam_calibration["cam2"]["T_cn_cnm1"])
    # cam_3: T_c3_c2 (cam3 -> cam2)
    T_c3_c2 = np.array(cam_calibration["cam3"]["T_cn_cnm1"])

    # 累积变换 (Target <- Source 链式乘法)
    # T_c1_c0 已经是相对于 cam0
    T_c2_c0 = T_c2_c1 @ T_c1_c0
    T_c3_c0 = T_c3_c2 @ T_c2_c0

    transforms_cn_to_c0 = [T_c0_c0, T_c1_c0, T_c2_c0, T_c3_c0]

    generalized_params_lines = []

    for idx, T_cn_c0 in enumerate(transforms_cn_to_c0):
        # 我们需要的是 T_c0_cn (Pose of camN in cam0 frame)
        # 即 T_cn_c0 的逆
        T_c0_cn = np.linalg.inv(T_cn_c0)

        # 提取旋转和平移
        R_c0_cn = T_c0_cn[:3, :3]
        t_c0_cn = T_c0_cn[:3, 3]

        # 要求：旋转分量取转置 (R^T)
        # R_transposed = R_c0_cn.T

        # 要求：转换成轴角形式
        axis_angle = rotation_matrix_to_axis_angle(R_c0_cn)

        # 格式化输出：rx ry rz tx ty tz
        line = f"{axis_angle[0]:.10f} {axis_angle[1]:.10f} {axis_angle[2]:.10f} {t_c0_cn[0]:.10f} {t_c0_cn[1]:.10f} {t_c0_cn[2]:.10f}"
        generalized_params_lines.append(line)

    return init_cams_lines, generalized_params_lines


def _pick_first_image(rgb_dir):
    files = [
        name
        for name in os.listdir(rgb_dir)
        if os.path.isfile(os.path.join(rgb_dir, name))
        and os.path.splitext(name)[1].lower() in VALID_EXTS
    ]
    files.sort()
    if not files:
        raise RuntimeError(f"`{rgb_dir}` 下没有可用图像")
    return files[0]


def _resolve_rgb_input(rgb_input):
    if os.path.isdir(rgb_input):
        first_name = _pick_first_image(rgb_input)
        rgb_path = os.path.join(rgb_input, first_name)
        return rgb_input, first_name, rgb_path

    if os.path.isfile(rgb_input):
        ext = os.path.splitext(rgb_input)[1].lower()
        if ext not in VALID_EXTS:
            raise ValueError(f"不支持的图像格式: {rgb_input}")
        rgb_dir = os.path.dirname(rgb_input)
        first_name = os.path.basename(rgb_input)
        return rgb_dir, first_name, rgb_input

    raise FileNotFoundError(f"找不到 `--rgb` 指定路径: {rgb_input}")


def _parse_intrinsics_from_init_lines(init_lines):
    cams = {}
    for line in init_lines:
        vals = list(map(float, line.split()))
        if len(vals) < 10:
            raise ValueError(f"InitCams 行格式错误: {line}")
        cam_id = int(vals[0])
        fx, fy, cx, cy = vals[1:5]
        xi = vals[5]
        k1, k2, p1, p2 = vals[6:10]
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        D = np.array([k1, k2, p1, p2], dtype=np.float64)
        cams[cam_id] = (K, xi, D)

    if 1 not in cams or 2 not in cams:
        raise ValueError("配置中缺少 cam_0/cam_1 内参")

    K1, xi1, D1 = cams[2]
    K2, xi2, D2 = cams[1]
    return K1, xi1, D1, K2, xi2, D2


def _parse_extrinsics_from_generalized_lines(generalized_lines):
    if len(generalized_lines) < 2:
        raise ValueError("外参行数不足，无法解析 cam_1 相对 cam_0 的位姿")
    vals = list(map(float, generalized_lines[1].split()))
    if len(vals) != 6:
        raise ValueError("外参格式错误，必须为 rx ry rz tx ty tz")
    r = np.array(vals[:3], dtype=np.float64)
    t = np.array(vals[3:], dtype=np.float64).reshape(3, 1)
    R, _ = cv2.Rodrigues(r)
    return R, t


def _rectify_pair(frame_cam0, frame_cam1, config_json):
    calib_data = load_json(config_json)
    init_lines, gen_lines = process_head_calibration(calib_data)
    if init_lines is None or gen_lines is None:
        raise RuntimeError("解析 config.json 失败")

    K1, xi1, D1, K2, xi2, D2 = _parse_intrinsics_from_init_lines(init_lines)
    R, T = _parse_extrinsics_from_generalized_lines(gen_lines)

    R1, R2 = cv2.omnidir.stereoRectify(R, T, RECTIFY_FLAG)
    Knew = np.array(
        [[KNEW_FX, 0, KNEW_CX], [0, KNEW_FY, KNEW_CY], [0, 0, 1]],
        dtype=np.float64,
    )
    size = (IMG_WIDTH_NEW, IMG_HEIGHT_NEW)

    map1L, map2L = cv2.omnidir.initUndistortRectifyMap(
        K1,
        D1,
        np.array([xi1], dtype=np.float64),
        R1,
        Knew,
        size,
        cv2.CV_32FC1,
        RECTIFY_FLAG,
    )
    map1R, map2R = cv2.omnidir.initUndistortRectifyMap(
        K2,
        D2,
        np.array([xi2], dtype=np.float64),
        R2,
        Knew,
        size,
        cv2.CV_32FC1,
        RECTIFY_FLAG,
    )

    rec_left = cv2.remap(frame_cam1, map1L, map2L, cv2.INTER_LINEAR)
    rec_right = cv2.remap(frame_cam0, map1R, map2R, cv2.INTER_LINEAR)
    return rec_left, rec_right


def _draw_epipolar(left_img, right_img):
    if left_img.shape[0] != right_img.shape[0]:
        target_size = (right_img.shape[1], left_img.shape[0])
        right_img = cv2.resize(right_img, target_size)

    stereo = cv2.hconcat([left_img, right_img])
    height, width = stereo.shape[:2]
    mid_x = left_img.shape[1]

    for y in range(0, height, 50):
        cv2.line(stereo, (0, y), (width, y), (0, 255, 0), 1)

    cv2.line(stereo, (mid_x, 0), (mid_x, height), (0, 0, 255), 2)
    cv2.putText(
        stereo,
        "LEFT",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        stereo,
        "RIGHT",
        (mid_x + 20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    return stereo


def run(config_json, rgb_input):
    if not os.path.isfile(config_json):
        raise FileNotFoundError(f"找不到配置文件: {config_json}")

    rgb_dir, first_name, rgb_path = _resolve_rgb_input(rgb_input)
    frame_rgb = cv2.imread(rgb_path)
    if frame_rgb is None:
        raise RuntimeError(f"读取图像失败: {rgb_path}")

    frame_height = frame_rgb.shape[0]
    frame_cam0 = frame_rgb[0 : frame_height // 4, :]
    frame_cam1 = frame_rgb[frame_height // 4 : 2 * frame_height // 4, :]

    rec_left, rec_right = _rectify_pair(frame_cam0, frame_cam1, config_json)
    result = _draw_epipolar(rec_left, rec_right)

    recording_dir = os.path.dirname(rgb_dir)
    output_dir = os.path.join(recording_dir, "rectify_new")
    left_dir = os.path.join(output_dir, "left")
    right_dir = os.path.join(output_dir, "right")
    os.makedirs(left_dir, exist_ok=True)
    os.makedirs(right_dir, exist_ok=True)

    left_path = os.path.join(left_dir, first_name)
    right_path = os.path.join(right_dir, first_name)
    output_path = os.path.join(output_dir, "output.png")

    cv2.imwrite(left_path, rec_left)
    cv2.imwrite(right_path, rec_right)
    cv2.imwrite(output_path, result)

    print(f"输入图像: {rgb_path}")
    print(f"左图输出: {left_path}")
    print(f"右图输出: {right_path}")
    print(f"极线输出: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="单帧鱼眼双目极线校正与可视化")
    parser.add_argument("--config_json", required=True, type=str, help="标定配置 JSON")
    parser.add_argument("--rgb", required=True, type=str, help="rgb 目录或单张图像路径")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(config_json=args.config_json, rgb_input=args.rgb)
