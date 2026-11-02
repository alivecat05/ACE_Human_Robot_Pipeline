import os
import cv2

from dt_apriltags import Detector
import numpy
import os
import json
import time
import numpy as np

BLOCK_SIZE = 1.1
MID_DIST = 1.1 * 4.5
INNER_DIST = 1.1 * 2.5
APRIL_GLOBAL_POSE = {
    0: [-55.05, 4.95, 0],
    1: [-55.05, 35.05, 0],
    2: [55.05, 35.05, 0],
    3: [55.05, 4.95, 0],
}

at_detector = Detector(families='tagStandard41h12',
                       nthreads=1,
                       quad_decimate=1.0,
                       quad_sigma=0.5,
                       refine_edges=1,
                       decode_sharpening=0.25,
                       debug=0)


camera_matrix = [
    609.282, 0, 640.0,
    0, 609.282, 345.0,
    0, 0, 1 
]
parameters = {
    "K": camera_matrix, 
    "tag_size": 0.055,
    "height": 720,
    "width": 1280,
}

calibration_data = {}
calibration_data["camera_matrix"] = np.array(camera_matrix).reshape(3, 3).tolist()


def visualize_det(img, tags) :

    color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    for tag in tags:
        for idx in range(len(tag.corners)):
            cv2.line(color_img, tuple(tag.corners[idx-1, :].astype(int)), tuple(tag.corners[idx, :].astype(int)), (0, 255, 0))

        cv2.putText(color_img, str(tag.tag_id),
                    org=(tag.corners[0, 0].astype(int)+10,tag.corners[0, 1].astype(int)+10),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.8,
                    color=(0, 0, 255))
    return color_img


####################### OPPOSITE CALIBRATE ################################
cap_kinect2 = cv2.VideoCapture(2)
cap_kinect2.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap_kinect2.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
ret2, image_oppo = cap_kinect2.read()

img = cv2.cvtColor(image_oppo, cv2.COLOR_BGR2GRAY)
cameraMatrix = numpy.array(parameters['K']).reshape((3,3))
camera_params = ( cameraMatrix[0,0], cameraMatrix[1,1], cameraMatrix[0,2], cameraMatrix[1,2] )
tags = at_detector.detect(img, True, camera_params, parameters['tag_size'])

color_img = visualize_det(img, tags)
cv2.imwrite("april_tag_oppo.png", color_img)


world_coords = []
image_coords = []
for tag in tags:
    tid = tag.tag_id
    world_tag_center = APRIL_GLOBAL_POSE[tid]
    image_tag_center = tag.center

    corners = {}
    for point in tag.corners:
        x, y = point
        if x < image_tag_center[0] and y < image_tag_center[1]:
            corners["top-left"] = point
        elif x > image_tag_center[0] and y < image_tag_center[1]:
            corners["top-right"] = point
        elif x > image_tag_center[0] and y > image_tag_center[1]:
            corners["bottom-right"] = point
        elif x < image_tag_center[0] and y > image_tag_center[1]:
            corners["bottom-left"] = point
    
    world_coords.append(world_tag_center)
    image_coords.append(image_tag_center)
    
    # t-l
    world_coords.append([
        world_tag_center[0]+INNER_DIST, 
        world_tag_center[1]-INNER_DIST, 
        world_tag_center[2]])
    image_coords.append(corners["top-left"])

    # t-r
    world_coords.append([
        world_tag_center[0]-INNER_DIST, 
        world_tag_center[1]-INNER_DIST, 
        world_tag_center[2]])
    image_coords.append(corners["top-right"])

    # b-l
    world_coords.append([
        world_tag_center[0]+INNER_DIST, 
        world_tag_center[1]+INNER_DIST, 
        world_tag_center[2]])
    image_coords.append(corners["bottom-left"])

    # b-r
    world_coords.append([
        world_tag_center[0]-INNER_DIST, 
        world_tag_center[1]+INNER_DIST, 
        world_tag_center[2]])
    image_coords.append(corners["bottom-right"])

ret, rotation_vector, translation_vector = cv2.solvePnP(
    np.array(world_coords), 
    np.array(image_coords), 
    np.array(camera_matrix).reshape(3, 3), 
    np.zeros(4), 
    flags=cv2.SOLVEPNP_ITERATIVE
)

rotation_vector, _ = cv2.Rodrigues(rotation_vector)
print("opposite view R w2c: ", rotation_vector)
print("opposite view t w2c: ", translation_vector)
R_wc = rotation_vector.T
t_wc = -rotation_vector.T @ translation_vector
print("opposite view R c2w: ", R_wc)
print("opposite view t c2w: ", t_wc)

calibration_data["oppo_camera"] = {
    "rotation_c2w": R_wc.tolist(),
    "translation_c2w": t_wc.tolist(),
    "rotation_w2c": rotation_vector.tolist(),
    "translation_w2c": translation_vector.tolist()
}


####################### SIDE CALIBRATE ################################
cap_kinect1 = cv2.VideoCapture(0)
cap_kinect1.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap_kinect1.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
ret1, image_side = cap_kinect1.read()

img = cv2.cvtColor(image_side, cv2.COLOR_BGR2GRAY)
cameraMatrix = numpy.array(parameters['K']).reshape((3,3))
camera_params = ( cameraMatrix[0,0], cameraMatrix[1,1], cameraMatrix[0,2], cameraMatrix[1,2] )
tags = at_detector.detect(img, True, camera_params, parameters['tag_size'])

color_img = visualize_det(img, tags)
cv2.imwrite("april_tag_side.png", color_img)


world_coords = []
image_coords = []
for tag in tags:
    tid = tag.tag_id
    world_tag_center = APRIL_GLOBAL_POSE[tid]
    image_tag_center = tag.center

    corners = {}
    for point in tag.corners:
        x, y = point
        if x < image_tag_center[0] and y < image_tag_center[1]:
            corners["top-left"] = point
        elif x > image_tag_center[0] and y < image_tag_center[1]:
            corners["top-right"] = point
        elif x > image_tag_center[0] and y > image_tag_center[1]:
            corners["bottom-right"] = point
        elif x < image_tag_center[0] and y > image_tag_center[1]:
            corners["bottom-left"] = point
    
    world_coords.append(world_tag_center)
    image_coords.append(image_tag_center)
    
    # t-l
    world_coords.append([
        world_tag_center[0]-INNER_DIST, 
        world_tag_center[1]-INNER_DIST, 
        world_tag_center[2]])
    image_coords.append(corners["top-left"])

    # t-r
    world_coords.append([
        world_tag_center[0]-INNER_DIST, 
        world_tag_center[1]+INNER_DIST, 
        world_tag_center[2]])
    image_coords.append(corners["top-right"])

    # b-l
    world_coords.append([
        world_tag_center[0]+INNER_DIST, 
        world_tag_center[1]-INNER_DIST, 
        world_tag_center[2]])
    image_coords.append(corners["bottom-left"])

    # b-r
    world_coords.append([
        world_tag_center[0]+INNER_DIST, 
        world_tag_center[1]+INNER_DIST, 
        world_tag_center[2]])
    image_coords.append(corners["bottom-right"])

ret, rotation_vector, translation_vector = cv2.solvePnP(
    np.array(world_coords), 
    np.array(image_coords), 
    np.array(camera_matrix).reshape(3, 3), 
    np.zeros(4), 
    flags=cv2.SOLVEPNP_ITERATIVE
)

rotation_vector, _ = cv2.Rodrigues(rotation_vector)
print("side view R w2c: ", rotation_vector)
print("side view t w2c: ", translation_vector)
R_wc = rotation_vector.T
t_wc = -rotation_vector.T @ translation_vector
print("side view R c2w: ", R_wc)
print("side view t c2w: ", t_wc)

calibration_data["side_camera"] = {
    "rotation_c2w": R_wc.tolist(),
    "translation_c2w": t_wc.tolist(),
    "rotation_w2c": rotation_vector.tolist(),
    "translation_w2c": translation_vector.tolist()
}

with open("camera_config.json", "w") as f:
    json.dump(calibration_data, f, indent=4)
