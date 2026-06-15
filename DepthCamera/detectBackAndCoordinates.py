# This script currently utilizes the YOLO26 keypoint model to identify the back of the compartment holding
# the jetson nano of the lead vehicle WITH LED beacon attachments. The YOLO26 model is trained to
# identify the back of the vehicle and mark the center of each beacon in order to construct the necessary
# points needed for its yaw, in relation to the follower vehicle. The depth camera also utilizes these points
# to calculate the xyz coordinates/position of the lead vehicle. The formatData function currently
# converts the xyz and yaw data into hexadecimal digits for possible serial communication with CAN.
# This script is currently made to run on a pc with a keyboard.

# Notes concerning current YOLO26 model:
# still has floating points when in unfamiliar positions (mostly at different heights and rotations
# of the depth camera (pitch and roll))

import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO
import math
import serial
import time
import csv

# to convert metrics for testing (meters to inches)
MTI = 39.37

# meters per bit for hex conversion
DIST_MPB = 0.00245
LR_MPB = 0.0196
UD_MPB = 0.0625
YAW_MPB = 0.625

# Depth camera streaming settings
RES_WIDTH = 640
RES_HEIGHT = 480
FPS = 30

def main():
    # ser = serial.Serial('COM3', 9600, timeout=1) # open when serial is needed
    time.sleep(2)

    pipeline = rs.pipeline()
    config = rs.config()

    # streaming settings
    config.enable_stream(rs.stream.color, RES_WIDTH, RES_HEIGHT, rs.format.bgr8, FPS)
    config.enable_stream(rs.stream.depth, RES_WIDTH, RES_HEIGHT, rs.format.z16, FPS)

    pipeline.start(config)

    # select machine learning model
    model = YOLO("BOVandBeaconKPModel2.pt")

    # for color and depth stream alignment
    align = rs.align(rs.stream.color)

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            if not color_frame or not depth_frame:
                continue
            
            color_image = np.asanyarray(color_frame.get_data())

            # use ML model on aligned frame
            results = model(color_image)

            # initialize array of keypoint coordinates in resolution coords (RES_WIDTHxRES_HEIGHT)
            points_rc = [[-1, -1] for _ in range(4)]

            # initialize array of keypoint coordinates in meters (x, y, z)
            points_mc = [[-1, -1, -1] for _ in range(4)]

            centerFound = False

            # extract data from keypoints and translate into 3D coordinates
            for result in results:
                for i, box in enumerate(result.boxes):
                    # check if it's a beacon or the center of the vehicle
                    if model.names[int(box.cls)] == 'beacons':
                        if float(box.conf) < 0.7:
                            continue
                        print(f"beacons detected")
                        # get box dimensions and display on stream
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        for j, kp in enumerate(result.keypoints.xy[i]):
                            # confidence of keypoint j of object i
                            if result.keypoints.conf[i][j] < 0.7:
                                continue
                            x = int(kp[0])
                            y = int(kp[1])

                            # convert to 3D coordinates
                            if x == RES_WIDTH:
                                x -= 1
                            if y == RES_HEIGHT:
                                y -= 1
                            depth = depth_frame.get_distance(x, y)
                            depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
                            point = rs.rs2_deproject_pixel_to_point(depth_intrin, [x, y], depth)

                            if j == 0:
                                cv2.circle(color_image, (x, y), 4, (0, 0, 255), 3) # red
                            elif j == 1:
                                cv2.circle(color_image, (x, y), 4, (255, 0, 0), 3) # blue
                            elif j == 2:
                                cv2.circle(color_image, (x, y), 4, (0, 255, 0), 3) # green
                            elif j == 3:
                                cv2.circle(color_image, (x, y), 4, (255, 0, 255), 3) # purple
                            else:
                                cv2.circle(color_image, (x, y), 4, (0, 0, 0), 3) # black

                            # add into arrays
                            points_rc[j] = [x, y]
                            points_mc[j] = point
                            print(f"Keypoint {j} depth: {depth}, coords: {points_mc[j]}")
                    elif model.names[int(box.cls)] == 'vehicle':
                        if float(box.conf) < 0.7:
                            continue
                        if result.keypoints.conf[i][0] < 0.7:
                            continue
                        print(f"vehicle with center detected")
                        centerFound = True
                        # get box dimensions and display on stream
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 0, 255), 3)

                        cx = int(result.keypoints.xy[i][0][0])
                        cy = int(result.keypoints.xy[i][0][1])
                        cz = depth_frame.get_distance(cx, cy)
                        cv2.circle(color_image, (cx, cy), 4, (255, 255, 255), 3) # white
                        # convert center coordinates to meters and display
                        depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
                        centerCoords = rs.rs2_deproject_pixel_to_point(depth_intrin, [cx, cy], cz)
            # calculate yaw using detected beacon positions
            # first check if the points exist (need 3 existing)
            n = calculateNormal(points_mc)
            if n[0] == -1:
                print(f"3 points could not be identified for yaw")
                yaw = -1
            else:
                yaw = getYaw(n)
            
            # edge cases
            # vehicle center detected and 0-2 beacons, move towards center
            translate = translateDetections(points_rc, centerFound)
            if translate == 0:
                locatable = True
            # no vehicle center but bottom left beacon detected, look/move more right
            elif translate == 1:
                locatable = False
                print("move/look more right, left beacon detected")
            # no vehicle center but bottom right beacon detected, look/move more left
            elif translate == 2:
                locatable = False
                print("move/look more left, right beacon detected")
            # 3+ beacons and center
            elif translate == 3:
                locatable = True
            # none detected
            else:
                locatable = False
            
            # MAYBE IMPORTANT: assuming that if 3 beacons are detected, then center should be in view as well
            if locatable:
                # for writing to csv file
                # with open("output.csv", mode="a", newline="", encoding="utf-8") as file:
                #     writer = csv.writer(file)
                #     data = (f"{centerCoords[0]} {centerCoords[1]} {centerCoords[2]} {yaw}")
                #     writer.writerow([data])
                # print coords to center with yaw
                print(f"Inches: X: {round(centerCoords[0] * MTI, 2)}, Y: {round(centerCoords[1] * MTI, 2)}, Z: {round(centerCoords[2] * MTI, 2)}, Yaw: {round(yaw, 2)}")
                print(f"Meters: X: {round(centerCoords[0], 2)}, Y: {round(centerCoords[1], 2)}, Z: {round(centerCoords[2], 2)}, Yaw: {round(yaw, 2)}")
                
                # format data for CAN bus (x = left and right, y = up and down, z = back and forth)
                data = formatData(centerCoords[0], centerCoords[1], centerCoords[2], yaw, translate)
                
                # write the serial data and display
                # ser.write(data)
                print(f"Data in hex: {data}")

            cv2.imshow('RBG Stream', color_image)
            key = cv2.waitKey(1)

            # stop when 'q' or 'esc' is pressed
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break
    finally:
        # ser.close()
        pipeline.stop()

def calculateNormal(points):
    # check if there are at least 3 valid points
    i = 0
    for point in points:
        if point[0] != -1:
            if i == 0:
                p1 = point
            if i == 1:
                p2 = point
            if i == 2:
                p3 = point
            i += 1
    
    # return vector outside of range if less than 3 valid points
    if i < 3:
        return [-1, -1, -1]

    # calculate normal vector
    u = [p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]]
    v = [p3[0] - p1[0], p3[1] - p1[1], p3[2] - p1[2]]
    n = [u[1]*v[2] - u[2]*v[1], u[2]*v[0] - u[0]*v[2], u[0]*v[1] - u[1]*v[0]]

    # keep normal plane consistent (only viewing from the back)
    if n[2] < 0:
        n[2] = -n[2]

    return n

def getYaw(n):
    yaw_rad = math.atan2(n[2], n[0])
    yaw_deg = math.degrees(yaw_rad) - 90
    return yaw_deg

def translateDetections(points_rc, centerFound):
    # 0 = center and < two beacons, 1 = bottom left, 2 = bottom right, 3 = all, -1 = none
    count = 0
    for point in points_rc:
        if point[0] != -1:
            count += 1
    if centerFound and count <= 2:
        return 0
    elif not centerFound and count == 1:
        if points_rc[3] == -1:
            return 1
        elif points_rc[2] == -1:
            return 2
        else:
            return -1
    elif centerFound and count >= 3:
        return 3
    else:
        return -1

# format data into CAN bus hex number 0x370 00(x) 0(y) 000(z) 00(yaw angle)
# for depth up to ~10m (0.00245m per bit for 4096), left/right from -2.5m to 2.5m (0.0196m per bit for 256),
# up/down from -0.5m to 0.5m (0.0625m per bit for 16), yaw angle from -80 to 80 degrees (0.625 degrees per bit for 256)
def formatData(x, y, z, yaw, translate):
    # format distance (depth)
    if z >= 4096 * DIST_MPB:
        z_hex = hex(4096)
    else:
        z_hex = hex(round(z / DIST_MPB))

    # format left/right, can be negative so normalize: 0m = 128 bits
    if abs(x) >= (256 * LR_MPB) / 2:
        if x < 0:
            x_hex = hex(0)
        else:
            x_hex = hex(255)
    else:
        x_hex = hex(128 + round(x / LR_MPB))

    # format up/down (0m = 8 bits)
    if abs(y) >= (16 * UD_MPB) / 2:
        if y < 0:
            y_hex = hex(0)
        else:
            y_hex = hex(15)
    else:
        y_hex = hex(8 + round(y / UD_MPB))

    # format yaw angle (0 degrees = 128 bits)
    if abs(yaw) >= (256 * YAW_MPB) / 2:
        if yaw < 0:
            yaw_hex = hex(0)
        else:
            yaw_hex = hex(255)
    else:
        yaw_hex = hex(128 + round(yaw / YAW_MPB))
    
    if translate == 0:
        yaw_hex = hex(128)

    return "0x370 " + x_hex.removeprefix("0x") + y_hex.removeprefix("0x") + z_hex.removeprefix("0x") + yaw_hex.removeprefix("0x")

main()