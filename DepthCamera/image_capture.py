import pyrealsense2 as rs
import numpy as np
import cv2
import os

# change save location accordingly
saveLocation = r"C:\Users\ethan\Intel RealSense\code\beacon image data"
os.makedirs(saveLocation, exist_ok = True)

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

i = 0
pipeline.start(config)
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
        depth_image = np.asanyarray(depth_frame.get_data())
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        cv2.imshow('Color', color_image)

        key = cv2.waitKey(1)
        
        # save frame as image after clicking space bar
        if key & 0xFF == ord(' '):
            # naming file
            color_filename = os.path.join(saveLocation, f"color_im_{i}.jpg")
            depth_filename = os.path.join(saveLocation, f'depth_im_{i}.png')

            # writing file
            cv2.imwrite(color_filename, color_image)
            cv2.imwrite(depth_filename, depth_colormap)

            i += 1
        # ends when pressing 'q' or reaching 2000 images
        elif key == ord('q') or i == 2000:
            cv2.destroyAllWindows()
            break

finally:
    pipeline.stop()