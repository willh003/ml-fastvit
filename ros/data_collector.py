#!/usr/bin/env python

# Copyright (c) 2019, The Personal Robotics Lab, The MuSHR Team, The Contributors of MuSHR
# License: BSD 3-Clause. See LICENSE.md file in root directory.

import rospy
from ackermann_msgs.msg import AckermannDriveStamped
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os
import message_filters
import csv

# def obs_callback(data, *args):
#     print("Loggy.")
#     return

# def cmd_callback(data, *args):
#     print("Logger.")
#     return

itr = 0
only_every = 20
class DataListener:
    def __init__(self, log_dir, cmd_topic='/car/mux/ackermann_cmd_mux/input/teleop', img_topic='/car/car/camera/color/image_raw'):
        self.max_velocity = rospy.get_param("~speed", 2.0)
        self.max_steering_angle = rospy.get_param("~max_steering_angle", 0.34)

        self.cmd_sub = message_filters.Subscriber(cmd_topic, AckermannDriveStamped, queue_size=10)
        self.obs_sub = message_filters.Subscriber(img_topic, Image, queue_size=10)

        # # test
        # rospy.Subscriber(cmd_topic, AckermannDriveStamped, cmd_callback)
        # rospy.Subscriber(img_topic, Image, obs_callback)

        self.bridge= CvBridge()
        self.ts = message_filters.ApproximateTimeSynchronizer([self.obs_sub, self.cmd_sub], 100, 1)

        self.ts.registerCallback(self.obs_cb)
        self.data_dir = log_dir

        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

        # with open(f'{self.data_dir}/action.csv', 'wb') as f:
        #     f.write('t, v, w')
        with open(f'{self.data_dir}/action.csv', 'w') as f:
            f.write('t, v, w')


    def obs_cb(self, image, action):
        print("Callback triggered.")
        if itr < only_every:
            itr += 1
            return
        time = action.header.stamp
        angle = action.drive.steering_angle
        speed = action.drive.speed

        cv_image = self.bridge.compressed_imgmsg_to_cv2(image)

        img_dir = os.path.join(self.data_dir, 'images')
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)

        # cv2.imwrite(os.path.join(img_dir,f'{time}.jpg'), cv_image)
        # with open(f'{self.data_dir}/action.csv', 'wb') as f:
        #     f.write(f'{time}, {speed}, {angle}')
        cv2.imwrite(os.path.join(img_dir, f'{time}.jpg'), cv_image)
        with open(f'{self.data_dir}/action.csv', 'a') as f:
            f.write(f'{time.to_sec()}, {speed}, {angle}')

        #rospy.sleep(.2)
        
from pathlib import Path
if __name__ == "__main__":
    rospy.init_node('data_listener_node')
    
    log_dir = Path("./datacollectortest").resolve()

    data_listener = DataListener(log_dir)

    print("MUSHR Listener Spinning...")
    rospy.spin()




