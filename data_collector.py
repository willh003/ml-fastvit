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

class DataListener:
    def __init__(self, cmd_topic, img_topic, log_dir):
        self.max_velocity = rospy.get_param("~speed", 2.0)
        self.max_steering_angle = rospy.get_param("~max_steering_angle", 0.34)

        self.cmd_sub = message_filters.Subscriber(cmd_topic, AckermannDriveStamped, queue_size=10)
        self.obs_sub = message_filters.Subscriber(img_topic, Image, queue_size=10)


        self.bridge= CvBridge()
        self.ts = message_filters.ApproximateTimeSynchronizer([self.obs_sub, self.cmd_sub], 10, .05)
        self.ts.registerCallback(self.obs_cb)
        self.data_dir = log_dir

        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

        with open(f'{self.data_dir}/action.csv', 'wb') as f:
            f.write('t, v, w')


    def obs_cb(self, image, action):
        time = action.header.stamp
        angle = action.drive.steering_angle
        speed = action.drive.speed

        cv_image = self.bridge.compressed_imgmsg_to_cv2(image)

        img_dir = os.path.join(self.data_dir, 'images')
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)

        cv2.imwrite(os.path.join(img_dir,f'{time}.jpg'), cv_image)
        with open(f'{self.data_dir}/action.csv', 'wb') as f:
            f.write(f'{time}, {speed}, {angle}')

        rospy.sleep(.2)
        






