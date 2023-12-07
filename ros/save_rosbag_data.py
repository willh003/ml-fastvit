import rosbag
import cv2
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
import rospy
import os
import time

def pad_with_zeros(n):
    s = str(n)
    if len(s) < 6:
        s = '0'*(6-len(s)) + s
    return s

def rosbag_to_img(bag_file, output_directory, topics):

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Create a bridge to convert ROS messages to OpenCV images
    bridge = CvBridge()
    time_interval = rospy.Duration(2.0)
    last_image_time = None
    # Open the rosbag
    img_count = 0
    try:
        with rosbag.Bag(bag_file, 'r') as bag:
            for topic, msg, t in bag.read_messages(topics=topics):
                # Check if the topic matches one of the specified topics
                if last_image_time is None or (t - last_image_time) >= time_interval:
                    last_image_time = t

                        
                    # Convert the CompressedImage message to an OpenCV image
                    img = bridge.compressed_imgmsg_to_cv2(msg)

                    # Generate a file name based on the topic and timestamp

                    file_name = f"{pad_with_zeros(img_count)}.png"
                    
                    # Save the image to the output directory
                    file_path = os.path.join(output_directory, file_name)
                    cv2.imwrite(file_path, img)
                    img_count += 1
                    
                    print(f"Saved {file_name}")
    except Exception as e:
        print(repr(e))

        print(f'{bag_file} corrupted')

if __name__=="__main__":
   # anymal_topics = ['/wide_angle_camera_front/image_color_rect/compressed', '/wide_angle_camera_rear/image_color_rect/compressed']
    #bag_file = '/home/pcgta/Documents/eth/data/Jul26_site_visit/output.bag'  
    
    sacson_topics = ['/fisheye_image/compressed']
    sacson_dir = 'full_data/sacson_raw'
    output_directory = 'full_data/sacson'  
    for data_dir in os.listdir(sacson_dir):
        path = os.path.join(sacson_dir, data_dir)
        if not os.path.isdir(path):
            continue
        
        for bag_file in os.listdir(path):
            
            bag_title, bag_ext = os.path.splitext(bag_file)
            if bag_ext != '.bag':
                continue
            
            output_path = os.path.join(output_directory, data_dir, bag_title)
            rosbag_to_img(os.path.join(path, bag_file), output_path, sacson_topics)
