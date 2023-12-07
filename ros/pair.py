import time
import rospy
import csv
from sensor_msgs.msg import Image
from ackermann_msgs.msg import AckermannDriveStamped

class SynchronizedDataRecorder:
    def __init__(self):
        self.csv_file = open(f'lab_data_{time.time()}.csv', 'w')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(['timestamp', 'image_bytes', 'steering_angle','speed'])
        self.teleop_data = None

        rospy.Subscriber("/car/car/camera/color/image_raw", Image, self.camera_callback)
        rospy.Subscriber("/car/mux/ackermann_cmd_mux/input/teleop", AckermannDriveStamped, self.ackermann_cmd_callback)

    def camera_callback(self, msg):
        if self.teleop_data is not None:
            timestamp = str(msg.header.stamp)
            image_data = msg.data
            steering_angle = str(self.teleop_data.drive.steering_angle)
            speed = str(self.teleop_data.drive.speed)

            self.csv_writer.writerow([timestamp, image_data, steering_angle, speed])
            self.teleop_data = None
            print("Synchronized pair saved at timestamp:", timestamp)

    def ackermann_cmd_callback(self, msg):
        self.teleop_data = msg

    def record_data(self):
        rospy.init_node('data_recorder', anonymous=True)
        rospy.spin()
        self.csv_file.close()
        print("Data recording completed. CSV file saved.")

if __name__ == '__main__':
    recorder = SynchronizedDataRecorder()
    recorder.record_data()
