import rospy
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive
from std_msgs.msg import String

class ActionSender:

    def __init__(self, action_topic):
        rospy.init_node('action_sender')
        self.action_topic = action_topic

        self.action_pub = rospy.Publisher(self.action_topic, AckermannDriveStamped, queue_size=10)
        self.string_pub = rospy.Publisher('/test_topic', String, queue_size=10)
        rospy.Timer(rospy.Duration(secs = 1.0), self.pub_string)
        self.fv = 0

    def pub_string(self, t):
        self.string_pub.publish(String(data='hello'))

    def update_action(self, timer_event):
        if self.fv == 0:
            self.fv = 1
        else:
            self.fv = 0
        
        msg = AckermannDriveStamped()
        msg.drive = AckermannDrive()
        msg.drive.speed = self.fv
        print(f'sending {self.fv}')

        self.action_pub.publish(msg)

if __name__=="__main__":
    sender = ActionSender('/car/mux/ackermann_cmd_mux/input/navigation')
    rospy.spin()
