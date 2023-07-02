#!/usr/bin/env python

import rospy
from arbotix_msgs.srv import SetSpeed
from std_msgs.msg import Float64


def head_motor_test_init(init_pos):
    """
    Move test head motor

    Args:
        init_pos (list): init positions

    """

    rospy.init_node("init_test_pos")
    speed_srv_list = []
    pos_pub_list = []
    name='full_head'
    node_name='neck_pan'
    speed_srv_name = node_name + name + '/set_speed'
    speed_srv_list.append(rospy.ServiceProxy(speed_srv_name, SetSpeed))
    pos_pub_name = node_name + name + '/command'
    pos_pub_list.append(rospy.Publisher(pos_pub_name, Float64, queue_size=5))

    for speed_srv in speed_srv_list:
        try:
            speed_srv(2.0)
        except rospy.ServiceException as e:
            print("Set Speed Failed ...")

    for pos_pub in pos_pub_list:
        pos_msg = Float64()
        pos_msg.data = init_pos
        pos_pub.publish(pos_msg)
        
    print("running")
    rospy.spin()
    


if __name__ == '__main__':
    head_motor_test_init(0)
