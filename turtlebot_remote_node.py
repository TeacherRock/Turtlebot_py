import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from geometry_msgs.msg import Twist    # for '/cmd_vel' 
from nav_msgs.msg import Odometry      # for '/odom' 
from sensor_msgs.msg import LaserScan  # for '/scan'

from rclpy.executors import SingleThreadedExecutor

import numpy as np
from math import sin, cos
from transforms3d import euler

from utils.Turtlebot.Turtlebot import *


class Turtlebot_Remote_Node(Node):
    def __init__(self):
        super().__init__('remote_node')
        self.odom_sub  = self.create_subscription(Odometry, 'odom', self.update_odom_data, 10)
        self.scan_sub  = self.create_subscription(LaserScan, 'scan', self.update_scan_data, 10)
        self.vel_pub   = self.create_publisher(Twist, 'cmd_vel', 10)
        self.odom_sub
        self.scan_sub
        self.odom_data = Odometry()
        self.scan_data = LaserScan()
        self.vel_data  = Twist()

        self.ini_pose_odom  = np.zeros((3, 1))
        self.R_ini_to_map = np.zeros((3, 3))
        self.T_ini_to_map = np.zeros((3, 3))

        self.pose_odom = np.zeros((3, 1))
        self.pose_map  = np.zeros((3, 1))

        self.ini = True    

    def update_odom_data(self, msg):
        self.odom_data = msg
        self.pose_odom[0] = self.odom_data.pose.pose.position.x
        self.pose_odom[1] = self.odom_data.pose.pose.position.y
        self.pose_odom[2] = self.quaternion_to_euler(self.odom_data.pose.pose.orientation)[2]
        if self.ini:
            self.ini_robot_pose()
            self.ini = False
        self.update_robot_pose()

    def update_scan_data(self, msg):
        self.scan_data = msg

    def send_cmd_vel(self, v, w):
        self.vel_data.linear.x  = v
        self.vel_data.angular.z = w
        self.vel_pub.publish(self.vel_data)

    def ini_robot_pose(self):
        self.ini_pose_odom[0] = self.odom_data.pose.pose.position.x
        self.ini_pose_odom[1] = self.odom_data.pose.pose.position.y
        self.ini_pose_odom[2] = self.quaternion_to_euler(self.odom_data.pose.pose.orientation)[2]
        self.R_ini_to_map = \
            np.array([[cos(self.ini_pose_odom[2]), -sin(self.ini_pose_odom[2]),   0],
                      [sin(self.ini_pose_odom[2]),  cos(self.ini_pose_odom[2]),   0],
                      [                         0,                           0,   1]])

        self.T_ini_to_map = \
            np.array([[1,  0, self.ini_pose_odom[0, 0]],
                      [0,  1, self.ini_pose_odom[1, 0]],
                      [0,  0,                        1]])
        print("initial pose : ", self.ini_pose_odom )
        print("self.ini_R_to_map : \n", self.R_ini_to_map)
        print("self.ini_T_to_map : \n", self.T_ini_to_map)

    def update_robot_pose(self):
        temp_pose_odom   = np.append(self.pose_odom[0:2], 1)
        temp_pose_map    = np.dot(np.linalg.inv(np.dot(self.R_ini_to_map, self.T_ini_to_map)), temp_pose_odom)
        temp_pose_map[2] = self.pose_odom[2] - self.ini_pose_odom[2]
        print(f"Current pose in  map is : ({temp_pose_map[0]:.4f}, {temp_pose_map[1]:.4f}, {temp_pose_map[2]:.4f})")
        print(f"Current pose in odom is : ({self.pose_odom[0]}, {self.pose_odom[1]}, {self.pose_odom[2]})")
        self.pose_map = temp_pose_map

    def move_robot(self, v, w):
        self.send_cmd_vel(v, w)

    def tracking(self, points):
        #####################
        ##      TO DO      ##
        #####################
        pass

    def CBF_tracking(self):
        turtlebot_Sys = Turtlebot()
        turtlebot_Sys.use_CLF = True
        turtlebot_Sys.use_CBF = True
        turtlebot_Sys.use_input_bound = True
        turtlebot_Sys.use_slack = True
        
        # change latter
        state_1, state_2, state_3, t = turtlebot_Sys.initialize()

        states = np.concatenate((state_1.T, state_2.T, state_3.T), axis=1)
        sampT = turtlebot_Sys.params.sampT

        while turtlebot_Sys.check_flag():
            t_itr_start = time.time()
            ts = np.append(ts, t)

            # Update the state in turtlebot_Sys
            turtlebot_Sys.state = self.pose_map
            states = np.concatenate((states, turtlebot_Sys.state.T), axis=0)

            turtlebot_Sys.set_ctrl_input(turtlebot_Sys)
            u_ref = turtlebot_Sys.u_ref(turtlebot_Sys.ctrl_input)
            qp_constraints = turtlebot_Sys.create_constraints(u_ref)
            try:
                u_modified = turtlebot_Sys.ctrl_qp(qp_constraints)
            except:
                print("Can not solve the QP")
                return states

            # state = turtlebot_Sys.step(u_modified)
            self.send_cmd_vel(turtlebot_Sys.params.v, u_modified)

            t  = t + sampT
            print(f"This iteration spend {time.time()-t_itr_start} s")
            if keyboard.is_pressed("esc"):
                break

    def quaternion_to_euler(self, quaternion):
        euler_angle = euler.quat2euler(
            [quaternion.w, quaternion.x, quaternion.y, quaternion.z]
        )
        return euler_angle

if __name__ == "__main__":
    rclpy.init(args=None)
    turtlebot_obj = Turtlebot_Remote_Node()

    try:
        executor = SingleThreadedExecutor()
        executor.add_node(turtlebot_obj)
        try:
            executor.spin()
        finally:
            turtlebot_obj.send_cmd_vel(0.0, 0.0)
            print("Program End")
            executor.shutdown()
            turtlebot_obj.destroy_node()
    finally:
        rclpy.shutdown()