#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from turtlesim.msg import Pose
from geometry_msgs.msg import Twist


class TurtkeControllerNode(Node):

    def __init__(self):
        super().__init__("kb_turtle_controller")
        self.cmd_vel_publisher_ = self.create_publisher(
            Twist,"/turtle1/cmd_vel", 10)
        self.pose_suscriber_ = self.create_subscription(
            Pose, "/turtle1/pose",self.pose_callback,10)
        self.get_logger().info("kb tutle controller started")

    def pose_callback(self, pose: Pose):
        cmd = Twist()
        if pose.x > 9.0 or pose.x < 2.0 or pose.y > 9.0 or pose.y < 2.0: # here we achive our goal on Right part no collision , then next lest's add the rest condition to
            cmd.linear.x = 2.0
            cmd.angular.z = 0.5#2.0
        else:
            cmd.linear.x = 7.0  #here no y and z , just to see how it goes only in the x direction with out changeing direction meaning angular vel in z axis is also zero 
            cmd.angular.z = 0.0    
        
        self.cmd_vel_publisher_.publish(cmd) # now we just publish the cmd to see how it work . No pose data used for the moment!




def main(args=None):
    rclpy.init(args=args)
    node = TurtkeControllerNode()
    rclpy.spin(node)
    rclpy.shutdown()

