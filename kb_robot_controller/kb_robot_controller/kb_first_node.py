#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

class MyNode(Node):

    def __init__(self):  # Fix the init method (double underscores)
        super().__init__("first_node")  # Fix the call to super()
        #self.get_logger().info("Hello, this is KB ROS2 new v2 ")  # Log message
        self.counter_ = 0
        self.create_timer(1.0, self.timer_callback)

    def timer_callback(self):
        self.get_logger().info("hello kb"+str(self.counter_))
        self.counter_+= 1

def main(args=None):
    rclpy.init(args=args)
    node = MyNode()  # Create the node instance
    rclpy.spin(node)  # Keep the node running and processing callbacks

    rclpy.shutdown()  # Shutdown the ROS client library

if __name__ == '__main__':  # Corrected if condition if we want to run directly from terminal
    main()
