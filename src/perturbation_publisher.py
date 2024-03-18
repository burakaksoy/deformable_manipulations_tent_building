#!/usr/bin/env python3

import sys
import rospy

import numpy as np
import time

from geometry_msgs.msg import Twist, Point, Quaternion, Pose
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker
from std_msgs.msg import Float32

import tf_conversions
import tf.transformations as tf_trans

from dlo_simulator_stiff_rods.msg import SegmentStateArray

class PerturbationPublisherNode:
    def __init__(self):
        self.pub_rate_odom = rospy.get_param("~pub_rate_odom", 50)

        self.deformable_object_state_topic_name = rospy.get_param("/dlo_state_topic_name") # subscribed

        self.custom_static_particles = None
        self.odom_topic_prefix = None
        while (not self.custom_static_particles):
            try:
                self.custom_static_particles = rospy.get_param("/custom_static_particles") # Default static particles 
                self.odom_topic_prefix = rospy.get_param("/custom_static_particles_odom_topic_prefix") # published
            except:
                rospy.logwarn("No particles obtained from ROS parameters!.")
                time.sleep(0.5)

        self.delta_x = rospy.get_param("~delta_x", 0.1) # m
        self.delta_y = rospy.get_param("~delta_y", 0.1) # m
        self.delta_z = rospy.get_param("~delta_z", 0.1) # m
        self.delta_th_x = rospy.get_param("~delta_th_x", 0.1745329) # rad
        self.delta_th_y = rospy.get_param("~delta_th_y", 0.1745329) # rad
        self.delta_th_z = rospy.get_param("~delta_th_z", 0.1745329) # rad

        # We will hold each follower current position and orientation in the following dictionaries
        # so each dict will have as many as the follower particles
        self.particle_positions = {}
        self.particle_orientations = {} 
        self.particle_twists = {}
        self.initial_values_set = False  # Initialization state variable

        self.sub_state = rospy.Subscriber(self.deformable_object_state_topic_name, SegmentStateArray, self.state_array_callback, queue_size=10)

        # we will create a 3 odom publisher for each follower particle
        # so each of the following dictionaries will have elements as many as the follower particles
        self.odom_publishers_delta_x = {}
        self.odom_publishers_delta_y = {}
        self.odom_publishers_delta_z = {}
        self.odom_publishers_delta_th_x = {}
        self.odom_publishers_delta_th_y = {}
        self.odom_publishers_delta_th_z = {}

        for particle in self.custom_static_particles:
            self.odom_publishers_delta_x[particle]    = rospy.Publisher(self.odom_topic_prefix + str(particle) + "_x",    Odometry, queue_size=10)
            self.odom_publishers_delta_y[particle]    = rospy.Publisher(self.odom_topic_prefix + str(particle) + "_y",    Odometry, queue_size=10)
            self.odom_publishers_delta_z[particle]    = rospy.Publisher(self.odom_topic_prefix + str(particle) + "_z",    Odometry, queue_size=10)
            self.odom_publishers_delta_th_x[particle] = rospy.Publisher(self.odom_topic_prefix + str(particle) + "_th_x", Odometry, queue_size=10)
            self.odom_publishers_delta_th_y[particle] = rospy.Publisher(self.odom_topic_prefix + str(particle) + "_th_y", Odometry, queue_size=10)
            self.odom_publishers_delta_th_z[particle] = rospy.Publisher(self.odom_topic_prefix + str(particle) + "_th_z", Odometry, queue_size=10)

            # self.odom_publishers_delta_x[particle]    = rospy.Publisher("x"    + "/" + str(particle) + "/" + self.odom_topic_prefix , Odometry, queue_size=10)
            # self.odom_publishers_delta_y[particle]    = rospy.Publisher("y"    + "/" + str(particle) + "/" + self.odom_topic_prefix , Odometry, queue_size=10)
            # self.odom_publishers_delta_z[particle]    = rospy.Publisher("z"    + "/" + str(particle) + "/" + self.odom_topic_prefix , Odometry, queue_size=10)
            # self.odom_publishers_delta_th_x[particle] = rospy.Publisher("th_x" + "/" + str(particle) + "/" + self.odom_topic_prefix , Odometry, queue_size=10)
            # self.odom_publishers_delta_th_y[particle] = rospy.Publisher("th_y" + "/" + str(particle) + "/" + self.odom_topic_prefix , Odometry, queue_size=10)
            # self.odom_publishers_delta_th_z[particle] = rospy.Publisher("th_z" + "/" + str(particle) + "/" + self.odom_topic_prefix , Odometry, queue_size=10)

        self.odom_pub_timer = rospy.Timer(rospy.Duration(1. / self.pub_rate_odom), self.odom_pub_timer_callback)


    def odom_pub_timer_callback(self,event):
        # Do not proceed until the initial values have been set
        if (not self.initial_values_set):
            return
        
        for particle in self.custom_static_particles:
            t_now = rospy.Time.now()
            frame_id = "map"

            # Prepare Odometry message
            odom = Odometry()
            odom.header.stamp = t_now
            odom.header.frame_id = frame_id 
            
            #------- POSITIONS --------------------
            # ----------------------------------------------------------------------------------------
            pose_delta_x = Pose()
            pose_delta_x.position.x = self.particle_positions[particle].x + self.delta_x
            pose_delta_x.position.y = self.particle_positions[particle].y 
            pose_delta_x.position.z = self.particle_positions[particle].z 

            # Keep the same orientation
            pose_delta_x.orientation = self.particle_orientations[particle]

            odom.pose.pose = pose_delta_x
            self.odom_publishers_delta_x[particle].publish(odom)
            # ----------------------------------------------------------------------------------------
            pose_delta_y = Pose()
            pose_delta_y.position.x = self.particle_positions[particle].x # + self.delta_x
            pose_delta_y.position.y = self.particle_positions[particle].y + self.delta_y
            pose_delta_y.position.z = self.particle_positions[particle].z # + self.delta_z

            # Keep the same orientation
            pose_delta_y.orientation = self.particle_orientations[particle]

            odom.pose.pose = pose_delta_y
            self.odom_publishers_delta_y[particle].publish(odom)
            # ----------------------------------------------------------------------------------------
            pose_delta_z = Pose()
            pose_delta_z.position.x = self.particle_positions[particle].x # + self.delta_x
            pose_delta_z.position.y = self.particle_positions[particle].y # + self.delta_y
            pose_delta_z.position.z = self.particle_positions[particle].z + self.delta_z

            # Keep the same orientation
            pose_delta_z.orientation = self.particle_orientations[particle]

            odom.pose.pose = pose_delta_z
            self.odom_publishers_delta_z[particle].publish(odom)

            #------- ORIENTATIONS --------------------
            # ----------------------------------------------------------------------------------------
            # Perturbation in X orientation
            pose_delta_th_x = Pose()
            # Keep the same position
            pose_delta_th_x.position = self.particle_positions[particle]

            # Calculate the perturbed orientation
            quaternion_delta_th_x = tf_trans.quaternion_about_axis(self.delta_th_x, (1, 0, 0))
            
            pose_delta_th_x.orientation = Quaternion(*tf_trans.quaternion_multiply(
                quaternion_delta_th_x,
                [self.particle_orientations[particle].x,
                self.particle_orientations[particle].y,
                self.particle_orientations[particle].z,
                self.particle_orientations[particle].w]))
            
            odom.pose.pose = pose_delta_th_x
            self.odom_publishers_delta_th_x[particle].publish(odom)

            # ----------------------------------------------------------------------------------------
            # Perturbation in Y orientation
            pose_delta_th_y = Pose()
            # Keep the same position
            pose_delta_th_y.position = self.particle_positions[particle]

            # Calculate the perturbed orientation
            quaternion_delta_th_y = tf_trans.quaternion_about_axis(self.delta_th_y, (0, 1, 0))
            
            pose_delta_th_y.orientation = Quaternion(*tf_trans.quaternion_multiply(
                quaternion_delta_th_y,
                [self.particle_orientations[particle].x,
                self.particle_orientations[particle].y,
                self.particle_orientations[particle].z,
                self.particle_orientations[particle].w]))
            
            odom.pose.pose = pose_delta_th_y
            self.odom_publishers_delta_th_y[particle].publish(odom)

            # ----------------------------------------------------------------------------------------
            # Perturbation in Z orientation
            pose_delta_th_z = Pose()
            # Keep the same position
            pose_delta_th_z.position = self.particle_positions[particle]

            # Calculate the perturbed orientation
            quaternion_delta_th_z = tf_trans.quaternion_about_axis(self.delta_th_z, (0, 0, 1))
            
            pose_delta_th_z.orientation = Quaternion(*tf_trans.quaternion_multiply(
                quaternion_delta_th_z,
                [self.particle_orientations[particle].x,
                self.particle_orientations[particle].y,
                self.particle_orientations[particle].z,
                self.particle_orientations[particle].w]))

            odom.pose.pose = pose_delta_th_z
            self.odom_publishers_delta_th_z[particle].publish(odom)



    def state_array_callback(self, states_msg):
        for particle in self.custom_static_particles:
            self.particle_positions[particle] = states_msg.states[particle].pose.position
            self.particle_orientations[particle] = states_msg.states[particle].pose.orientation
            self.particle_twists[particle] = states_msg.states[particle].twist

        if not self.initial_values_set:
            # After all initial relative positions and orientations have been calculated, set the initialization state variable to True
            self.initial_values_set = True


if __name__ == "__main__":
    rospy.init_node('perturbation_publisher_node', anonymous=False)

    node = PerturbationPublisherNode()

    rospy.spin()
