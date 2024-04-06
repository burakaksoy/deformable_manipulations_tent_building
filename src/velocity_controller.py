#!/usr/bin/env python3

import sys
import rospy
import numpy as np
import time
import math
from datetime import datetime

from geometry_msgs.msg import Twist, Point, Quaternion, Pose, Wrench, Vector3
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker
from std_msgs.msg import Float32, Float32MultiArray

from dlo_simulator_stiff_rods.msg import SegmentStateArray
from dlo_simulator_stiff_rods.msg import MinDistanceDataArray

from deformable_manipulations_tent_building.msg import ControllerStatus

from std_srvs.srv import SetBool, SetBoolResponse
from std_srvs.srv import Empty, EmptyResponse

import tf.transformations as tf_trans

import cvxpy as cp

def format_number(n, digits=4):
    # Round to the specified number of decimal digits
    formatted = f"{n:.{digits}f}"
    # Remove trailing zeros and decimal point if necessary
    formatted = formatted.rstrip('0').rstrip('.')
    # Replace '-0' with '0'
    if formatted == '-0':
        return '0'
    return formatted

# def pretty_print_array(arr, digits=4):
#     np.set_printoptions(formatter={'float': lambda x: format_number(x, digits)}, 
#                         linewidth=150, 
#                         suppress=True)  # Suppress small numbers
#     print(arr)

def pretty_print_array(arr, precision=2, width=5):
    format_spec = f"{{:>{width}.{precision}f}}"
    for row in arr:
        print(" ".join(format_spec.format(value) if value != 0 else " " * width for value in row))

class NominalController:
    def __init__(self, Kp=1.0, Kd=0.0, MAX_TIMESTEP = 0.1):
        # PD gains
        self.Kp = np.array(Kp)
        self.Kd = np.array(Kd)

        self.last_time = None # time.time()

        # Velocity commands will only be considered if they are spaced closer than MAX_TIMESTEP
        self.MAX_TIMESTEP = MAX_TIMESTEP

    def output(self, error, vel):
        # Calculate the output
        output = self.Kp * error - self.Kd * vel

        return output
    
    def get_dt(self):
        current_time = time.time()
        if self.last_time:
            dt = current_time - self.last_time
            self.last_time = current_time
            if dt > self.MAX_TIMESTEP:
                dt = 0.0
                rospy.logwarn("Controller took more time than specified MAX_TIMESTEP duration, resetting to 0.")
            return dt
        else:
            self.last_time = current_time
            return 0.0
        
class VelocityControllerNode:
    def __init__(self):
        self.enabled = False  # Flag to enable/disable controller

        # To Store the time when the controller is enabled/disabled
        self.controller_enabled_time = 0.0
        self.controller_enabled_time_str = ""
        self.controller_disabled_time = 0.0
        # iteration counter for the controller to calculate the average performance
        self.controller_itr = 0 

        self.nominal_control_enabled = rospy.get_param("~nominal_control_enabled", True)
        self.obstacle_avoidance_enabled = rospy.get_param("~obstacle_avoidance_enabled", True)
        self.stress_avoidance_enabled = rospy.get_param("~stress_avoidance_enabled", True)

        # Create the service server for enable/disable the controller
        self.set_enable_controller_server = rospy.Service('~set_enable_controller', SetBool, self.set_enable_controller)

        self.pub_rate_odom = rospy.get_param("~pub_rate_odom", 50)

        self.custom_static_particles = None
        self.odom_topic_prefix = None
        while (not self.custom_static_particles):
            try:
                self.custom_static_particles = rospy.get_param("/custom_static_particles") # Default static particles 
                # self.custom_static_particles = [5]
                self.odom_topic_prefix = rospy.get_param("/custom_static_particles_odom_topic_prefix") # published
            except:
                rospy.logwarn("No particles obtained from ROS parameters!.")
                time.sleep(0.5)
        
        # Create information publishers for the evaluation of the controller
        self.info_pub_controller_status = rospy.Publisher("~info_controller_status", ControllerStatus, queue_size=10) # Publishes the status of the controller when it is enabled/disabled

        self.info_pub_target_pos_error_avr_norm = rospy.Publisher("~info_target_pos_error_avr_norm", Float32, queue_size=10) # average norm of the target position errors
        self.info_pub_target_ori_error_avr_norm = rospy.Publisher("~info_target_ori_error_avr_norm", Float32, queue_size=10) # average norm of the target orientation errors
        
        self.info_pub_overall_min_distance_collision = rospy.Publisher("~info_overall_min_distance_collision", Float32, queue_size=10)
        
        self.info_pub_stress_avoidance_performance = rospy.Publisher("~info_stress_avoidance_performance", Float32, queue_size=10)
        self.info_pub_stress_avoidance_performance_avr = rospy.Publisher("~info_stress_avoidance_performance_avr", Float32, queue_size=10)


        self.info_pub_wildcard_array = rospy.Publisher("~info_wildcard_array", Float32MultiArray, queue_size=10)
        self.info_pub_wildcard_scalar = rospy.Publisher("~info_wildcard_scalar", Float32, queue_size=10)

        # Create an (odom) Publisher for each static particle (i.e. held particles by the robots) as control output to them.
        self.odom_publishers = {}
        for particle in self.custom_static_particles:
            self.odom_publishers[particle] = rospy.Publisher(self.odom_topic_prefix + str(particle), Odometry, queue_size=10)

        self.delta_x = rospy.get_param("/perturbation_publisher/delta_x", 0.1) # m
        self.delta_y = rospy.get_param("/perturbation_publisher/delta_y", 0.1) # m
        self.delta_z = rospy.get_param("/perturbation_publisher/delta_z", 0.1) # m
        self.delta_th_x = rospy.get_param("/perturbation_publisher/delta_th_x", 0.1745329) # rad
        self.delta_th_y = rospy.get_param("/perturbation_publisher/delta_th_y", 0.1745329) # rad
        self.delta_th_z = rospy.get_param("/perturbation_publisher/delta_th_z", 0.1745329) # rad

        # Controller gains  [x,y,z, Rx,Ry,Rz]
        self.kp = np.array(rospy.get_param("~kp", [1.0,1.0,1.0, 1.0,1.0,1.0]))
        self.kd = np.array(rospy.get_param("~kd", [0.0,0.0,0.0, 0.0,0.0,0.0]))

        self.k_low_pass_ft = rospy.get_param("~k_low_pass_ft", 0.9) # low pass filter coefficient for the ft values of the previous values
        self.k_low_pass_min_d = rospy.get_param("~k_low_pass_min_d", 0.5) # low pass filter coefficient for the minimum distance values of the previous values

        self.max_linear_velocity = rospy.get_param("~max_linear_velocity", 0.1) # m/s
        self.max_angular_velocity = rospy.get_param("~max_angular_velocity", 0.15) # rad/s

        self.acceptable_pos_err_avr_norm = rospy.get_param("~acceptable_pos_err_avr_norm", 0.01) # m
        self.acceptable_ori_err_avr_norm = rospy.get_param("~acceptable_ori_err_avr_norm", 0.1) # rad

        # Particle/Segment ids of the tip points of the tent pole 
        # to be placed into the grommets
        self.tip_particles = rospy.get_param("~tip_particles", [0,39])
        
        # Offset distance from the obstacles
        self.d_obstacle_offset = rospy.get_param("~d_obstacle_offset", 0.05)

        # Obstacle avoidance free zone distance
        # further than this distance, no obstacles considered by the controller 
        self.d_obstacle_freezone = rospy.get_param("~d_obstacle_freezone", 2.0)

        # Obstacle avoidance performance record variables
        self.overall_min_distance_collision = float('inf')

        # Obstacle avoidance alpha(h_obstacle) function coefficients 
        self.c1_alpha_obstacle = rospy.get_param("~c1_alpha_obstacle", 0.05)
        self.c2_alpha_obstacle = rospy.get_param("~c2_alpha_obstacle", 2.0)
        self.c3_alpha_obstacle = rospy.get_param("~c3_alpha_obstacle", 2.0)

        # Safe wrench values for the robots, assumed to be fixed and the same everywhere. 
        # TODO: Make it variable based on the robot kinematics and dynamics in the future.
        self.wrench_max = np.array(rospy.get_param("~wrench_max", [200.0, 200.0, 200.0, 15.0, 15.0, 15.0]))

        # stress offset values for each axis [Fx,Fy,Fz, Tx,Ty,Tz]
        self.w_stress_offset = np.array(rospy.get_param("~w_stress_offset", [30.0, 30.0, 30.0, 2.25, 2.25, 2.25])) 

        # alpha(h_ft) function robot stress coefficients for each axis [Fx,Fy,Fz, Tx,Ty,Tz]
        self.c1_alpha_ft = np.array(rospy.get_param("~c1_alpha_ft", [1.0, 1.0, 1.0, 0.5, 0.5, 0.5]))
        self.c2_alpha_ft = np.array(rospy.get_param("~c2_alpha_ft", [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]))
        self.c3_alpha_ft = np.array(rospy.get_param("~c3_alpha_ft", [0.8, 0.8, 0.8, 1.0, 1.0, 1.0]))

        # stress avoidance performance record variables
        self.stress_avoidance_performance_avr = 0.0
        self.stress_avoidance_performance_sum = 0.0
        self.stress_avoidance_performance_ever_zero = False

        # Grommet poses as targets for the tip points
        # Each element holds [[x,y,z],[Rx,Ry,Rz(euler angles in degrees)]]
        target_poses_basic =  rospy.get_param("~target_poses", [[[-1, -1, 0.14], [5,   0, 90]], \
                                                                [[ 1, -1, 0.14], [175, 0, 90]]])
        
        self.target_poses = {} # Each item will be a Pose() msg class
        for i, particle in enumerate(self.tip_particles):
            self.target_poses[particle] = self.calculate_target_pose(target_poses_basic[i])

        ## ----------------------------------------------------------------------------------------
        ## SETUP FOR DEFORMABLE OBJECT STATE READINGS FROM SIMULATION PERTURBATIONS

        self.deformable_object_state_topic_name = rospy.get_param("/dlo_state_topic_name") # subscribed
        # this is also like prefix to the perturbed particles' new states

        # Dictionaries that will hold the state of the custom_static_particles and tip_particles
        self.particle_positions = {}
        self.particle_orientations = {}
        self.particle_twists = {}
        self.particle_wrenches = {}

        # Subscriber for deformable object states to figure out the current particle positions
        self.sub_state = rospy.Subscriber(self.deformable_object_state_topic_name, SegmentStateArray, self.state_array_callback, queue_size=10)
        
        # Subscribers to the particle states with perturbations      
        ## We create 6 subscribers for perturbed states of each custom static particle
        self.subs_state_dx = {}
        self.subs_state_dy = {}
        self.subs_state_dz = {}
        self.subs_state_dth_x = {}
        self.subs_state_dth_y = {}
        self.subs_state_dth_z = {}

        # Dictionaries to store perturbed states of the custom static particles
        self.particle_positions_dx = {}
        self.particle_orientations_dx = {}
        self.particle_twists_dx = {}
        self.particle_wrenches_dx = {}

        self.particle_positions_dy = {}
        self.particle_orientations_dy = {}
        self.particle_twists_dy = {}
        self.particle_wrenches_dy = {}

        self.particle_positions_dz = {}
        self.particle_orientations_dz = {}
        self.particle_twists_dz = {}
        self.particle_wrenches_dz = {}

        self.particle_positions_dth_x = {}
        self.particle_orientations_dth_x = {}
        self.particle_twists_dth_x = {}
        self.particle_wrenches_dth_x = {}

        self.particle_positions_dth_y = {}
        self.particle_orientations_dth_y = {}
        self.particle_twists_dth_y = {}
        self.particle_wrenches_dth_y = {}

        self.particle_positions_dth_z = {}
        self.particle_orientations_dth_z = {}
        self.particle_twists_dth_z = {}
        self.particle_wrenches_dth_z = {}

        self.states_set_particles = [] # For bookkeeping of which custom static particles are obtained their all perturbed state readings at least once.

        ## Create the subscribers to states with perturbations
        for particle in self.custom_static_particles:
            # Prepare the topic names of that particle
            state_dx_topic_name    = self.deformable_object_state_topic_name + "_" + str(particle) + "_x" 
            state_dy_topic_name    = self.deformable_object_state_topic_name + "_" + str(particle) + "_y" 
            state_dz_topic_name    = self.deformable_object_state_topic_name + "_" + str(particle) + "_z" 
            state_dth_x_topic_name = self.deformable_object_state_topic_name + "_" + str(particle) + "_th_x" 
            state_dth_y_topic_name = self.deformable_object_state_topic_name + "_" + str(particle) + "_th_y" 
            state_dth_z_topic_name = self.deformable_object_state_topic_name + "_" + str(particle) + "_th_z" 

            # Create the subscribers (also takes the particle argument)
            self.subs_state_dx[particle]    = rospy.Subscriber(state_dx_topic_name,    SegmentStateArray, self.state_array_dx_callback,    particle, queue_size=10)
            self.subs_state_dy[particle]    = rospy.Subscriber(state_dy_topic_name,    SegmentStateArray, self.state_array_dy_callback,    particle, queue_size=10)
            self.subs_state_dz[particle]    = rospy.Subscriber(state_dz_topic_name,    SegmentStateArray, self.state_array_dz_callback,    particle, queue_size=10)
            self.subs_state_dth_x[particle] = rospy.Subscriber(state_dth_x_topic_name, SegmentStateArray, self.state_array_dth_x_callback, particle, queue_size=10)
            self.subs_state_dth_y[particle] = rospy.Subscriber(state_dth_y_topic_name, SegmentStateArray, self.state_array_dth_y_callback, particle, queue_size=10)
            self.subs_state_dth_z[particle] = rospy.Subscriber(state_dth_z_topic_name, SegmentStateArray, self.state_array_dth_z_callback, particle, queue_size=10)
        
        ## ----------------------------------------------------------------------------------------
        ## SETUP FOR MINIMUM DISTANCE READINGS FROM SIMULATION PERTURBATIONS
        self.min_distance_topic_name = rospy.get_param("/min_dist_to_rb_topic_name") # subscribed, 
        # this is also like prefix to the perturbed particles' new minimum distances

        # Subscriber to figure out the current deformable object minimum distances to the rigid bodies in the scene 
        self.sub_min_distance = rospy.Subscriber(self.min_distance_topic_name, MinDistanceDataArray, self.min_distances_array_callback, queue_size=10)

        # Subscribers to the particle minimum distances with perturbations      
        ## We create 6 subscribers for perturbed states of each custom static particle
        self.subs_min_distance_dx = {}
        self.subs_min_distance_dy = {}
        self.subs_min_distance_dz = {}
        self.subs_min_distance_dth_x = {}
        self.subs_min_distance_dth_y = {}
        self.subs_min_distance_dth_z = {}

        # Dictionaries to store minimum distances caused by the perturbation on the custom static particles
        self.min_distances = {}
        self.min_distances_dx = {}
        self.min_distances_dy = {}
        self.min_distances_dz = {}
        self.min_distances_dth_x = {}
        self.min_distances_dth_y = {}
        self.min_distances_dth_z = {}

        self.min_distances_set_particles = [] # For bookkeeping of which custom static particles are obtained their all perturbed min distance readings at least once.
        self.min_distances_set_particles_obstacles = [] 

        ## Create the subscribers to minimum distances with perturbations
        for particle in self.custom_static_particles:
            # Prepare the topic names of that particle
            min_distance_dx_topic_name    = self.min_distance_topic_name + "_" + str(particle) + "_x" 
            min_distance_dy_topic_name    = self.min_distance_topic_name + "_" + str(particle) + "_y" 
            min_distance_dz_topic_name    = self.min_distance_topic_name + "_" + str(particle) + "_z" 
            min_distance_dth_x_topic_name = self.min_distance_topic_name + "_" + str(particle) + "_th_x" 
            min_distance_dth_y_topic_name = self.min_distance_topic_name + "_" + str(particle) + "_th_y" 
            min_distance_dth_z_topic_name = self.min_distance_topic_name + "_" + str(particle) + "_th_z" 
        
            # Create the subscribers (also takes the particle argument)
            self.subs_min_distance_dx[particle]    = rospy.Subscriber(min_distance_dx_topic_name,    MinDistanceDataArray, self.min_distance_array_dx_callback,    particle, queue_size=10)
            self.subs_min_distance_dy[particle]    = rospy.Subscriber(min_distance_dy_topic_name,    MinDistanceDataArray, self.min_distance_array_dy_callback,    particle, queue_size=10)
            self.subs_min_distance_dz[particle]    = rospy.Subscriber(min_distance_dz_topic_name,    MinDistanceDataArray, self.min_distance_array_dz_callback,    particle, queue_size=10)
            self.subs_min_distance_dth_x[particle] = rospy.Subscriber(min_distance_dth_x_topic_name, MinDistanceDataArray, self.min_distance_array_dth_x_callback, particle, queue_size=10)
            self.subs_min_distance_dth_y[particle] = rospy.Subscriber(min_distance_dth_y_topic_name, MinDistanceDataArray, self.min_distance_array_dth_y_callback, particle, queue_size=10)
            self.subs_min_distance_dth_z[particle] = rospy.Subscriber(min_distance_dth_z_topic_name, MinDistanceDataArray, self.min_distance_array_dth_z_callback, particle, queue_size=10)

        ## ----------------------------------------------------------------------------------------
            
        # Create the (centralized) controller that will publish odom to each follower particle properly        
        # self.nominal_controller = NominalController(self.kp, self.kd, self.pub_rate_odom*2.0)
        
        self.control_outputs = {} 
        for particle in self.custom_static_particles:
            self.control_outputs[particle] = np.zeros(6) # initialization for the velocity command

        # Start the control
        self.calculate_control_timer = rospy.Timer(rospy.Duration(1. / self.pub_rate_odom), self.calculate_control_outputs_timer_callback)
        self.odom_pub_timer          = rospy.Timer(rospy.Duration(1. / self.pub_rate_odom), self.odom_pub_timer_callback)





    def set_enable_controller(self, request):
        self.controller_enabler(request.data, cause="manual")
        return SetBoolResponse(True, 'Successfully set enabled state to {}'.format(request.data))
    
    def controller_enabler(self, enable, cause="manual"):
        self.enabled = enable

        if not self.enabled:
            # Stop the particles
            self.odom_publishers_publish_zero_velocities()

        ## Publish the controller status information
        status_msg = ControllerStatus()
        # Fill in common fields
        status_msg.time = rospy.Time.now()  # Example: current time
        status_msg.cause = cause 
        status_msg.enabled = enable

        if enable:
            self.controller_enabled_time = rospy.Time.now()
            self.controller_enabled_time_str = self.get_system_timestamp()
            status_msg.total_duration = -1.0  # -1.0 means the controller is still enabled
            status_msg.rate = -1.0  # -1.0 means the controller is still enabled
            
            # Reset the performance metrics
            self.controller_itr = 0
            self.stress_avoidance_performance_avr = 0.0
            self.stress_avoidance_performance_sum = 0.0
            self.stress_avoidance_performance_ever_zero = False
            self.overall_min_distance_collision = float('inf')

            status_msg.stress_avoidance_performance_avr = 0.0
            status_msg.min_distance = float('inf')
        else:
            self.controller_disabled_time = rospy.Time.now()
            # Calculate the duration when the controller is disabled
            status_msg.total_duration = (self.controller_disabled_time - self.controller_enabled_time).to_sec()
            status_msg.rate = self.controller_itr / status_msg.total_duration # rate of the controller iterations
            
            status_msg.stress_avoidance_performance_avr = self.stress_avoidance_performance_avr
            status_msg.min_distance = self.overall_min_distance_collision

            # ----------------------------------------------------------------------------------------
            ## Print the performance metrics suitable for a csv file
            rospy.loginfo("Controller performance CSV suitable metrics: ")
            rospy.loginfo("Titles: ft_on, baseline, success, min_distance, rate, duration, stress_avoidance_performance_avr, stress_avoidance_performance_ever_zero, start_time")

            ft_on_str = "1" if self.stress_avoidance_enabled else "0"
            baseline_str = "1" if not self.stress_avoidance_enabled and not self.obstacle_avoidance_enabled else "0"
            success_str = "1" if (cause == "automatic") else "0"
            min_distance_str = str(self.overall_min_distance_collision)
            rate_str = str(status_msg.rate)
            duration_str = str(status_msg.total_duration)
            stress_avoidance_performance_avr_str = str(self.stress_avoidance_performance_avr)
            stress_avoidance_performance_ever_zero_str = "1" if self.stress_avoidance_performance_ever_zero else "0"
            # Create start time string with YYYY-MM-DD-Hour-Minute-Seconds format for example 2024-12-31-17-41-34
            start_time_str = self.controller_enabled_time_str

            csv_line = ",".join([ft_on_str, baseline_str, success_str, min_distance_str, rate_str, duration_str, stress_avoidance_performance_avr_str, stress_avoidance_performance_ever_zero_str, start_time_str])
            # Print the csv line green color if the controller is successful else red color
            if (cause == "automatic"):
                # Print the csv line orange color if stress avoidance performance is ever zero
                if self.stress_avoidance_performance_ever_zero:
                    rospy.loginfo("\033[93m" + csv_line + "\033[0m")
                else:
                    # Print the csv line green color if the controller is successful and stress avoidance performance is never zero
                    rospy.loginfo("\033[92m" + csv_line + "\033[0m")
            else:
                rospy.loginfo("\033[91m" + csv_line + "\033[0m")
            # ----------------------------------------------------------------------------------------
        
        # Publish the status message
        self.info_pub_controller_status.publish(status_msg)
        
    def calculate_target_pose(self, target_pose_basic):
        # target_pose_basic: Holds the target pose as a list formatted as [[x,y,z],[Rx,Ry,Rz(euler angles in degrees)]]
        # This function converts it to a Pose msg

        # Extract position and orientation from target_pose_basic
        position_data, orientation_data = target_pose_basic

        # Convert orientation from Euler angles (degrees) to radians
        orientation_radians = np.deg2rad(orientation_data)

        # Convert orientation from Euler angles to quaternion
        quaternion_orientation = tf_trans.quaternion_from_euler(*orientation_radians)

        # Prepare the pose msg
        target_pose = Pose()
        target_pose.position = Point(*position_data)
        target_pose.orientation = Quaternion(*quaternion_orientation)
        return target_pose

    def state_array_callback(self, states_msg):
        for particle in (self.custom_static_particles + self.tip_particles):
            self.particle_positions[particle] = states_msg.states[particle].pose.position
            self.particle_orientations[particle] = states_msg.states[particle].pose.orientation
            self.particle_twists[particle] = states_msg.states[particle].twist

            wrench_array = self.wrench_to_numpy(states_msg.states[particle].wrench)
            if particle not in self.particle_wrenches:
                self.particle_wrenches[particle] = wrench_array
            else:
                # Apply low-pass filter to the force and torque values
                self.particle_wrenches[particle] = self.k_low_pass_ft*self.particle_wrenches[particle] + (1 - self.k_low_pass_ft)*wrench_array

    def state_array_dx_callback(self, states_msg, perturbed_particle):
        self.update_state_arrays(states_msg, perturbed_particle, self.particle_positions_dx, self.particle_orientations_dx, self.particle_twists_dx, self.particle_wrenches_dx)

    def state_array_dy_callback(self, states_msg, perturbed_particle):
        self.update_state_arrays(states_msg, perturbed_particle, self.particle_positions_dy, self.particle_orientations_dy, self.particle_twists_dy, self.particle_wrenches_dy)

    def state_array_dz_callback(self, states_msg, perturbed_particle):
        self.update_state_arrays(states_msg, perturbed_particle, self.particle_positions_dz, self.particle_orientations_dz, self.particle_twists_dz, self.particle_wrenches_dz)

    def state_array_dth_x_callback(self, states_msg, perturbed_particle):
        self.update_state_arrays(states_msg, perturbed_particle, self.particle_positions_dth_x, self.particle_orientations_dth_x, self.particle_twists_dth_x, self.particle_wrenches_dth_x)

    def state_array_dth_y_callback(self, states_msg, perturbed_particle):
        self.update_state_arrays(states_msg, perturbed_particle, self.particle_positions_dth_y, self.particle_orientations_dth_y, self.particle_twists_dth_y, self.particle_wrenches_dth_y)

    def state_array_dth_z_callback(self, states_msg, perturbed_particle):
        self.update_state_arrays(states_msg, perturbed_particle, self.particle_positions_dth_z, self.particle_orientations_dth_z, self.particle_twists_dth_z, self.particle_wrenches_dth_z)

    def update_state_arrays(self, states_msg, perturbed_particle, positions_dict, orientations_dict, twists_dict, wrenches_dict):
        if not (perturbed_particle in positions_dict):
            positions_dict[perturbed_particle] = {}
            orientations_dict[perturbed_particle] = {}
            twists_dict[perturbed_particle] = {}
            wrenches_dict[perturbed_particle] = {}

        for particle in (self.custom_static_particles + self.tip_particles):
            positions_dict[perturbed_particle][particle] = states_msg.states[particle].pose.position
            orientations_dict[perturbed_particle][particle] = states_msg.states[particle].pose.orientation
            twists_dict[perturbed_particle][particle] = states_msg.states[particle].twist

            wrench_array = self.wrench_to_numpy(states_msg.states[particle].wrench)
            if particle not in wrenches_dict[perturbed_particle]:
                wrenches_dict[perturbed_particle][particle] = wrench_array
            else:
                # Apply low-pass filter to the force and torque values
                wrenches_dict[perturbed_particle][particle] = self.k_low_pass_ft*wrenches_dict[perturbed_particle][particle] + (1 - self.k_low_pass_ft)*wrench_array

    def is_perturbed_states_set_for_particle(self,particle):
        """
        Checks if the perturbed state parameters (dx, dy, dz, dthx, dthy, dthz) are set for a specified particle.
        This function determines if the custom static particle has obtained perturbed states along the x, y, z axes at least once.
        If these states are set, the particle is added to the 'states_set_particles' list,
        and the function returns True if the 'self.particle_positions' of that particle is not None. Otherwise, it returns False.
        """
        if particle in self.states_set_particles:
            return (particle in self.particle_positions)
        else:
            check = ((particle in self.particle_positions_dx) and  
                     (particle in self.particle_positions_dy) and 
                     (particle in self.particle_positions_dz) and
                     (particle in self.particle_positions_dth_x) and  
                     (particle in self.particle_positions_dth_y) and 
                     (particle in self.particle_positions_dth_z))
            if check:
                self.states_set_particles.append(particle)
                return (particle in self.particle_positions)
            else:
                return False

    def calculate_jacobian_tip(self):
        """
        Calculates the Jacobian matrix that defines the relation btw. 
        the robot hold points (custom_static_particles) 6DoF poses and
        the tent pole tip points (tip_particles) 6 DoF poses.
        The result is a 12x12 matrix for 2 tip and 2 holding point poses. 
        """
        J = np.zeros((6*len(self.tip_particles),6*len(self.custom_static_particles)))

        for idx_tip, tip in enumerate(self.tip_particles):
            for idx_particle, particle in enumerate(self.custom_static_particles):
                
                # Do not proceed until the initial values have been set
                if ((not self.is_perturbed_states_set_for_particle(particle))):
                    rospy.logwarn_throttle(1,"[calculate_jacobian_tip func.] particle: " + str(particle) + " state is not published yet or it does not have for perturbed states.")
                    continue

                # calculate the pose differences:
                # btw. the current tip pose and the perturbated tip pose caused by the perturbation of custom_static_particle pose in each direction
                current_pose = Pose()
                perturbed_pose = Pose()

                current_pose.position = self.particle_positions[tip]
                current_pose.orientation = self.particle_orientations[tip]
                
                # dx direction
                perturbed_pose.position = self.particle_positions_dx[particle][tip]
                perturbed_pose.orientation = self.particle_orientations_dx[particle][tip]
                pose_difference = self.calculate_pose_difference(current_pose,perturbed_pose)
                J[ (6*idx_tip) : (6*(idx_tip+1)) , 6*idx_particle+0 ] = pose_difference/self.delta_x

                # dy direction
                perturbed_pose.position = self.particle_positions_dy[particle][tip]
                perturbed_pose.orientation = self.particle_orientations_dy[particle][tip]
                pose_difference = self.calculate_pose_difference(current_pose,perturbed_pose)
                J[ (6*idx_tip) : (6*(idx_tip+1)) , 6*idx_particle+1 ] = pose_difference/self.delta_y

                # dz direction
                perturbed_pose.position = self.particle_positions_dz[particle][tip]
                perturbed_pose.orientation = self.particle_orientations_dz[particle][tip]
                pose_difference = self.calculate_pose_difference(current_pose,perturbed_pose)
                J[ (6*idx_tip) : (6*(idx_tip+1)) , 6*idx_particle+2 ] = pose_difference/self.delta_z

                # dth_x direction
                perturbed_pose.position = self.particle_positions_dth_x[particle][tip]
                perturbed_pose.orientation = self.particle_orientations_dth_x[particle][tip]
                pose_difference = self.calculate_pose_difference(current_pose,perturbed_pose)
                J[ (6*idx_tip) : (6*(idx_tip+1)) , 6*idx_particle+3 ] = pose_difference/self.delta_th_x

                # dy direction
                perturbed_pose.position = self.particle_positions_dth_y[particle][tip]
                perturbed_pose.orientation = self.particle_orientations_dth_y[particle][tip]
                pose_difference = self.calculate_pose_difference(current_pose,perturbed_pose)
                J[ (6*idx_tip) : (6*(idx_tip+1)) , 6*idx_particle+4 ] = pose_difference/self.delta_th_y

                # dz direction
                perturbed_pose.position = self.particle_positions_dth_z[particle][tip]
                perturbed_pose.orientation = self.particle_orientations_dth_z[particle][tip]
                pose_difference = self.calculate_pose_difference(current_pose,perturbed_pose)
                J[ (6*idx_tip) : (6*(idx_tip+1)) , 6*idx_particle+5 ] = pose_difference/self.delta_th_z                
        return J
    
    def calculate_error_tip(self):
        err = np.zeros((6*len(self.tip_particles),1))

        avr_norm_pos_err = 0.0
        avr_norm_ori_err = 0.0

        for idx_tip, tip in enumerate(self.tip_particles):
            # Do not proceed until the initial values have been set
            if not (tip in self.particle_positions):
                rospy.logwarn("Tip particle: " + str(tip) + " state is not obtained yet.")
                continue

            current_pose = Pose()
            current_pose.position = self.particle_positions[tip]
            current_pose.orientation = self.particle_orientations[tip]

            target_pose = self.target_poses[tip]

            err[(6*idx_tip) : (6*(idx_tip+1)), 0] = self.calculate_pose_target_error(current_pose,target_pose)

            avr_norm_pos_err += np.linalg.norm(err[(6*idx_tip) : (6*(idx_tip+1)), 0][0:3])/len(self.tip_particles)
            avr_norm_ori_err += np.linalg.norm(err[(6*idx_tip) : (6*(idx_tip+1)), 0][3:6])/len(self.tip_particles)

        return err, avr_norm_pos_err, avr_norm_ori_err

    def min_distances_array_callback(self, min_distances_msg):
        # Create a set to track the IDs in the current message
        current_ids = set()

        # Iterate over the incoming message and update the dictionary
        for data in min_distances_msg.data:
            rigid_body_index = data.index2

            current_ids.add(rigid_body_index)
            if rigid_body_index not in self.min_distances or self.min_distances[rigid_body_index] == float('inf'):
                # Initialize new ID or update 'inf' value with current minDistance
                self.min_distances[rigid_body_index] = data.minDistance
            else:
                # Apply low pass filter for existing ID
                self.min_distances[rigid_body_index] = self.k_low_pass_min_d*self.min_distances[rigid_body_index] + (1-self.k_low_pass_min_d)*data.minDistance

        # Set values to float('inf') for IDs not in current message
        for key in list(self.min_distances.keys()):
            if key not in current_ids:
                self.min_distances[key] = float('inf')
   
    def min_distance_array_dx_callback(self, min_distances_msg, perturbed_particle):
        self.update_min_distances(self.min_distances_dx, min_distances_msg, perturbed_particle)

    def min_distance_array_dy_callback(self, min_distances_msg, perturbed_particle):
        self.update_min_distances(self.min_distances_dy, min_distances_msg, perturbed_particle)

    def min_distance_array_dz_callback(self, min_distances_msg, perturbed_particle):
        self.update_min_distances(self.min_distances_dz, min_distances_msg, perturbed_particle)

    def min_distance_array_dth_x_callback(self, min_distances_msg, perturbed_particle):
        self.update_min_distances(self.min_distances_dth_x, min_distances_msg, perturbed_particle)

    def min_distance_array_dth_y_callback(self, min_distances_msg, perturbed_particle):
        self.update_min_distances(self.min_distances_dth_y, min_distances_msg, perturbed_particle)

    def min_distance_array_dth_z_callback(self, min_distances_msg, perturbed_particle):
        self.update_min_distances(self.min_distances_dth_z, min_distances_msg, perturbed_particle)

    def update_min_distances(self, min_distances, min_distances_msg, perturbed_particle):
        if perturbed_particle not in min_distances:
            min_distances[perturbed_particle] = {}

        # Create a set to track the IDs in the current message
        current_ids = set()

        # Iterate over the incoming message and update the dictionary
        for data in min_distances_msg.data:
            rigid_body_index = data.index2

            current_ids.add(rigid_body_index)
            if rigid_body_index not in min_distances[perturbed_particle] or min_distances[perturbed_particle][rigid_body_index] == float('inf'):
                # Initialize new ID or update 'inf' value with current minDistance
                min_distances[perturbed_particle][rigid_body_index] = data.minDistance
            else:
                # Apply low pass filter for existing ID
                min_distances[perturbed_particle][rigid_body_index] = self.k_low_pass_min_d*min_distances[perturbed_particle][rigid_body_index] + (1 - self.k_low_pass_min_d)*data.minDistance

        # Set values to float('inf') for IDs not in current message
        for key in list(min_distances[perturbed_particle].keys()):
            if key not in current_ids:
                min_distances[perturbed_particle][key] = float('inf')

    def is_perturbed_min_distances_set(self, particle):
        if particle in self.min_distances_set_particles:
            return True
        else:
            check = ((particle in self.min_distances_dx) and  
                     (particle in self.min_distances_dy) and 
                     (particle in self.min_distances_dz) and
                     (particle in self.min_distances_dth_x) and  
                     (particle in self.min_distances_dth_y) and 
                     (particle in self.min_distances_dth_z))
            if check:
                self.min_distances_set_particles.append(particle)
                return True
            else:
                return False
            
    def is_perturbed_min_distances_set_for_obstacle_id(self, particle, key):
        if not self.is_perturbed_min_distances_set(particle):
            return False

        # key here is the rigid body index of the obstacle
        if (particle, key) in self.min_distances_set_particles_obstacles:
            return True
        else:
            check = ((key in self.min_distances_dx[particle]) and  
                     (key in self.min_distances_dy[particle]) and 
                     (key in self.min_distances_dz[particle]) and
                     (key in self.min_distances_dth_x[particle]) and  
                     (key in self.min_distances_dth_y[particle]) and 
                     (key in self.min_distances_dth_z[particle]))
            if check:
                self.min_distances_set_particles_obstacles.append((particle, key))
                return True
            else:
                return False

    def calculate_jacobian_obstacle_min_distance(self, key):
        """
        Calculates the Jacobian matrix that defines the relation btw. 
        the robot hold points (custom_static_particles) 6DoF poses and
        the minimum distance to the obstacles. 
        The result is a row matrix(vector) (e.g. dimension 1x12 for two holding point poses)
        """
        J = np.zeros((1,6*len(self.custom_static_particles)))

        for idx_particle, particle in enumerate(self.custom_static_particles):
            # Do not proceed if the minimum distance is not set for the obstacle
            if not self.is_perturbed_min_distances_set_for_obstacle_id(particle, key):
                rospy.logwarn_throttle(1,"[calculate_jacobian_obstacle_min_distance func.] particle: "+str(particle)+", obstacle index: "+str(key)\
                              +", min distances are not published yet or it does not have for perturbed states")
                continue

            # dx direction
            J[0, 6*idx_particle+0] = (self.min_distances_dx[particle][key] - self.min_distances[key])/self.delta_x if self.delta_x != 0.0 else 0.0
            # dy direction
            J[0, 6*idx_particle+1] = (self.min_distances_dy[particle][key] - self.min_distances[key])/self.delta_y if self.delta_y != 0.0 else 0.0
            # dz direction
            J[0, 6*idx_particle+2] = (self.min_distances_dz[particle][key] - self.min_distances[key])/self.delta_z if self.delta_z != 0.0 else 0.0
            # dth_x direction
            J[0, 6*idx_particle+3] = (self.min_distances_dth_x[particle][key] - self.min_distances[key])/self.delta_th_x if self.delta_th_x != 0.0 else 0.0
            # dy direction
            J[0, 6*idx_particle+4] = (self.min_distances_dth_y[particle][key] - self.min_distances[key])/self.delta_th_y if self.delta_th_y != 0.0 else 0.0
            # dz direction
            J[0, 6*idx_particle+5] = (self.min_distances_dth_z[particle][key] - self.min_distances[key])/self.delta_th_z if self.delta_th_z != 0.0 else 0.0

        # Replace non-finite elements with 0
        J[~np.isfinite(J)] = 0

        return J

    def calculate_jacobian_ft(self):
        """
        Calculates the Jacobian matrix that defines the relation btw.
        the robot hold points (custom_static_particles) 6DoF poses and
        the forces and torques applied by the robots.
        The result is a 12x12 matrix for 2 holding point poses.
        """
        J = np.zeros((6*len(self.custom_static_particles),6*len(self.custom_static_particles)))

        for idx_particle1, particle1 in enumerate(self.custom_static_particles):
            for idx_particle, particle in enumerate(self.custom_static_particles):
                # Do not proceed until the initial values have been set
                if ((not self.is_perturbed_states_set_for_particle(particle))):
                    rospy.logwarn_throttle(1,"[calculate_jacobian_ft func.] particle: " + str(particle) + " state is not published yet or it does not have for perturbed states.")
                    continue

                # calculate the wrench differences:
                # btw. the current holding point wrench and the perturbated holding point wrench caused by the perturbation of custom_static_particle pose in each direction

                # Note that the forces and torques are negated to calculate the wrenches needed to be applied by the robots.
                current_wrench = self.particle_wrenches[particle1]
                
                # dx direction
                perturbed_wrench = self.particle_wrenches_dx[particle][particle1]
                J[ (6*idx_particle1) : (6*(idx_particle1+1)) , 6*idx_particle+0 ] = (perturbed_wrench-current_wrench)/self.delta_x
                # dy direction
                perturbed_wrench = self.particle_wrenches_dy[particle][particle1]
                J[ (6*idx_particle1) : (6*(idx_particle1+1)) , 6*idx_particle+1 ] = (perturbed_wrench-current_wrench)/self.delta_y
                # dz direction
                perturbed_wrench = self.particle_wrenches_dz[particle][particle1]
                J[ (6*idx_particle1) : (6*(idx_particle1+1)) , 6*idx_particle+2 ] = (perturbed_wrench-current_wrench)/self.delta_z
                # dth_x direction
                perturbed_wrench = self.particle_wrenches_dth_x[particle][particle1]
                J[ (6*idx_particle1) : (6*(idx_particle1+1)) , 6*idx_particle+3 ] = (perturbed_wrench-current_wrench)/self.delta_th_x
                # dy direction
                perturbed_wrench = self.particle_wrenches_dth_y[particle][particle1]
                J[ (6*idx_particle1) : (6*(idx_particle1+1)) , 6*idx_particle+4 ] = (perturbed_wrench-current_wrench)/self.delta_th_y
                # dz direction
                perturbed_wrench = self.particle_wrenches_dth_z[particle][particle1]
                J[ (6*idx_particle1) : (6*(idx_particle1+1)) , 6*idx_particle+5 ] = (perturbed_wrench-current_wrench)/self.delta_th_z            

        return J

    def calculate_safe_control_output(self, nominal_u):

        ## ---------------------------------------------------
        ## Define optimization variables
        u = cp.Variable(6*len(self.custom_static_particles))

        ## Define weights for control inputs
        weights = np.ones(6*len(self.custom_static_particles))
        # Assign less weight on z axis positional motions (i.e make them more dynamic)
        # weights[0::6] = 0.5 # Note that x axis position is every 1st element of each 6 element sets in the weight vector
        # weights[1::6] = 0.5 # Note that y axis position is every 2nd element of each 6 element sets in the weight vector
        weights[2::6] = 0.1 # Note that z axis position is every 3rd element of each 6 element sets in the weight vector
        # weights[3::6] = 0.8 # Note that x axis rotation is every 4th element of each 6 element sets in the weight vector
        # weights[4::6] = 0.8 # Note that y axis rotation is every 5th element of each 6 element sets in the weight vector
        # weights[5::6] = 0.8 # Note that z axis rotation is every 6th element of each 6 element sets in the weight vector

        # Define cost function with weights
        cost = cp.sum_squares(cp.multiply(weights, u - nominal_u)) / 2.0

        # Initialize the constraints
        constraints = []
        ## ---------------------------------------------------

        ## ---------------------------------------------------
        # DEFINE COLLISION AVOIDANCE CONTROL BARRIER CONSTRAINTS FOR EACH SCENE MINIMUM DISTANCE READINGS
        
        overall_min_distance = float('inf')
        for key in list(self.min_distances.keys()):
            # key here is the rigid body index of the obstacle

            # update the overall minimum distance
            if self.min_distances[key] < overall_min_distance:
                overall_min_distance = self.min_distances[key]

            if self.obstacle_avoidance_enabled:
                h = self.min_distances[key] - self.d_obstacle_offset # Control Barrier Function (CBF)
                alpha_h = self.alpha_collision_avoidance(h)
                
                # Calculate the obstacle minimum distance Jacobian 
                J_obs_min_dist = self.calculate_jacobian_obstacle_min_distance(key) # 1x12

                # pretty_print_array(J_obs_min_dist, precision=4)
                # print("---------------------------")
                
                # # publish J for information
                # J_msg = Float32MultiArray(data=np.ravel(J_obs_min_dist))
                # self.info_J_publisher.publish(J_msg)

                J_tolerance = 0.001
                if np.any(np.abs(J_obs_min_dist) >= J_tolerance):
                        # Add collision avoidance to the constraints
                        constraints += [J_obs_min_dist @ u >= -alpha_h]
                else:
                    pass
                    # pretty_print_array(J_obs_min_dist, precision=4)
                    # rospy.logwarn_throttle(1,"For obstacle index: " + str(key) + ", ignored J_obs_min_dist and obstacle constraint is not added")
        
        # publish the current overall minimum distance to collision for information
        self.info_pub_overall_min_distance_collision.publish(Float32(data=overall_min_distance))
        
        # Update the controller's last enabled period overall minimum distance for the performance monitoring
        if overall_min_distance < self.overall_min_distance_collision:
            self.overall_min_distance_collision = overall_min_distance

        ## ---------------------------------------------------
                
        ## ---------------------------------------------------
        # DEFINE STRESS CONTROL BARRIER CONSTRAINTS 
        
        # Initialize the variables for the stress avoidance
        h_ft = np.zeros(6*len(self.custom_static_particles)) # Control Barrier Function (CBF) for the forces and torques
        h_ft_normalized = np.zeros(6*len(self.custom_static_particles)) # CBF normalized values for performance monitoring
        alpha_h_ft = np.zeros(6*len(self.custom_static_particles)) # alpha for the forces and torques
        sign_ft = np.zeros(6*len(self.custom_static_particles)) # sign for the forces and torques

        # Calculate the stress avoidance constraints for each custom static particle
        for idx_particle, particle in enumerate(self.custom_static_particles):
            h_ft[6*idx_particle:6*(idx_particle+1)] = self.wrench_max - self.w_stress_offset - np.abs(self.particle_wrenches[particle]) # h_ft = wrench_max - wrench_offset - |wrench|
            h_ft_normalized[6*idx_particle:6*(idx_particle+1)] = (h_ft[6*idx_particle:6*(idx_particle+1)]+self.w_stress_offset)/self.wrench_max
            alpha_h_ft[6*idx_particle:6*(idx_particle+1)] = self.alpha_robot_stress(h_ft[6*idx_particle:6*(idx_particle+1)])
            sign_ft[6*idx_particle:6*(idx_particle+1)] = np.sign(self.particle_wrenches[particle])

        # Calculate stress avoidance performance monitoring values
        self.calculate_and_publish_stress_avoidance_performance(h_ft_normalized)

        if self.stress_avoidance_enabled:
            # Calculate the forces and torques Jacobian
            J_ft = self.calculate_jacobian_ft() # 12x12
            # pretty_print_array(J_ft, precision=2)
            # print("---------------------------")

            # Mutiply the sign of the wrenches elementwise 
            # with the matrix multiplication of the Jacobian with the control input
            # to obtain the forces and torques.
            # Add stress avoidance to the constraints
            constraints += [cp.multiply(-sign_ft, (J_ft @ u)) >= -alpha_h_ft]
        ## ---------------------------------------------------
            
        ## ---------------------------------------------------
        ## Add also limit to the feasible u
        
        # With using the same limits to both linear and angular velocities
        # u_max = 0.1
        # constraints += [cp.norm(u,'inf') <= u_max] # If inf-norm used, the constraint is linear, use OSQP solver (still QP)
        # # constraints += [cp.norm(u,2)     <= u_max] # If 2-norm is used, the constraint is Quadratic, use CLARABEL or ECOS solver (Not QP anymore, a conic solver is needed)

        # With using the different limits to linear and angular velocities
        u_linear_max = self.max_linear_velocity*3.0 # 0.3 # 0.1
        u_angular_max = self.max_angular_velocity*3.0 # 0.5 # 0.15

        # The following slices select every first 3 elements of each group of 6 in the array u.
        linear_indices = np.concatenate([np.arange(i, i+3) for i in range(0, 6*len(self.custom_static_particles), 6)])
        # Apply constraint to linear components
        constraints += [cp.norm(u[linear_indices],'inf') <= u_linear_max]

        # The following slices select every second 3 elements of each group of 6 in the array u.
        angular_indices = np.concatenate([np.arange(i+3, i+6) for i in range(0, 6*len(self.custom_static_particles), 6)])
        # Apply constraint to angular components
        constraints += [cp.norm(u[angular_indices],'inf') <= u_angular_max]

        ## ---------------------------------------------------

        ## ---------------------------------------------------
        # Define and solve problem
        problem = cp.Problem(cp.Minimize(cost), constraints)

        # # For warm-start
        # if hasattr(self, 'prev_optimal_u'):
        #     u.value = self.prev_optimal_u

        # init_t = time.time() # For timing
        try:
            # problem.solve() # Selects automatically
            problem.solve(solver=cp.CLARABEL) #  
            # problem.solve(solver=cp.CVXOPT) # (warm start capable)
            # problem.solve(solver=cp.ECOS) # 
            # problem.solve(solver=cp.ECOS_BB) # 
            # problem.solve(solver=cp.GLOP) # NOT SUITABLE
            # problem.solve(solver=cp.GLPK) # NOT SUITABLE
            # problem.solve(solver=cp.GUROBI) # 
            # problem.solve(solver=cp.MOSEK) # Encountered unexpected exception importing solver CBC
            # problem.solve(solver=cp.OSQP) #  (default) (warm start capable)
            # problem.solve(solver=cp.PDLP) # NOT SUITABLE
            # problem.solve(solver=cp.SCIPY) # NOT SUITABLE 
            # problem.solve(solver=cp.SCS) # (warm start capable)

        except cp.error.SolverError as e:
            rospy.logwarn("QP solver Could not solve the problem: {}".format(e))
            # self.prev_optimal_u = None # to reset warm-start
            return None

        # rospy.logwarn("QP solver calculation time: " + str(1000*(time.time() - init_t)) + " ms.") # For timing

        # Print the available qp solvers
        # e.g. ['CLARABEL', 'CVXOPT', 'ECOS', 'ECOS_BB', 'GLOP', 'GLPK', 'GLPK_MI', 'GUROBI', 'MOSEK', 'OSQP', 'PDLP', 'SCIPY', 'SCS']
        rospy.loginfo_once("Installed solvers:" + str(cp.installed_solvers()))
        # Print the solver used to solve the problem (once)
        rospy.loginfo_once("Solver used: " + str(problem.solver_stats.solver_name))

        # check if problem is infeasible or unbounded
        if problem.status in ["infeasible", "unbounded"]:
            rospy.logwarn("QP solver, The problem is {}.".format(problem.status))
            # self.prev_optimal_u = None # to reset warm-start
            return None
        
        # # For warm-start in the next iteration
        # self.prev_optimal_u = u.value
        
        # Return optimal u
        return u.value

    def calculate_control_outputs_timer_callback(self,event):
        # Only publish if enabled
        if self.enabled:
            # Increment the controller iteration
            self.controller_itr += 1

            # Calculate the nominal control outputs
            if self.nominal_control_enabled:
                J_tip = self.calculate_jacobian_tip() # 12x12
                # pretty_print_array(J_tip)
                # print("---------------------------")

                err_tip, pos_err_avr_norm, ori_err_avr_norm = self.calculate_error_tip() # 12x1, scalar, scalar
                # pretty_print_array(err_tip)
                # print("---------------------------")

                # publish error norms for information
                self.info_pub_target_pos_error_avr_norm.publish(Float32(data=pos_err_avr_norm))
                self.info_pub_target_ori_error_avr_norm.publish(Float32(data=ori_err_avr_norm))

                # Disable the controller if the error is below the threshold
                if (pos_err_avr_norm < (self.acceptable_pos_err_avr_norm+self.d_obstacle_offset)) and (ori_err_avr_norm < self.acceptable_ori_err_avr_norm):
                    # call set_enable service to disable the controller
                    self.controller_enabler(enable=False, cause="automatic")
                    # Create green colored log info message
                    rospy.loginfo("\033[92m" + "Error norms are below the thresholds. The controller is disabled." + "\033[0m")
                    return
            
                # Calculate the nominal control output
                control_output = np.squeeze(np.dot(np.linalg.pinv(J_tip), err_tip)) # (12,)
                # Apply the proportinal gains
                for idx_particle, particle in enumerate(self.custom_static_particles):
                    # Get nominal control output of that particle
                    control_output[6*idx_particle:6*(idx_particle+1)] = self.kp * control_output[6*idx_particle:6*(idx_particle+1)] # nominal
            else:
                # Set the nominal control output to zero
                control_output = np.zeros(6*len(self.custom_static_particles))

            # print("Nominal control_output")
            # pretty_print_array(control_output)
            # print("---------------------------")

            # init_t = time.time()                
            # Calculate safe control output with obstacle avoidance        
            control_output = self.calculate_safe_control_output(control_output) # safe # (12,)
            # rospy.logwarn("QP solver calculation time: " + str(1000*(time.time() - init_t)) + " ms.")
            
            # Assign the calculated control inputs
            for idx_particle, particle in enumerate(self.custom_static_particles):    
                if control_output is not None:
                    self.control_outputs[particle] = control_output[6*idx_particle:6*(idx_particle+1)]
                    # print("Particle " + str(particle) + " u: " + str(self.control_outputs[particle]))
                else:
                    self.control_outputs[particle] = np.zeros(6) # None

            
            
    def odom_pub_timer_callback(self,event):
        # Only publish if enabled
        if self.enabled:
            for particle in self.custom_static_particles:
                # Do not proceed until the initial values have been set
                if ((not (particle in self.particle_positions)) or \
                    (not self.is_perturbed_states_set_for_particle(particle)) or \
                    (not self.is_perturbed_min_distances_set(particle))):   
                    # print("---------------------------")
                    continue
                    
                if self.control_outputs[particle] is not None:
                    # Prepare Odometry message
                    odom = Odometry()
                    odom.header.stamp = rospy.Time.now()
                    odom.header.frame_id = "map"

                    # dt_check = self.nominal_controllers[particle].get_dt() # 
                    dt = 1./self.pub_rate_odom

                    # Scale down the calculated output if its norm is higher than the specified norm max_u
                    control_outputs_linear = self.scale_down_vector(self.control_outputs[particle][:3], max_u=self.max_linear_velocity)

                    # Control output is the new position
                    odom.pose.pose.position.x =  self.particle_positions[particle].x + control_outputs_linear[0]*dt
                    odom.pose.pose.position.y =  self.particle_positions[particle].y + control_outputs_linear[1]*dt
                    odom.pose.pose.position.z =  self.particle_positions[particle].z + control_outputs_linear[2]*dt

                    # The new orientation
                    axis_angle = np.array(self.control_outputs[particle][3:6])
                    angle = np.linalg.norm(axis_angle)
                    axis = axis_angle / angle if (angle > 1e-9) else np.array([0, 0, 1])
                    
                    # Scale down the angle as omega.
                    angle = self.scale_down_vector(angle, max_u=self.max_angular_velocity) if (angle > 1e-9) else angle

                    # Update axis_angle
                    axis_angle = angle * axis

                    # Convert axis-angle to quaternion
                    delta_orientation = tf_trans.quaternion_about_axis(angle * dt, axis)

                    # Update the current orientation by multiplying with delta orientation
                    current_orientation = [self.particle_orientations[particle].x, 
                                        self.particle_orientations[particle].y, 
                                        self.particle_orientations[particle].z, 
                                        self.particle_orientations[particle].w]
                    
                    new_orientation = tf_trans.quaternion_multiply(delta_orientation, current_orientation)

                    # Assign the new orientation to the Odometry message
                    odom.pose.pose.orientation.x = new_orientation[0]
                    odom.pose.pose.orientation.y = new_orientation[1]
                    odom.pose.pose.orientation.z = new_orientation[2]
                    odom.pose.pose.orientation.w = new_orientation[3]

                    # Assign the linear and angular velocities to the Odometry message
                    odom.twist.twist.linear.x = control_outputs_linear[0]
                    odom.twist.twist.linear.y = control_outputs_linear[1]
                    odom.twist.twist.linear.z = control_outputs_linear[2]
                    odom.twist.twist.angular.x = axis_angle[0]
                    odom.twist.twist.angular.y = axis_angle[1]
                    odom.twist.twist.angular.z = axis_angle[2]

                    # Update the pose of the particle 
                    # self.particle_positions[particle] = odom.pose.pose.position
                    # self.particle_orientations[particle] = odom.pose.pose.orientation

                    # Publish
                    self.odom_publishers[particle].publish(odom)
                else:
                    self.control_outputs[particle] = np.zeros(6)

    def odom_publishers_publish_zero_velocities(self):
        """
        This is a helper function that can be used to publish zero velocities for all particles.
        Publishes zero velocities for all particles.
        Useful for stopping the particles when the controller is disabled.
        """
        for particle in self.custom_static_particles:
            # Set the control outputs to zero
            self.control_outputs[particle] = np.zeros(6)

            # Prepare Odometry message
            odom = Odometry()
            odom.header.stamp = rospy.Time.now()
            odom.header.frame_id = "map"

            # Assign the current position and orientation
            odom.pose.pose.position = self.particle_positions[particle]
            odom.pose.pose.orientation = self.particle_orientations[particle]

            # Assign zero velocities
            odom.twist.twist.linear.x = 0.0
            odom.twist.twist.linear.y = 0.0
            odom.twist.twist.linear.z = 0.0
            odom.twist.twist.angular.x = 0.0
            odom.twist.twist.angular.y = 0.0
            odom.twist.twist.angular.z = 0.0

            # Publish
            self.odom_publishers[particle].publish(odom)

    def wrench_to_numpy(self, wrench):
        """
        Converts a ROS wrench message to a numpy array.

        :param wrench: The wrench (force and torque) in ROS message format.
        :return: A numpy array representing the wrench.
        """
        # Combine force and torque arrays
        return np.array([wrench.force.x, wrench.force.y, wrench.force.z, wrench.torque.x, wrench.torque.y, wrench.torque.z])
    
    def calculate_pose_difference(self, current_pose, perturbed_pose):
        """ 
        Calculates the pose difference between the current pose and the perturbed pose.
        The input poses are given as ROS pose msgs.
        """
        # Position difference
        diff_x = perturbed_pose.position.x - current_pose.position.x
        diff_y = perturbed_pose.position.y - current_pose.position.y
        diff_z = perturbed_pose.position.z - current_pose.position.z

        # Orientation difference
        current_orientation = [current_pose.orientation.x, current_pose.orientation.y, current_pose.orientation.z, current_pose.orientation.w]
        perturbed_orientation = [perturbed_pose.orientation.x, perturbed_pose.orientation.y, perturbed_pose.orientation.z, perturbed_pose.orientation.w]

        # Relative rotation quaternion from current to perturbed
        quaternion_difference = tf_trans.quaternion_multiply(tf_trans.quaternion_inverse(current_orientation), perturbed_orientation)

        # Normalize the quaternion to avoid numerical issues
        quaternion_difference = self.normalize_quaternion(quaternion_difference)

        # AXIS-ANGLE ORIENTATION ERROR/DIFFERENCE DEFINITION
        # # Convert quaternion difference to rotation vector (axis-angle representation)
        # angle = 2 * np.arccos(quaternion_difference[3])

        # # Adjust angle to be within the range [-π, π]
        # if angle > np.pi:
        #     angle -= 2 * np.pi

        # # Handling small angles with an approximation
        # small_angle_threshold = 1e-3
        # if np.abs(angle) < small_angle_threshold:
        #     # Use small angle approximation
        #     rotation_vector = 2 * np.array(quaternion_difference[:3])
        # else:
        #     # Regular calculation for larger angles
        #     axis = quaternion_difference[:3] / np.sin(angle / 2)
        #     rotation_vector = axis * angle

        # EULER ANGLES POSE ERROR/DIFFERENCE DEFINITION
        # Convert quaternion to Euler angles for a more intuitive error representation
        rotation_vector = tf_trans.euler_from_quaternion(quaternion_difference)

        # return np.array([diff_x, diff_y, diff_z]), rotation_vector
        # Combine position difference and rotation vector into a 6x1 vector
        return np.array([diff_x, diff_y, diff_z] + list(rotation_vector))
    
    def calculate_pose_target_error(self, current_pose, target_pose):
        """ 
        Calculates the pose error between the current pose and the target pose.
        The input poses are given as ROS pose msgs.
        """
        # Position error
        err_x = target_pose.position.x - current_pose.position.x
        err_y = target_pose.position.y - current_pose.position.y
        err_z = target_pose.position.z - current_pose.position.z

        # Orientation error as quaternion
        current_orientation = [current_pose.orientation.x, current_pose.orientation.y, current_pose.orientation.z, current_pose.orientation.w]
        target_orientation = [target_pose.orientation.x, target_pose.orientation.y, target_pose.orientation.z, target_pose.orientation.w]

        # Relative rotation quaternion from current to target
        quaternion_error = tf_trans.quaternion_multiply(tf_trans.quaternion_inverse(current_orientation), target_orientation)

        # Normalize the quaternion to avoid numerical issues
        quaternion_error = self.normalize_quaternion(quaternion_error)

        # AXIS-ANGLE ORIENTATION ERROR/DIFFERENCE DEFINITION
        # # Convert quaternion error to rotation vector (axis-angle representation)
        # angle = 2 * np.arccos(quaternion_error[3])

        # # Adjust angle to be within the range [-π, π]
        # if angle > np.pi:
        #     angle -= 2 * np.pi

        # # Handling small angles with an approximation
        # small_angle_threshold = 1e-3
        # if np.abs(angle) < small_angle_threshold:
        #     # Use small angle approximation
        #     rotation_vector = 2 * np.array(quaternion_error[:3])
        # else:
        #     # Regular calculation for larger angles
        #     axis = quaternion_error[:3] / np.sin(angle / 2)
        #     rotation_vector = axis * angle

        # EULER ANGLES POSE ERROR/DIFFERENCE DEFINITION
        # Convert quaternion to Euler angles for a more intuitive error representation
        rotation_vector = tf_trans.euler_from_quaternion(quaternion_error)

        # return np.array([err_x, err_y, err_z]), rotation_vector
        # Combine position difference and rotation vector into a 6x1 vector
        return np.array([err_x, err_y, err_z] + list(rotation_vector))
    
    def get_system_timestamp(self):
        # Get the current system time
        now = datetime.now()
        # Format the datetime object into a string similar to rosbag filenames
        timestamp = now.strftime("%Y-%m-%d-%H-%M-%S")
        return timestamp

    def multiply_quaternions(self, q1, q2):
        """
        Multiply two quaternions.
        """
        w0, x0, y0, z0 = q1.w, q1.x, q1.y, q1.z
        w1, x1, y1, z1 = q2.w, q2.x, q2.y, q2.z

        w = w0*w1 - x0*x1 - y0*y1 - z0*z1
        x = w0*x1 + x0*w1 + y0*z1 - z0*y1
        y = w0*y1 - x0*z1 + y0*w1 + z0*x1
        z = w0*z1 + x0*y1 - y0*x1 + z0*w1

        return Quaternion(x, y, z, w)

    def inverse_quaternion(self, q):
        """
        Calculate the inverse of a quaternion.
        """
        w, x, y, z = q.w, q.x, q.y, q.z
        norm = w**2 + x**2 + y**2 + z**2

        return Quaternion(-x/norm, -y/norm, -z/norm, w/norm)
    
    def quaternion_to_matrix(self, quat):
        """
        Return 3x3 rotation matrix from quaternion geometry msg.
        """
        # Extract the quaternion components
        q = np.array([quat.x, quat.y, quat.z, quat.w], dtype=np.float64, copy=True)

        n = np.dot(q, q)
        if n < np.finfo(float).eps:
            return np.identity(3)
        
        q *= np.sqrt(2.0 / n)
        q = np.outer(q, q)

        return np.array((
            (1.0-q[1, 1]-q[2, 2],     q[0, 1]-q[2, 3],     q[0, 2]+q[1, 3]),
            (    q[0, 1]+q[2, 3], 1.0-q[0, 0]-q[2, 2],     q[1, 2]-q[0, 3]),
            (    q[0, 2]-q[1, 3],     q[1, 2]+q[0, 3], 1.0-q[0, 0]-q[1, 1])
            ), dtype=np.float64)

    def normalize_quaternion(self, quaternion):
        norm = np.linalg.norm(quaternion)
        if norm == 0:
            raise ValueError("Cannot normalize a quaternion with zero norm.")
        return quaternion / norm

    def scale_down_vector(self, u, max_u):
        norm_u = np.linalg.norm(u)

        if norm_u > max_u:
            u = u / norm_u * max_u

        return u

    def calculate_and_publish_stress_avoidance_performance(self, h_ft_normalized):
        """
        Calculates and publishes the performance of the stress avoidance.
        """

        def f(x, c_p=2.773):
            """
            Function to boost values further away from zero.
            c_p is a constant that determines the steepness of the function, 
            c_p must be positive for boosting.
            """
            return (np.exp(-c_p * x) - 1) / (np.exp(-c_p) - 1)

        # Check if any of the h_ft_normalized values are less than 0
        if np.any(h_ft_normalized < 0.0):
            # Publish the performance as 0 if any of the h_ft_normalized values are less than 0
            performance = 0.0
        else:
            # Multiply all the h_ft_normalized values
            performance = np.prod(h_ft_normalized)
            # Apply the f(x) function to the product
            performance = f(performance)

        # Publish the performance
        self.info_pub_stress_avoidance_performance.publish(Float32(data=performance))

        # Update the performance history
        self.stress_avoidance_performance_sum += performance
        self.stress_avoidance_performance_avr = self.stress_avoidance_performance_sum / self.controller_itr

        if performance <= 0.0:
            self.stress_avoidance_performance_ever_zero = True

        # Publish the average performance
        self.info_pub_stress_avoidance_performance_avr.publish(Float32(data=self.stress_avoidance_performance_avr))

    def alpha_collision_avoidance(self,h):
        """
        Calculates the value of extended_class_K function \alpha(h) for COLLISION AVOIDANCE
        Piecewise Linear function is used when h is less than 0,
        when h is greater or equal to 0 a nonlinear function is used.
        See: https://www.desmos.com/calculator/hc6lc7nzkk for the function visualizations
        """        
        if (h < 0):
            alpha_h = self.c3_alpha_obstacle*h
        elif (h > self.d_obstacle_freezone) or abs(self.d_obstacle_freezone-h) < 1e-6:
            alpha_h = float('inf')
        else:
            alpha_h = (self.c1_alpha_obstacle*h)/(self.d_obstacle_freezone-h)**self.c2_alpha_obstacle
        
        return alpha_h        
    
    def alpha_robot_stress(self, h):
        """
        Calculates the value of extended_class_K function \alpha(h) for ROBOT STRESS.
        Returns 6x1 alpha_h vector.
        Note that h is 6x1 vector and adjusted to be less than or equal to the wrench_max (h_ft = wrench_max - |wrench|).
        Piecewise Linear function is used when h is less than 0,
        when h is greater or equal to 0 a nonlinear function is used.
        See: https://www.desmos.com/calculator/hc6lc7nzkk for the function visualizations
        """
        # Initialize alpha_h with zeros
        alpha_h = np.zeros(h.shape)

        # Boolean masks for the conditions
        condition_positive_or_zero = h >= 0
        condition_negative = h < 0
        condition_close_to_limit = (self.wrench_max - h) < 1e-6  # Add an epsilon threshold to avoid division by zero

        # Default values for when h is close to wrench_max
        alpha_h[condition_close_to_limit] = float('inf')  # Assign a default value or handle appropriately

        # Calculate for h values greater or equal to 0, excluding values close to the limit
        valid_condition_positive_or_zero = condition_positive_or_zero & ~condition_close_to_limit
        alpha_h[valid_condition_positive_or_zero] = (self.c1_alpha_ft[valid_condition_positive_or_zero] * h[valid_condition_positive_or_zero]) / \
                                                    (self.wrench_max[valid_condition_positive_or_zero] - h[valid_condition_positive_or_zero]) ** self.c2_alpha_ft[valid_condition_positive_or_zero]

        # Calculate for h values less than 0
        alpha_h[condition_negative] = self.c3_alpha_ft[condition_negative] * h[condition_negative]

        return alpha_h
    

if __name__ == "__main__":
    rospy.init_node('velocity_controller_node', anonymous=False)

    node = VelocityControllerNode()

    rospy.spin()
