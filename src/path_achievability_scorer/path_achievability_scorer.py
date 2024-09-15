import os
import re
import traceback
import numpy as np
import time
import sys
import csv
from pathlib import Path

try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle

import rospy
import rosbag 

import tf.transformations as tf_trans

from dlo_simulator_stiff_rods.msg import SegmentStateArray
import threading


from geometry_msgs.msg import Twist, Point, PointStamped, Quaternion, Pose, PoseStamped, Wrench, Vector3
from nav_msgs.msg import Odometry, Path

import matplotlib.pyplot as plt
import matplotlib.animation as animation

from dlo_state_approximator import min_dist_to_polyline
from dlo_state_approximator import dlo_fwd_kin

# from .acceleration_time_optimizer import AccelerationTimeOptimizer

from scipy.signal import savgol_filter

# # Set the default DPI for all images
# plt.rcParams['figure.dpi'] = 100  # e.g. 300 dpi
# # Set the default figure size
# # plt.rcParams['figure.figsize'] = [25.6, 19.2]  # e.g. 6x4 inches
# plt.rcParams['figure.figsize'] = [12.8, 9.6]  # e.g. 6x4 inches

def set_axes_equal(ax):
    """
    Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    """

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


class PathAchievabilityScorer:
    def __init__(self):
        self.reset_planned_path_variables()
        
        self.plots_enabled = False
        self.frames = [] # frames for the animation of the pose comparison
        
        # OPTIONAL: Set the scene ID and experiment number
        self.scene_id = None
        self.experiment_number = None 
        
        self.update_dlo_state_lock = threading.Lock()
        
        # wait time for the particles in dynamic simulation to reach the steady state
        # self.wp_wait_time = rospy.get_param("~wp_wait_time", 0.0) # seconds 
        self.wp_wait_time = rospy.get_param("~wp_wait_time", 0.1) # seconds 
        # self.wp_wait_time = rospy.get_param("~wp_wait_time", 0.2) # seconds 
        # self.wp_wait_time = rospy.get_param("~wp_wait_time", 0.01) # seconds 
        
        self.v_max = rospy.get_param("~max_linear_velocity", 0.3) # m/s
        self.omega_max = rospy.get_param("~max_angular_velocity", 0.450) # rad/s
        
        # self.a_max = rospy.get_param("~max_linear_acceleration", 0.5) # m/s^2
        # self.alpha_max = rospy.get_param("~max_angular_acceleration", 0.75) # rad/s^2
        # self.a_max = rospy.get_param("~max_linear_acceleration", 1.0) # m/s^2
        # self.alpha_max = rospy.get_param("~max_angular_acceleration", 1.5) # rad/s^2
        # self.a_max = rospy.get_param("~max_linear_acceleration", 2.0) # m/s^2
        # self.alpha_max = rospy.get_param("~max_angular_acceleration", 3.0) # rad/s^2
        # self.a_max = rospy.get_param("~max_linear_acceleration", 4.0) # m/s^2
        # self.alpha_max = rospy.get_param("~max_angular_acceleration", 6.0) # rad/s^2
        self.a_max = rospy.get_param("~max_linear_acceleration", 10.0) # m/s^2
        self.alpha_max = rospy.get_param("~max_angular_acceleration", 15.0) # rad/s^2
        
        self.custom_static_particles = None
        self.odom_topic_prefix = None
        while (not self.custom_static_particles):
            try:
                self.custom_static_particles = rospy.get_param("/custom_static_particles") # Default static particles 
                # self.custom_static_particles = [34]
                self.odom_topic_prefix = rospy.get_param("/custom_static_particles_odom_topic_prefix") # published
            except:
                rospy.logwarn("No particles obtained from ROS parameters!.")
                time.sleep(0.5)
        
        # self.pub_rate_odom = rospy.get_param("~pub_rate_odom", 50)
        
        # Create an (odom) Publisher for each static particle (i.e. held particles by the robots) as control output to them.
        self.odom_publishers = {}
        for particle in self.custom_static_particles:
            self.odom_publishers[particle] = rospy.Publisher(self.odom_topic_prefix + str(particle), Odometry, queue_size=10)
            rospy.sleep(0.1)  # Small delay to ensure publishers are fully set up
        
        # Create the planned path publisher
        # self.path_pub = rospy.Publisher("~planned_path", Path, queue_size=10)
        self.path_pub = rospy.Publisher("/tent_building_velocity_controller/planned_path", Path, queue_size=10)
        rospy.sleep(0.1)  # Small delay to ensure publishers are fully set up
            
        # Create the planned path publisher for the particles
        self.path_pub_particles = {}
        for particle in self.custom_static_particles:
            # self.path_pub_particles[particle] = rospy.Publisher("~planned_path_particle_" + str(particle), Path, queue_size=10)
            self.path_pub_particles[particle] = rospy.Publisher("/tent_building_velocity_controller/planned_path_particle_" + str(particle), Path, queue_size=10)    
            rospy.sleep(0.1)  # Small delay to ensure publishers are fully set up
        
        # Subscribed topic name for the deformable object state
        self.deformable_object_state_topic_name = rospy.get_param("/dlo_state_topic_name") # subscribed
        # this is also like prefix to the perturbed particles' new states

        # Dictionaries that will hold the state of the custom_static_particles and tip_particles
        self.particle_positions = {}
        self.particle_orientations = {}
        self.particle_twists = {}
        self.particle_wrenches = {}

        self.current_full_state = None # To store the current full state of the deformable object
        
        # Subscriber for deformable object states to figure out the current particle positions
        self.sub_state = rospy.Subscriber(self.deformable_object_state_topic_name, SegmentStateArray, self.state_array_callback, queue_size=10)
        rospy.sleep(0.1)  # Small delay to ensure publishers are fully set up
    
    def score_path_from_pickle_file(self, path_pickle_file, scene_id = None, experiment_number= None):
        
        # Set the scene ID and experiment number if provided for plotting purposes
        if scene_id:
            self.scene_id = scene_id
        if experiment_number:
            self.experiment_number = experiment_number
        
        # Reset the planned path variables
        self.reset_planned_path_variables()
        # Load the path from the pickle file
        path_variables, planning_success = self.load_path_from_file(path_pickle_file)

        if planning_success:
            # Set the path variables
            self.set_path(path_variables)            
            
            # Score the path
            return self.score_path()
        else:
            # Return None if planning was not successful
            return None # scores returned as None if planning was not successful
        
    def score_path(self):
        errs = []
        peak_err = 0.0 
        peak_err_waypoint_idx = 0
        avr_err = 0.0
        err_changes = [] 
        peak_err_change = 0.0   
        peak_err_change_idx = 0
        avr_err_change = 0.0
        
        print("path number of waypoints: ", len(self.planned_path))

        start_time = time.time()
        while self.planned_path_current_target_index < len(self.planned_path)-1:
        # while self.planned_path_current_target_index < 10:
            self.planned_path_current_target_index += 1
            
            # Update the current target pose
            self.update_current_path_target_poses(self.planned_path_current_target_index)
            
            # Apply the control to the particles
            # self.send_to_target_poses_basic()
            self.send_to_target_poses()
            
            # Wait for the particles to reach the target pose
            rospy.sleep(self.wp_wait_time) # seconds
            
            # Calculate the pose errors btw the particles and the target poses
            err = self.calculate_pose_errors() *1000 # convert to mm
            # print(f"Error at waypoint index {self.planned_path_current_target_index}: {err}")

            # Save the scores to arrays
            # Append the error to the errors array
            errs.append(err)
            
            # Update the peak error and the peak error waypoint index
            if err > peak_err:
                peak_err = err
                peak_err_waypoint_idx = self.planned_path_current_target_index
                
            # Append the error change to the error changes array
            if self.planned_path_current_target_index == 0:
                err_changes.append(0.0)
            
            if self.planned_path_current_target_index > 0:
                err_change = err - errs[self.planned_path_current_target_index - 1]
                err_changes.append(err_change)
                
                # Update the peak error change and the peak error change waypoint index
                if err_change > peak_err_change:
                    peak_err_change = err_change
                    peak_err_change_idx = self.planned_path_current_target_index
                    
        # Calculate the average error
        avr_err = np.mean(errs)
                
        # Calculate the average error change
        avr_err_change = np.mean(err_changes)
                
        scores = (errs,
                peak_err,
                peak_err_waypoint_idx,
                avr_err,
                
                err_changes,
                peak_err_change,
                peak_err_change_idx,
                avr_err_change)
        
        smoothed_scores = self.smooth_scores(errs)
        
        # Append the smoothed scores to the scores
        scores = scores + smoothed_scores
        
        end_time = time.time()
        
        # Calculate the average time taken for the path scoring per waypoint
        avr_time_per_waypoint = (end_time - start_time)/len(self.planned_path)
        # Append the average time taken for the path scoring per waypoint to the scores
        scores = scores + (avr_time_per_waypoint,)
            
        # Reverse the same path to return to the initial position
        rospy.loginfo("Reversing the path to return to the initial position.")
        while self.planned_path_current_target_index > 0:
            self.planned_path_current_target_index -= 1
            
            # Update the current target pose
            self.update_current_path_target_poses(self.planned_path_current_target_index)
            
            # Apply the control to the particles
            self.send_to_target_poses(speedup=50.0)
            
            # # Wait for the particles to reach the target pose
            # rospy.sleep(self.wp_wait_time) # seconds

        
        return scores
    
    def calculate_pose_errors(self):
        if self.update_dlo_state_lock.acquire(blocking=False):
            try:
                # Convert the current full state to a numpy array
                current_full_state_dict = self.parse_state_as_dict(self.current_full_state) 
                current_full_state_np = self.convert_state_dict_to_numpy(current_full_state_dict) # shape: (n, 7), n: number of particles
                
                # calculate the DLO length from the approximated state positions
                initial_approximated_state_pos = self.initial_n_target_states[1]
                dlo_length = self.calculate_dlo_length(initial_approximated_state_pos)
                # print("DLO Length: ", dlo_length)
                
                # Current path index approximated dlo joint values
                joint_values = self.planned_path_approximated_dlo_joint_values[self.planned_path_current_target_index] 
                
                # Calculate the approximated state positions using the forward kinematics
                approximated_pos = dlo_fwd_kin(joint_pos=joint_values, dlo_l=dlo_length) 
                
                # # OPTIONAL: Plot the approximated positions and the original positions for comparison visualization
                if self.plots_enabled:
                    self.plot_dlo_sim_state_vs_rigid_link_apprx_comparison(polyline=approximated_pos, points=current_full_state_np[:,0:3])
                self.frames.append((current_full_state_np[:,0:3], approximated_pos))
                
                particle_errors = min_dist_to_polyline(points=current_full_state_np[:,0:3], polyline=approximated_pos) # errors for each particle
                avg_error = np.mean(particle_errors) # average error of the current path waypoint index
                
                return avg_error
            
            except Exception as e:
                rospy.logerr(f"Error calculating pose errors: {e}")
                rospy.logerr(traceback.format_exc())
                return None
            finally:
                self.update_dlo_state_lock.release()
    
    def send_to_target_poses_basic(self):
        """ This is the basic version of the send_to_target_poses function with odom publishers. 
        
        It only publishes the target poses to the particles without any velocity profile.
        But this causes abrupt changes in the position and orientation of the particles and so
        introduces oscillations in the simulated deformable object.
        This requires a wait time for the particles to reach the steady state before the next waypoint is reached.
        Hence the comparison of the approximated state with the rigid link approximation can be done only after the particles reach the steady state for accurate results in the comparisons of the states.
        This causes more time to be spent for the simulation and the comparison of the states during the achievability scoring.
        """
        for particle in self.custom_static_particles:
            # Do not proceed until the initial values have been set
            if not particle in self.particle_positions:
                continue
                
            # Prepare Odometry message
            odom = Odometry()
            odom.header.stamp = rospy.Time.now()
            odom.header.frame_id = "map"

            # Control output is the new position
            odom.pose.pose = self.planned_path_current_target_poses_of_particles[particle]

            # Publish
            self.odom_publishers[particle].publish(odom)

    # ----------------------------------------------------------------------------------
    def send_to_target_poses(self, speedup=1.0):
        # Gradually change the position and orientation of the particles 
        # to the target pose (self.planned_path_current_target_poses_of_particles[particle]) 
        # based on a trapezoidal velocity profile described with
        # given self.a_max, self.v_max, self.alpha_max, self.omega_max
        
        rate = rospy.Rate(500)  # Control frequency

        # Initialize variables
        max_time = 0
        acc_time = 0 # desired acceleration time
        velocity_profiles = {}

        # Precompute velocity profiles for all particles and find max time
        for particle in self.custom_static_particles:
            if particle not in self.particle_positions:
                continue

            current_position = self.particle_positions[particle]
            current_orientation = self.particle_orientations[particle]  # Assuming quaternion
            target_pose = self.planned_path_current_target_poses_of_particles[particle]
            target_position = target_pose.position
            target_orientation = target_pose.orientation  # Assuming quaternion

            # Compute 3D distance and orientation difference as angle (based on axis-angle representation)
            segment_distance, segment_direction = self.compute_distance(current_position, target_position)  # 3D distance
            segment_rotation, segment_rot_axis = self.compute_orientation_difference_axis_angle(current_orientation, target_orientation)

            # Compute trapezoidal profile for position and orientation
            t_pos, t_pos_acc = self.compute_min_time_needed_with_trapezoidal_profile(segment_distance, self.a_max*speedup, self.v_max*speedup)
            t_ori, t_ori_acc = self.compute_min_time_needed_with_trapezoidal_profile(segment_rotation, self.alpha_max*speedup, self.omega_max*speedup)

            # Store velocity profile components
            velocity_profiles[particle] = {
                't_tot': 0.0,  # Total time (placeholder)
                't_a_p': 0.0,  # position acceleration duration (placeholder), assumed same as deceleration duration
                't_a_o': 0.0,  # orientation acceleration duration (placeholder), assumed same as deceleration duration
                'a_p': 0.0,  # position acceleration (placeholder)
                'a_o': 0.0,  # orientation acceleration (placeholder)
                'd_p': segment_distance,  # 3D distance to the target
                'd_o': segment_rotation,  # Orientation difference (angle in radians)
                'dir_p': segment_direction,  # Unit Vector to target position (zeros vector if d_p=0)
                'dir_o': segment_rot_axis,  # Unit Axis orientation vector (unit vector in z direction if d_o=0)
                'start_p': current_position,  # Starting position (Point)
                'start_o': current_orientation,  # Starting orientation (Quaternion)
                'target_p': target_position,  # Target position (Point)
                'target_o': target_orientation,  # Target orientation (Quaternion)
            }

            # Find the maximum time needed across all particles and corresponding acceleration times
            # max_time = max(max_time, t_pos, t_ori)
            if t_pos > max_time:
                max_time = t_pos
                acc_time = t_pos_acc
            if t_ori > max_time:
                max_time = t_ori
                acc_time = t_ori_acc
                
            # print(velocity_profiles[particle])
            
        # Re-iterate to update the total time needed for each particle
        # and to update the velocity profiles
        for particle, profile in velocity_profiles.items():
            velocity_profiles[particle]['t_tot'] = max_time
            
            # Find the acceleration time for position and orientation based on the max time and the desired acceleration time
            if profile['d_p'] > 0.0:
                a_p, t_a_p = self.compute_fixed_time_trapezoidal_profile_params(max_time, acc_time, profile['d_p'], self.a_max*speedup, self.v_max*speedup)
            else:
                a_p = 0.0
                t_a_p = 0.0
            
            if profile['d_o'] > 0.0:
                a_o, t_a_o = self.compute_fixed_time_trapezoidal_profile_params(max_time, acc_time, profile['d_o'], self.alpha_max*speedup, self.omega_max*speedup)
            else:
                a_o = 0.0
                t_a_o = 0.0
            
            # Update the velocity profiles
            velocity_profiles[particle]['t_a_p'] = t_a_p
            velocity_profiles[particle]['t_a_o'] = t_a_o
            velocity_profiles[particle]['a_p'] = a_p
            velocity_profiles[particle]['a_o'] = a_o
            
        # Main loop to update particles simultaneously
        start_time = rospy.Time.now().to_sec()
        while not rospy.is_shutdown():
            current_time = rospy.Time.now().to_sec()
            elapsed_time = current_time - start_time

            for particle, profile in velocity_profiles.items():
                if particle not in self.particle_positions:
                    continue

                # Update position and orientation based on trapezoidal profile
                new_position = self.compute_position_from_trapezoidal_profile(profile, elapsed_time)
                new_orientation = self.compute_orientation_from_trapezoidal_profile(profile, elapsed_time)

                # Compute twist (linear and angular velocity) for the particle
                linear_velocity = self.compute_linear_velocity_from_profile(profile, elapsed_time)
                angular_velocity = self.compute_angular_velocity_from_profile(profile, elapsed_time)

                # Prepare odometry message
                odom = Odometry()
                odom.header.stamp = rospy.Time.now()
                odom.header.frame_id = "map"
                
                odom.pose.pose.position = new_position
                odom.pose.pose.orientation = new_orientation
                
                odom.twist.twist.linear.x = linear_velocity[0]
                odom.twist.twist.linear.y = linear_velocity[1]
                odom.twist.twist.linear.z = linear_velocity[2]
                odom.twist.twist.angular.x = angular_velocity[0]
                odom.twist.twist.angular.y = angular_velocity[1]
                odom.twist.twist.angular.z = angular_velocity[2]

                # Publish odometry and twist
                self.odom_publishers[particle].publish(odom)

            if elapsed_time > max_time:
                break  # Stop if all particles have completed their movements
        
            rate.sleep()  # Maintain the control loop rate
    
    def compute_distance(self, current_position, target_position):
        # Compute 3D distance between two points
        vec = np.array([target_position.x - current_position.x, target_position.y - current_position.y, target_position.z - current_position.z])
        dist = np.linalg.norm(vec)
        
        if dist > 0.0:
            return dist, vec
        else:
            return 0.0, np.zeros_like(vec)
    
    def compute_orientation_difference_axis_angle(self, current_orientation, target_orientation):
        current_orientation = [current_orientation.x, current_orientation.y, current_orientation.z, current_orientation.w]
        target_orientation = [target_orientation.x, target_orientation.y, target_orientation.z, target_orientation.w]
        
        # Relative rotation quaternion from current to target
        quaternion_error = tf_trans.quaternion_multiply(target_orientation, tf_trans.quaternion_inverse(current_orientation))

        # Normalize the quaternion to avoid numerical issues
        quaternion_error = self.normalize_quaternion(quaternion_error)

        # AXIS-ANGLE ORIENTATION ERROR/DIFFERENCE DEFINITION
        # Convert quaternion difference to rotation vector (axis-angle representation)
        rotation_vector = self.quaternion_to_rotation_vec(quaternion_error)
        
        angle = np.linalg.norm(rotation_vector)
        
        if angle > 0.0:
            axis = rotation_vector / angle
        else:
            axis = np.array([0.0, 0.0, 1.0])
            
        return angle, axis

    def quaternion_to_rotation_vec(self, quaternion):
        """
        Converts a quaternion to axis-angle representation.
        Minimal representation of the orientation error. (3,) vector.
        Arguments:
            quaternion: The quaternion to convert as a numpy array in the form [x, y, z, w].
        Returns:
            rotation_vector: Minimal axis-angle representation of the orientation error. (3,) vector.
            Norm if the axis_angle is the angle of rotation.
        """
        angle = 2 * np.arccos(quaternion[3])

        # Handling small angles with an approximation
        small_angle_threshold = 1e-6
        if np.abs(angle) < small_angle_threshold:
            # Use small angle approximation
            rotation_vector = np.array([0.,0.,0.])
            
            # # Also calculate the representation Jacobian inverse
            # J_inv = np.eye((3,3))
        else:
            # Regular calculation for larger angles
            axis = quaternion[:3] / np.sin(angle/2.0)
            # Normalize the axis
            axis = axis / np.linalg.norm(axis)
            rotation_vector = angle * axis 
            
            # # Also calculate the representation Jacobian inverse
            # k_hat = self.hat(axis)
            # cot = 1.0 / np.tan(angle/2.0)
            # J = -angle/2.0 * (k_hat + cot*(k_hat @ k_hat)) + np.outer(axis, axis)
            # J_inv = np.linalg.inv(J)
            
        return rotation_vector # , J_inv

    def normalize_quaternion(self, quaternion):
        norm = np.linalg.norm(quaternion)
        if norm == 0:
            raise ValueError("Cannot normalize a quaternion with zero norm.")
        return quaternion / norm

    def compute_min_time_needed_with_trapezoidal_profile(self, delta, a_max, v_max):
        # Check if it's a full trapezoidal profile or triangular profile
        if (v_max ** 2) / a_max < delta:  # Full trapezoidal profile
            t_acc = v_max / a_max
            t_const = (delta - (v_max ** 2) / a_max) / v_max
            t_total = 2 * t_acc + t_const
        else:  # Triangular profile (cannot reach max velocity)
            t_acc = np.sqrt(delta / a_max)
            t_total = 2 * t_acc
            t_const = 0

        # Return total needed time and the acceleration time
        return t_total, t_acc

    def compute_fixed_time_trapezoidal_profile_params(self, t_total, t_acc_d, delta, a_max, v_max):
        """ Calculate the trapezoidal profile parameters (acc and t_acc) for a known 
            time duration and distance (delta) of the profile, given the max acceleration and velocity.
            Assuming the acceleration and deceleration times are the same.
            Assuming the acceleration values are fixed with values either acc, -acc, or 0,
            which is needed for Trapezoidal velocity profile.
            
        """
        acc = delta / (t_acc_d*(t_total - t_acc_d))
        
        EPSILON = 1e-9  # Set a small threshold for precision errors

        if acc <= a_max + EPSILON and acc * t_acc_d <= v_max + EPSILON:
            return acc, t_acc_d
        else:
            # Find the closest valid acceleration and acceleration time
            t_acc = self.find_closest_t_acc(x0=t_acc_d, 
                                            a=-t_total, 
                                            b=-delta/a_max, 
                                            c=t_total - delta/v_max, 
                                            d=t_total/2)
            acc = delta / (t_acc*(t_total - t_acc)) 
            return acc, t_acc
            
    def find_closest_t_acc(self, x0, a, b, c, d):
        # Solve the quadratic inequality x^2 + ax <= b
        # Find the roots of x^2 + ax - b = 0
        discriminant = a**2 + 4*b
        
        if discriminant < 0:
            rospy.logwarn(f"Discriminant = {discriminant} < 0. No real solution for the quadratic inequality.")

        root1 = (-a + np.sqrt(discriminant)) / 2
        root2 = (-a - np.sqrt(discriminant)) / 2

        # The solution for x^2 + ax <= b is between the roots root1 and root2
        lower_bound = min(root1, root2)
        upper_bound = max(root1, root2)

        # Now, intersect with other constraints: x <= c, 0 < x <= d
        feasible_lower_bound = max(0, lower_bound)
        feasible_upper_bound = min(upper_bound, c, d)

        # Check if the feasible region is valid
        if feasible_lower_bound >= feasible_upper_bound:
            raise ValueError("No feasible solution due to constraints.")

        # Now find the value of x in the feasible region that is closest to x0
        if x0 < feasible_lower_bound:
            return feasible_lower_bound
        elif x0 > feasible_upper_bound:
            return feasible_upper_bound
        else:
            return x0

    def compute_linear_velocity_from_profile(self, profile, elapsed_time):
        if elapsed_time >= profile['t_tot']:
            return np.zeros(3)  # Stop if time exceeds
        v_norm =  self.v_from_trap_vel_profile(t=elapsed_time, a=profile['a_p'], t_acc=profile['t_a_p'], t_total=profile['t_tot'])
        v_dir = profile['dir_p']
        return v_norm * v_dir

    def compute_angular_velocity_from_profile(self, profile, elapsed_time):
        if elapsed_time >= profile['t_tot']:
            return np.zeros(3)  # Stop if time exceeds
        w_norm = self.v_from_trap_vel_profile(t=elapsed_time, a=profile['a_o'], t_acc=profile['t_a_o'], t_total=profile['t_tot'])
        w_dir = profile['dir_o']
        return w_norm * w_dir
    
    def compute_position_from_trapezoidal_profile(self, profile, elapsed_time):
        if elapsed_time >= profile['t_tot']:
            return profile['target_p']

        start_position = np.array([profile['start_p'].x, profile['start_p'].y, profile['start_p'].z])

        # Compute the distance moved based on the trapezoidal velocity profile
        p = self.p_from_trap_vel_profile(t=elapsed_time, a=profile['a_p'], t_acc=profile['t_a_p'], t_total=profile['t_tot'], delta=profile['d_p'])        
        if profile['d_p'] > 0.0:
            new_position = start_position +  profile['dir_p']*(p/profile['d_p'])
        else:
            new_position = start_position

        return Point(x=new_position[0], y=new_position[1], z=new_position[2])

    def compute_orientation_from_trapezoidal_profile(self, profile, elapsed_time):
        if elapsed_time >= profile['t_tot']:
            return profile['target_o']

        # Compute rotated angle on the trapezoidal velocity profile
        p = self.p_from_trap_vel_profile(elapsed_time, a=profile['a_o'], t_acc=profile['t_a_o'], t_total=profile['t_tot'], delta=profile['d_o'])
        
        # Compute the new orientation
        new_orientation = self.apply_axis_angle_rotation(profile['start_o'], profile['dir_o'], p)
        
        return Quaternion(x=new_orientation[0], y=new_orientation[1], z=new_orientation[2], w=new_orientation[3])

    def p_from_trap_vel_profile(self, t, a, t_acc, t_total, delta):
        if t >= t_total:
            return delta
        
        if t < t_acc:  # Acceleration phase
            return 0.5 * a * t**2
        elif t < t_total - t_acc:  # Constant velocity phase
            return (0.5 * a*t_acc**2) + (a*t_acc) * (t - t_acc)
        else:  # Deceleration phase
            return delta - 0.5*a*(t_total - t)**2

    def v_from_trap_vel_profile(self, t, a, t_acc, t_total):
        # t: current time
        # a: acceleration
        # t_acc: time to reach max velocity
        # t_total: total time of the profile    
        
        if t >= t_total:
            return 0.0
        
        if t < t_acc:  # Acceleration phase
            return a*t
        elif t < t_total - t_acc:  # Constant velocity phase
            return a*t_acc
        else:  # Deceleration phase
            return a*t_acc - a*(t - (t_total-t_acc))    
    
    def apply_axis_angle_rotation(self, start_orientation, axis, angle):
        # Convert the start orientation to a numpy array
        start_quat = [start_orientation.x, start_orientation.y, start_orientation.z, start_orientation.w]

        delta_orientation = tf_trans.quaternion_about_axis(angle, axis)
        
        # Apply the rotation to the start orientation
        new_orientation = tf_trans.quaternion_multiply(delta_orientation, start_quat)
        return new_orientation
    # ----------------------------------------------------------------------------------
    
    def update_current_path_target_poses(self, current_target_index):
        """
        Updates the current target poses and velocities of the particles based on the current target index.
        """
        
        # update the current target pose
        if current_target_index < len(self.planned_path):
            self.planned_path_current_target_pose = self.planned_path[current_target_index].pose

        # update the current target poses of the particles
        for particle, path in self.planned_path_of_particles.items():
            if current_target_index < len(self.planned_path):
                self.planned_path_current_target_poses_of_particles[particle] = path[current_target_index].pose
        
    def load_path_from_file(self, path_file):
        try:
            with open(path_file, 'rb') as inp:
                plan_data = pickle.load(inp)
            
                self.plan_data_ompl = plan_data[0]
                self.plan_data_trajopt = plan_data[1] 
                self.performance_data = plan_data[2]
                self.initial_n_target_states = plan_data[3]
                
                # GUIDE:
                # plan_data_ompl = (ompl_path, ompl_path_points, ompl_path_cumulative_lengths, ompl_path_cumulative_rotations,
                #                 ompl_path_direction_vectors, ompl_path_rotation_vectors, ompl_path_of_particles,
                #                 ompl_path_points_of_particles, ompl_path_cumulative_lengths_of_particles,
                #                 ompl_path_cumulative_rotations_of_particles, ompl_path_direction_vectors_of_particles,
                #                 ompl_path_rotation_vectors_of_particles, ompl_path_approximated_dlo_joint_values)
                # plan_data_trajopt = (trajopt_path, trajopt_path_points, trajopt_path_cumulative_lengths, trajopt_path_cumulative_rotations,
                #                     trajopt_path_direction_vectors, trajopt_path_rotation_vectors, trajopt_path_of_particles,
                #                     trajopt_path_points_of_particles, trajopt_path_cumulative_lengths_of_particles,
                #                     trajopt_path_cumulative_rotations_of_particles, trajopt_path_direction_vectors_of_particles,
                #                     trajopt_path_rotation_vectors_of_particles, trajopt_path_approximated_dlo_joint_values)
                # performance_data = (experiment_id, simplified_dlo_num_segments, avr_state_approx_error, planning_success,
                #                     planning_time_ompl, planning_time_trajopt, total_planning_time, ompl_path_length, trajopt_path_length)
                # initial_n_target_states = (initial_full_state, initial_approximated_state_pos, initial_approximated_state_joint_pos, 
                #                         target_full_state, target_approximated_state_pos, target_approximated_state_joint_pos)
                
                planning_success = self.performance_data[3]

            return self.plan_data_trajopt, planning_success
        except Exception as e:
            rospy.logerr(f"Error loading path from file: {e}")
            return None, False

    def set_path(self, path_variables):
        """
        Sets the path variables.
        """
        rospy.loginfo("Setting the path variables.")
        
        (self.planned_path,
        self.planned_path_points,
        self.planned_path_cumulative_lengths,
        self.planned_path_cumulative_rotations,
        self.planned_path_direction_vectors,
        self.planned_path_rotation_vectors,
        self.planned_path_of_particles,
        self.planned_path_points_of_particles,
        self.planned_path_cumulative_lengths_of_particles,
        self.planned_path_cumulative_rotations_of_particles,
        self.planned_path_direction_vectors_of_particles,
        self.planned_path_rotation_vectors_of_particles, 
        self.planned_path_approximated_dlo_joint_values) = path_variables
        
        # Publish the planned path for visualization
        self.publish_path(self.planned_path)
        
        # Publish the planned path of the particles for visualization
        self.publish_paths_of_particles(self.planned_path_of_particles)
    
    def reset_planned_path_variables(self):
        self.planned_path = [] # (it is a list of PoseStamped() msgs)
        self.planned_path_points = None # path 3D xyz points as numpy array
        self.planned_path_cumulative_lengths = [0.0] # cumulative lengths of the path segments
        self.planned_path_cumulative_rotations = [0.0] # cumulative rotations of the path segments obtained from the angles in axis-angle representation consecutive rotations in radians
        self.planned_path_direction_vectors = [] # directions of the path segments as unit vectors
        self.planned_path_rotation_vectors = [] # rotation axes of the path segments as unit vectors

        self.planned_path_completion_rate = 0.0 # completion rate of the planned path
        self.planned_path_current_target_index = -1 # index of the current waypoint in the planned path
        self.planned_path_current_target_pose = None # current target pose of the planned path as a Pose() msg
                
        self.planned_path_of_particles = {} # planned path of the particles as a list of Pose() msgs
        self.planned_path_points_of_particles = {} # path 3D xyz points of the particles as numpy array
        self.planned_path_cumulative_lengths_of_particles = {} # cumulative lengths of the path segments of the particles
        self.planned_path_cumulative_rotations_of_particles = {} # cumulative rotations of the path segments of the particles obtained from the angles in axis-angle representation consecutive rotations in radians
        self.planned_path_direction_vectors_of_particles = {} # directions of the path segments of the particles as unit vectors (list of n elements with each element is a 3D vector for each particle)
        self.planned_path_rotation_vectors_of_particles = {} # rotation axes of the path segments of the particles as unit vectors (list of n elements with each element is a 3D vector for each particle)

        self.planned_path_current_target_poses_of_particles = {} # current target poses of the particles as Pose() msgs
        
        self.planned_path_velocity_profile_of_particles = {} # velocity profile of the particles
        self.planned_path_current_target_velocities_of_particles = {} # current target velocities of the particles based on the velocity profile
        
    def state_array_callback(self, states_msg):
        if self.update_dlo_state_lock.acquire(blocking=False):
            try:
                self.current_full_state = states_msg
        
                for particle in (self.custom_static_particles):
                    self.particle_positions[particle] = states_msg.states[particle].pose.position
                    self.particle_orientations[particle] = states_msg.states[particle].pose.orientation
                    self.particle_twists[particle] = states_msg.states[particle].twist

                    wrench_array = self.wrench_to_numpy(states_msg.states[particle].wrench)
                    if particle not in self.particle_wrenches:
                        self.particle_wrenches[particle] = wrench_array
                    else:
                        # Apply low-pass filter to the force and torque values (not needed for this code, did not use it)
                        self.particle_wrenches[particle] = wrench_array
                
            except Exception as e:
                rospy.logerr(f"Error updating the DLO state: {e}")
                rospy.logerr(traceback.format_exc())
            finally:
                self.update_dlo_state_lock.release()

    def parse_state_as_dict(self, states_msg):
        # Parse SegmentStateArray from dlo_simulator_stiff_rods.msg as a dictionary
        # with "id,p_x,p_y,p_z,o_x,o_y,o_z,o_w" are the keys
        # where id is the particle id and p_x, p_y, p_z are the position components
        # and o_x, o_y, o_z, o_w are the orientation components
        
        # Initialize an empty dictionary to store data
        data = {'id': [], 'p_x': [], 'p_y': [], 'p_z': [], 'o_x': [], 'o_y': [], 'o_z': [], 'o_w': []}
        
        # Iterate through each SegmentState in the SegmentStateArray
        for state in states_msg.states:
            # Extract the segment id
            data['id'].append(state.id)
            
            # Extract the position components
            data['p_x'].append(state.pose.position.x)
            data['p_y'].append(state.pose.position.y)
            data['p_z'].append(state.pose.position.z)
            
            # Extract the orientation components
            data['o_x'].append(state.pose.orientation.x)
            data['o_y'].append(state.pose.orientation.y)
            data['o_z'].append(state.pose.orientation.z)
            data['o_w'].append(state.pose.orientation.w)
        
        return data

    def convert_state_dict_to_numpy(self, state_dict):
        # Extract the lists from the dictionary
        p_x = state_dict['p_x']
        p_y = state_dict['p_y']
        p_z = state_dict['p_z']
        o_x = state_dict['o_x']
        o_y = state_dict['o_y']
        o_z = state_dict['o_z']
        o_w = state_dict['o_w']

        # Stack the lists into a 2D numpy array and transpose it
        dlo_state = np.array([p_x, p_y, p_z, o_x, o_y, o_z, o_w]).T # shape: (n, 7)

        return dlo_state
    
    def calculate_dlo_length(self, approximated_state_pos):
        # Link lengths
        link_lengths = np.linalg.norm(approximated_state_pos[1:,:] - approximated_state_pos[:-1,:], axis=1)
        return np.sum(link_lengths)
                
    def wrench_to_numpy(self, wrench):
        """
        Converts a ROS wrench message to a numpy array.

        :param wrench: The wrench (force and torque) in ROS message format.
        :return: A numpy array representing the wrench.
        """
        # Combine force and torque arrays
        return np.array([wrench.force.x, wrench.force.y, wrench.force.z, wrench.torque.x, wrench.torque.y, wrench.torque.z])

    def publish_path(self, planned_path):
        """
        Publishes the planned path of the robot to the /path topic.
        """
        # Create a Path message
        path = Path()
        path.header.frame_id = "map"
        path.header.stamp = rospy.Time.now()

        # Add the poses to the path
        for pose in planned_path:
            # print(pose)
            path.poses.append(pose)

        # Publish the path
        self.path_pub.publish(path)
                
    def publish_paths_of_particles(self, planned_path_of_particles):
        """
        Publishes the planned path of the particles to the /path_of_particles topic.
        """
        # Create a Path message
        path_of_particles = {}
        for particle, path in planned_path_of_particles.items():
            path_msg = Path()
            path_msg.header.frame_id = "map"
            path_msg.header.stamp = rospy.Time.now()

            # Add the poses to the path
            for pose in path:
                path_msg.poses.append(pose)
                # print(pose)

            # Publish the path
            path_of_particles[particle] = path_msg

            self.path_pub_particles[particle].publish(path_msg)

    def smooth_scores(self, errs):
        """
        Smooths the scores using Savitzky-Golay filter
        https://en.wikipedia.org/wiki/Savitzky%E2%80%93Golay_filter
        """
        window_size = len(errs) // 4
        poly_order=3
    
        # Smooth the errors
        smoothed_errs = savgol_filter(errs, window_size, poly_order)
        
        err_changes = np.diff(smoothed_errs)
        
        # Smooth the error changes
        smoothed_err_changes = savgol_filter(err_changes, window_size, poly_order)
        
        peak_err = np.max(smoothed_errs)
        peak_err_change = np.max(smoothed_err_changes)
        
        peak_err_waypoint_idx = np.argmax(smoothed_errs)
        peak_err_change_idx = np.argmax(smoothed_err_changes)
                    
        # Calculate the average error
        avr_err = np.mean(smoothed_errs)
                
        # Calculate the average error change
        avr_err_change = np.mean(smoothed_err_changes)
                
        scores = (smoothed_errs,
                peak_err,
                peak_err_waypoint_idx,
                avr_err,
                
                smoothed_err_changes,
                peak_err_change,
                peak_err_change_idx,
                avr_err_change)
        
        return scores
        
            
    def plot_dlo_sim_state_vs_rigid_link_apprx_comparison(self, polyline, points, perpendicular_angle=False):
        try:
            ax = plt.figure().add_subplot(projection='3d')
            ax.figure.set_size_inches(32, 18)
            
            if self.scene_id is not None and self.experiment_number is not None:
                fig_title = f"Pose Comparisons\nScene {self.scene_id}, Experiment {self.experiment_number}, Waypoint {self.planned_path_current_target_index}"
            else:
                fig_title = f"Pose Comparisons\nWaypoint {self.planned_path_current_target_index}"
                
            ax.set_title(fig_title, fontsize=30)
            
            ax.plot(points[:, 0], points[:, 1], points[:, 2],
                    'Xg', label='Original Centers', markersize=10, mec='k', alpha=0.5)
            
            ax.plot(polyline[:, 0], polyline[:, 1], polyline[:, 2],
                    '-b', label='Approximation Line', markersize=12, linewidth=3)
        
            ax.legend(fontsize=20)
            ax.tick_params(axis='both', which='major', labelsize=20)
            set_axes_equal(ax)
            
            if perpendicular_angle:
                # Combine and center the data
                data = np.vstack((points, polyline))
                mean_data = np.mean(data, axis=0)
                data_centered = data - mean_data

                # Compute PCA
                cov_mat = np.cov(data_centered.T)
                eigenvalues, eigenvectors = np.linalg.eigh(cov_mat)
                idx = np.argsort(eigenvalues)[::-1]
                eigenvectors = eigenvectors[:, idx]

                # The viewing direction is the third principal component
                viewing_direction = eigenvectors[:, 2]

                # Compute elevation and azimuth angles
                azim = np.degrees(np.arctan2(viewing_direction[1], viewing_direction[0]))
                elev = np.degrees(np.arcsin(viewing_direction[2] / np.linalg.norm(viewing_direction)))

                # Set the view to be perpendicular to the first two principal axes
                ax.view_init(elev=elev, azim=azim)
            
            plt.show()
        except Exception as e:
            print(f"Error plotting the path points: {e}")
        
    def plot_dlo_sim_state_vs_rigid_link_apprx_comparisons(self, frames, 
                                                        save_as_animation=False, 
                                                        animation_file=None,
                                                        perpendicular_angle=False):
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from matplotlib import animation

            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            # Set figure size
            ax.figure.set_size_inches(32, 18)
            
            # Initialize the plot objects for the original points and the approximation line
            orig_points_plot, = ax.plot([], [], [], 'Xg', label='Original Centers', markersize=10, mec='k', alpha=0.5)
            approx_line_plot, = ax.plot([], [], [], '-b', label='Approximation Line', linewidth=3)
        
            # Function to set equal axis scaling for all three dimensions
            def set_axes_equal(ax):
                """Set equal scaling for all three axes in a 3D plot."""
                limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
        
                # Calculate the span for each axis
                spans = limits[:, 1] - limits[:, 0]
                # Find the maximum span across all axes
                max_span = np.max(spans)
                # Calculate the centers for each axis
                centers = np.mean(limits, axis=1)
        
                # Set limits so all axes cover the same range
                for i, center in enumerate(centers):
                    limits[i] = [center - max_span / 2, center + max_span / 2]
        
                ax.set_xlim3d(limits[0])
                ax.set_ylim3d(limits[1])
                ax.set_zlim3d(limits[2])
        
            # Dynamically adjust the axis limits and set them to equal scale
            def set_dynamic_axes_limits(points, polyline):
                all_data = np.vstack([points, polyline])  # Combine both sets of points
                x_limits = [np.min(all_data[:, 0]), np.max(all_data[:, 0])]
                y_limits = [np.min(all_data[:, 1]), np.max(all_data[:, 1])]
                z_limits = [np.min(all_data[:, 2]), np.max(all_data[:, 2])]
        
                ax.set_xlim(x_limits)
                ax.set_ylim(y_limits)
                ax.set_zlim(z_limits)
                set_axes_equal(ax)  # Ensure equal scaling after setting limits
        
            # Initialize function for FuncAnimation
            def init():
                orig_points_plot.set_data([], [])
                orig_points_plot.set_3d_properties([])
                approx_line_plot.set_data([], [])
                approx_line_plot.set_3d_properties([])
                return orig_points_plot, approx_line_plot
        
            # Update function for FuncAnimation, with dynamic title for each waypoint
            def update_frame(data):
                frame_idx, frame_data = data  # Unpack the tuple
                points, polyline = frame_data
                orig_points_plot.set_data(points[:, 0], points[:, 1])
                orig_points_plot.set_3d_properties(points[:, 2])
                approx_line_plot.set_data(polyline[:, 0], polyline[:, 1])
                approx_line_plot.set_3d_properties(polyline[:, 2])
                
                # Update the title to reflect the scene, experiment ID, and current waypoint number
                fig_title = f"Scene {self.scene_id}, Experiment {self.experiment_number}, Waypoint {frame_idx}"
                ax.set_title(fig_title, fontsize=30)
                
                # Update the axes limits to fit the current frame's data and set equal scaling
                set_dynamic_axes_limits(points, polyline)
                
                if perpendicular_angle:
                    # Compute PCA and adjust the view
                    all_data = np.vstack((points, polyline))
                    mean_data = np.mean(all_data, axis=0)
                    data_centered = all_data - mean_data

                    # Compute PCA
                    cov_mat = np.cov(data_centered.T)
                    eigenvalues, eigenvectors = np.linalg.eigh(cov_mat)
                    idx = np.argsort(eigenvalues)[::-1]
                    eigenvectors = eigenvectors[:, idx]

                    # The viewing direction is the third principal component
                    viewing_direction = eigenvectors[:, 2]

                    # Compute elevation and azimuth angles
                    azim = np.degrees(np.arctan2(viewing_direction[1], viewing_direction[0]))
                    elev = np.degrees(np.arcsin(viewing_direction[2] / np.linalg.norm(viewing_direction)))

                    # Set the view to be perpendicular to the first two principal axes
                    ax.view_init(elev=elev, azim=azim)
                else:
                    # Optionally, reset to default view if needed
                    ax.view_init(elev=30, azim=-60)  # Default Matplotlib 3D view angles

                return orig_points_plot, approx_line_plot

            # Create the animation, pass frames as tuples with (index, data)
            ani = animation.FuncAnimation(fig, update_frame, frames=enumerate(frames), init_func=init, interval=self.wp_wait_time*1000, blit=True)
        
            # Automatically generate the filename if none is provided
            if save_as_animation:
                if animation_file is None:
                    formatted_experiment_id = f"{self.experiment_number:03d}"
                    animation_file = f"pose_comparison_scene_{self.scene_id}_experiment_{formatted_experiment_id}.mp4"
                
                ani.save(animation_file, writer='ffmpeg', fps=5)
                print(f"Animation saved as {animation_file}")
            else:
                plt.show()
        
        except Exception as e:
            print(f"Error plotting the path points: {e}")

def plot_scores(scores):
    """ Plot the scores of the path with respect to the waypoints, also highlight the peak error and the peak error change points.

    Args:
        errs (List[float]): List of avr errors for each waypoint
        peak_err (float): Peak error value
        peak_err_waypoint_idx (int): Index of the waypoint with the peak error
        avr_err (float): Average error value 
        err_changes (List[float]): List of error changes for each waypoint
        peak_err_change (float): Peak error change value
        peak_err_change_idx (int): Index of the waypoint with the peak error change
        avr_err_change (float): Average error change value
    """
    
    # Plot the scores
    fig, ax = plt.subplots(2, 1, figsize=(32, 18))

    if len(scores) == 9:
        (errs,
        peak_err,
        peak_err_waypoint_idx,
        avr_err,
        
        err_changes,
        peak_err_change,
        peak_err_change_idx,
        avr_err_change,
        
        scoring_duration_per_waypoint) = scores
        
        # Figure title
        fig.suptitle(f"Scene {scene_id} Experiment {experiment_number} Path Achievability Scores", fontsize=40)
        
        ax[0].plot(errs, label="Error", markersize=12,  linewidth=8)
        ax[0].plot([peak_err_waypoint_idx], [peak_err], 'ro', label="Peak Error", markersize=12,  linewidth=8)
        ax[0].axhline(y=avr_err, color='g', linestyle='--', label="Average Error", markersize=12,  linewidth=8)
        ax[0].set_title("Error vs Waypoint Index", fontsize=35)
        # ax[0].set_xlabel("Waypoint Index", fontsize=30)
        ax[0].set_ylabel("Error (mm)",fontsize=30)
        ax[0].legend(fontsize=35)
        ax[0].tick_params(axis='both', which='major', labelsize=20)
        
        ax[1].plot(err_changes, label="Error Change", markersize=12,  linewidth=8, )
        ax[1].plot([peak_err_change_idx], [peak_err_change], 'ro', label="Peak Error Increase", markersize=12,  linewidth=8)
        ax[1].axhline(y=avr_err_change, color='g', linestyle='--', label="Average Error Change", markersize=12,  linewidth=8)
        ax[1].set_title("Error Change vs Waypoint Index", fontsize=35)
        ax[1].set_xlabel("Waypoint Index", fontsize=30)
        ax[1].set_ylabel("Error Change (mm)", fontsize=30)
        ax[1].legend(fontsize=35)
        ax[1].tick_params(axis='both', which='major', labelsize=20)
        
        plt.show()
        
    if len(scores) == 17:
        (errs,
        peak_err,
        peak_err_waypoint_idx,
        avr_err,
        
        err_changes,
        peak_err_change,
        peak_err_change_idx,
        avr_err_change,
        
        smoothed_errs,
        smoothed_peak_err,
        smoothed_peak_err_waypoint_idx,
        smoothed_avr_err,
        
        err_changes_on_smoothed,
        peak_err_change_on_smoothed,
        peak_err_change_idx_on_smoothed,
        avr_err_change_on_smoothed,
        
        scoring_duration_per_waypoint) = scores
    
        # Figure title
        fig.suptitle(f"Scene {scene_id} Experiment {experiment_number} Path Achievability Scores", fontsize=40)
        
        # ------------------------------ Error --------------------------------
        # Plot the scores
        ax[0].plot(errs, label="Error", markersize=12,  linewidth=8)
        ax[0].plot([peak_err_waypoint_idx], [peak_err], 'ro', label="Peak Error", markersize=12,  linewidth=8)
        ax[0].axhline(y=avr_err, color='g', linestyle='--', label="Average Error", markersize=12,  linewidth=8)
        
        # Plot the smoothed scores
        ax[0].plot(smoothed_errs, label="Smoothed Error", markersize=12,  linewidth=8)
        ax[0].plot([smoothed_peak_err_waypoint_idx], [smoothed_peak_err], 'bo', label="Smoothed Peak Error", markersize=12,  linewidth=8)
        ax[0].axhline(y=smoothed_avr_err, color='b', linestyle='--', label="Smoothed Average Error", markersize=12,  linewidth=8)
        
        ax[0].set_title("Error vs Waypoint Index", fontsize=35)
        # ax[0].set_xlabel("Waypoint Index", fontsize=30)
        ax[0].set_ylabel("Error (mm)",fontsize=30)
        ax[0].legend(fontsize=35)
        ax[0].tick_params(axis='both', which='major', labelsize=20)
        
        # ------------------------------ Error Change --------------------------------
        # Plot the scores
        ax[1].plot(err_changes, label="Error Change", markersize=12,  linewidth=8, )
        ax[1].plot([peak_err_change_idx], [peak_err_change], 'ro', label="Peak Error Increase", markersize=12,  linewidth=8)
        ax[1].axhline(y=avr_err_change, color='g', linestyle='--', label="Average Error Change", markersize=12,  linewidth=8)
        
        # Plot the smoothed scores
        ax[1].plot(err_changes_on_smoothed, label="Smoothed Error Change", markersize=12,  linewidth=8)
        ax[1].plot([peak_err_change_idx_on_smoothed], [peak_err_change_on_smoothed], 'bo', label="Smoothed Peak Error Increase", markersize=12,  linewidth=8)
        ax[1].axhline(y=avr_err_change_on_smoothed, color='b', linestyle='--', label="Smoothed Average Error Change", markersize=12,  linewidth=8)
        
        ax[1].set_title("Error Change vs Waypoint Index", fontsize=35)
        ax[1].set_xlabel("Waypoint Index", fontsize=30)
        ax[1].set_ylabel("Error Change (mm)", fontsize=30)
        ax[1].legend(fontsize=35)
        ax[1].tick_params(axis='both', which='major', labelsize=20)
        
        plt.show()
    

def save_scores(scene_id, experiment_number, saved_paths_dir, scores):
    """ Save the scores to a csv file

    Args:
        scene_id (int): Scene ID
        experiment_number (int): Experiment Number
        scores (Tuple): Tuple of scores
    """
    rospy.loginfo("Saving scores to a csv file")
    
    if scores is not None:
        # Process the file name
        scene_dir = f"scene_{scene_id}"
        file_name = f"scene_{scene_id}_path_achievability_scores.csv"  # e.g. "scene_1_path_achievability_scores.csv"
        
        scores_csv_file = os.path.expanduser(os.path.join(saved_paths_dir, scene_dir, file_name))
        
        # Also create a folder to store the scores as pickle files
        scores_pickle_dir = os.path.expanduser(os.path.join(saved_paths_dir, scene_dir, "scores"))
        if not os.path.exists(scores_pickle_dir):
            os.makedirs(scores_pickle_dir)
            print(f"Created a folder to store the scores: {scores_pickle_dir}")
            
        # Save the scores as a pickle file
        # File name for the results
        # Ensure experiment_id is always three digits long with leading zeros
        formatted_experiment_id = f"{experiment_number:03d}"
        scores_pickle_file = os.path.join(scores_pickle_dir, f"scene_{scene_id}_experiment_{formatted_experiment_id}_achievability_scores.pkl")
        with open(scores_pickle_file, 'wb') as f:
            pickle.dump(scores, f)
        rospy.loginfo(f"Scores saved to {scores_pickle_file}")
        
        # Save some useful score statistics to a csv file
        if len(scores) == 9:
            (errs,
            peak_err,
            peak_err_waypoint_idx,
            avr_err,
            
            err_changes,
            peak_err_change,
            peak_err_change_idx,
            avr_err_change,
            
            scoring_duration_per_waypoint) = scores
            
            row_title = ["experiment_id", "peak_err", "peak_err_waypoint_idx", "avr_err", "peak_err_change", "peak_err_change_idx", "avr_err_change", "scoring_duration_per_waypoint"]
            
            # If the file does not exist, create it and write the header
            if not os.path.exists(scores_csv_file):
                with open(scores_csv_file, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(row_title)
                    
            # Write the scores
            with open(scores_csv_file, 'a', newline='') as f:
                writer = csv.writer(f)
                if scores is not None:
                    writer.writerow([experiment_number, peak_err, peak_err_waypoint_idx, avr_err, peak_err_change, peak_err_change_idx, avr_err_change, scoring_duration_per_waypoint])
                else:
                    writer.writerow([experiment_number] + ["Nan"]*(len(row_title)-1))
                    rospy.logwarn("Scores are None, writing Nan values to the csv file")
                rospy.loginfo(f"Scores saved to {scores_csv_file}")
            
        if len(scores) == 17:
            (errs,
            peak_err,
            peak_err_waypoint_idx,
            avr_err,
            
            err_changes,
            peak_err_change,
            peak_err_change_idx,
            avr_err_change,
            
            smoothed_errs,
            smoothed_peak_err,
            smoothed_peak_err_waypoint_idx,
            smoothed_avr_err,
            
            err_changes_on_smoothed,
            peak_err_change_on_smoothed,
            peak_err_change_idx_on_smoothed,
            avr_err_change_on_smoothed,
            
            scoring_duration_per_waypoint) = scores
            
            row_title = ["experiment_id", "peak_err", "peak_err_waypoint_idx", "avr_err", "peak_err_change", "peak_err_change_idx", "avr_err_change",
                        "smoothed_peak_err", "smoothed_peak_err_waypoint_idx", "smoothed_avr_err", "peak_err_change_on_smoothed", "peak_err_change_idx_on_smoothed", "avr_err_change_on_smoothed", "scoring_duration_per_waypoint"]
            
            # If the file does not exist, create it and write the header
            if not os.path.exists(scores_csv_file):
                with open(scores_csv_file, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(row_title)
                    
            # Write the scores
            with open(scores_csv_file, 'a', newline='') as f:
                writer = csv.writer(f)
                if scores is not None:
                    writer.writerow([experiment_number, peak_err, peak_err_waypoint_idx, avr_err, peak_err_change, peak_err_change_idx, avr_err_change,
                                    smoothed_peak_err, smoothed_peak_err_waypoint_idx, smoothed_avr_err, peak_err_change_on_smoothed, peak_err_change_idx_on_smoothed, avr_err_change_on_smoothed, scoring_duration_per_waypoint])
                else:
                    writer.writerow([experiment_number] + ["Nan"]*(len(row_title)-1))
                    rospy.logwarn("Scores are None, writing Nan values to the csv file")
                rospy.loginfo(f"Scores saved to {scores_csv_file}")

        
if __name__ == "__main__":
    rospy.init_node('path_achievability_scorer_node', anonymous=False)
    
    scorer = PathAchievabilityScorer()
    
    # User inputs
    scene_id = 3
    experiment_number = 1
    saved_paths_dir = "~/catkin_ws_deformable/src/deformable_manipulations_tent_building/src/tesseract_planner/generated_plans_i9_10885h"
    # saved_paths_dir = "~/catkin_ws_deformable/src/deformable_manipulations_tent_building/src/tesseract_planner/generated_plans_i9_10885h_10_segments"
    
    # Process the file name
    scene_dir = f"scene_{scene_id}"
    formatted_experiment_id = f"{experiment_number:03d}"
    file_name = f"scene_{scene_id}_experiment_{formatted_experiment_id}_data.pkl"  # e.g. "scene_1_experiment_001_data.pkl"
    path_pickle_file = os.path.expanduser(os.path.join(saved_paths_dir, scene_dir, file_name))
    rospy.loginfo("Path (Pickle) File to be scored: " + path_pickle_file)
    
    # Score the path
    scores = scorer.score_path_from_pickle_file(path_pickle_file=path_pickle_file, 
                                                scene_id=scene_id, experiment_number=experiment_number)
    
    # Print and plot the scores
    if scores is not None:
        # Unpack the scores
        (errs,
        peak_err,
        peak_err_waypoint_idx,
        avr_err,
        
        err_changes,
        peak_err_change,
        peak_err_change_idx,
        avr_err_change,
        
        smoothed_errs,
        smoothed_peak_err,
        smoothed_peak_err_waypoint_idx,
        smoothed_avr_err,
        
        err_changes_on_smoothed,
        peak_err_change_on_smoothed,
        peak_err_change_idx_on_smoothed,
        avr_err_change_on_smoothed,
        
        scoring_duration_per_waypoint) = scores # SCORE UNITS ARE IN MILLIMETERS!!
                
        # Print the scores
        rospy.loginfo(f"Peak Error: {peak_err} at waypoint index {peak_err_waypoint_idx}")
        rospy.loginfo(f"Average Error: {avr_err}")
        rospy.loginfo(f"Peak Error Change: {peak_err_change} at waypoint index {peak_err_change_idx}")
        rospy.loginfo(f"Average Error Change: {avr_err_change}")
        
        # Print the smoothed scores
        rospy.loginfo(f"Smoothed Peak Error: {smoothed_peak_err} at waypoint index {smoothed_peak_err_waypoint_idx}")
        rospy.loginfo(f"Smoothed Average Error: {smoothed_avr_err}")
        rospy.loginfo(f"Smoothed Peak Error Change: {peak_err_change_on_smoothed} at waypoint index {peak_err_change_idx_on_smoothed}")
        rospy.loginfo(f"Smoothed Average Error Change: {avr_err_change_on_smoothed}")
        
        rospy.loginfo(f"Average scoring time per waypoint: {scoring_duration_per_waypoint} seconds.")
        
        # Plot the scores
        # plot_scores(scores) # Plot ALL scores
        # plot_scores(scores[:8] + (scores[-1],)) # Plot only the raw scores (First 8 elements, and the last element is the scoring duration)
        plot_scores(scores[-9:]) # Plot only the smoothed scores (Last 9 elements)
        
        # Once all iterations are done, call the function to generate the animation
        scorer.plot_dlo_sim_state_vs_rigid_link_apprx_comparisons(scorer.frames, 
                                                                save_as_animation=True, 
                                                                animation_file=None)
    else:
        rospy.logwarn("Scores are None")

    # Save the scores to a csv file
    save_scores(scene_id, experiment_number, saved_paths_dir, scores)

    rospy.spin()