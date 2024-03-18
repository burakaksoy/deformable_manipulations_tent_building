#!/usr/bin/env python3

import sys
import rospy
import numpy as np
import time

from geometry_msgs.msg import Twist, Point, Quaternion, Pose
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker
from std_msgs.msg import Float32, Float32MultiArray

from dlo_simulator_stiff_rods.msg import SegmentStateArray

from std_srvs.srv import SetBool, SetBoolResponse
from std_srvs.srv import Empty, EmptyResponse

import tf.transformations as tf_trans

# import cvxpy as cp

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

        # Create the service server for enable/disable the controller
        self.set_enable_controller_server = rospy.Service('~set_enable_controller', SetBool, self.set_enable_controller)

        self.pub_rate_odom = rospy.get_param("~pub_rate_odom", 50)

        self.custom_static_particles = None
        self.odom_topic_prefix = None
        while (not self.custom_static_particles):
            try:
                self.custom_static_particles = rospy.get_param("/custom_static_particles") # Default static particles 
                self.odom_topic_prefix = rospy.get_param("/custom_static_particles_odom_topic_prefix") # published
            except:
                rospy.logwarn("No particles obtained from ROS parameters!.")
                time.sleep(0.5)
        
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

        # Particle/Segment ids of the tip points of the tent pole 
        # to be placed into the grommets
        self.tip_particles = rospy.get_param("~tip_particles", [0,39])

        # Grommet poses as targets for the tip points
        # Each element holds [[x,y,z],[Rx,Ry,Rz(euler angles in degrees)]]
        target_poses_basic =  rospy.get_param("~target_poses", [[[-1, -1, 0.14], [5,   0, 90]], \
                                                                [[ 1, -1, 0.14], [175, 0, 90]]])
        
        self.target_poses = {} # Each item will be a Pose() msg class
        for i, particle in enumerate(self.tip_particles):
            self.target_poses[particle] = self.calculate_target_pose(target_poses_basic[i])

        self.initial_values_set = False  # Initialization state variable

        self.deformable_object_state_topic_name = rospy.get_param("/dlo_state_topic_name") # subscribed

        # Dictionaries that will hold the state of the custom_static_particles and tip_particles
        self.particle_positions = {}
        self.particle_orientations = {}
        self.particle_twists = {}

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

        self.particle_positions_dy = {}
        self.particle_orientations_dy = {}
        self.particle_twists_dy = {}

        self.particle_positions_dz = {}
        self.particle_orientations_dz = {}
        self.particle_twists_dz = {}

        self.particle_positions_dth_x = {}
        self.particle_orientations_dth_x = {}
        self.particle_twists_dth_x = {}

        self.particle_positions_dth_y = {}
        self.particle_orientations_dth_y = {}
        self.particle_twists_dth_y = {}

        self.particle_positions_dth_z = {}
        self.particle_orientations_dth_z = {}
        self.particle_twists_dth_z = {}

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
        

        
        
        # Create the (centralized) controller that will publish odom to each follower particle properly        
        # self.nominal_controller = NominalController(self.kp, self.kd, self.pub_rate_odom*2.0)
        
        self.control_outputs = {} 
        for particle in self.custom_static_particles:
            self.control_outputs[particle] = np.zeros(6) # initialization for the velocity command



        # Start the control
        self.calculate_control_timer = rospy.Timer(rospy.Duration(1. / self.pub_rate_odom), self.calculate_control_outputs_timer_callback)
        self.odom_pub_timer          = rospy.Timer(rospy.Duration(1. / self.pub_rate_odom), self.odom_pub_timer_callback)





    def set_enable_controller(self, request):
        self.enabled = request.data
        return SetBoolResponse(True, 'Successfully set enabled state to {}'.format(self.enabled))

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

        if not self.initial_values_set:
            # After all initial relative positions and orientations have been calculated, set the initialization state variable to True
            self.initial_values_set = True

    def is_perturbed_states_set_for_particle(self,particle):
        """
        Checks if the pertrubed state parameters (dx, dy, dz, dthx, dthy, dthz) are set for a specified particle.
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

    def state_array_dx_callback(self, states_msg, perturbed_particle):
        if not (perturbed_particle  in self.particle_positions_dx):
            self.particle_positions_dx[perturbed_particle] = {}
            self.particle_orientations_dx[perturbed_particle] = {}
            self.particle_twists_dx[perturbed_particle] = {}

        for particle in (self.custom_static_particles + self.tip_particles):
            self.particle_positions_dx[perturbed_particle][particle] = states_msg.states[particle].pose.position
            self.particle_orientations_dx[perturbed_particle][particle] = states_msg.states[particle].pose.orientation
            self.particle_twists_dx[perturbed_particle][particle] = states_msg.states[particle].twist

    def state_array_dy_callback(self, states_msg, perturbed_particle):
        if not (perturbed_particle in self.particle_positions_dy):
            self.particle_positions_dy[perturbed_particle] = {}
            self.particle_orientations_dy[perturbed_particle] = {}
            self.particle_twists_dy[perturbed_particle] = {}

        for particle in (self.custom_static_particles + self.tip_particles):
            self.particle_positions_dy[perturbed_particle][particle] = states_msg.states[particle].pose.position
            self.particle_orientations_dy[perturbed_particle][particle] = states_msg.states[particle].pose.orientation
            self.particle_twists_dy[perturbed_particle][particle] = states_msg.states[particle].twist

    def state_array_dz_callback(self, states_msg, perturbed_particle):
        if not (perturbed_particle in self.particle_positions_dz):
            self.particle_positions_dz[perturbed_particle] = {}
            self.particle_orientations_dz[perturbed_particle] = {}
            self.particle_twists_dz[perturbed_particle] = {}

        for particle in (self.custom_static_particles + self.tip_particles):
            self.particle_positions_dz[perturbed_particle][particle] = states_msg.states[particle].pose.position
            self.particle_orientations_dz[perturbed_particle][particle] = states_msg.states[particle].pose.orientation
            self.particle_twists_dz[perturbed_particle][particle] = states_msg.states[particle].twist

    def state_array_dth_x_callback(self, states_msg, perturbed_particle):
        if not (perturbed_particle in self.particle_positions_dth_x):
            self.particle_positions_dth_x[perturbed_particle] = {}
            self.particle_orientations_dth_x[perturbed_particle] = {}
            self.particle_twists_dth_x[perturbed_particle] = {}

        for particle in (self.custom_static_particles + self.tip_particles):
            self.particle_positions_dth_x[perturbed_particle][particle] = states_msg.states[particle].pose.position
            self.particle_orientations_dth_x[perturbed_particle][particle] = states_msg.states[particle].pose.orientation
            self.particle_twists_dth_x[perturbed_particle][particle] = states_msg.states[particle].twist

    def state_array_dth_y_callback(self, states_msg, perturbed_particle):
        if not (perturbed_particle in self.particle_positions_dth_y):
            self.particle_positions_dth_y[perturbed_particle] = {}
            self.particle_orientations_dth_y[perturbed_particle] = {}
            self.particle_twists_dth_y[perturbed_particle] = {}

        for particle in (self.custom_static_particles + self.tip_particles):
            self.particle_positions_dth_y[perturbed_particle][particle] = states_msg.states[particle].pose.position
            self.particle_orientations_dth_y[perturbed_particle][particle] = states_msg.states[particle].pose.orientation
            self.particle_twists_dth_y[perturbed_particle][particle] = states_msg.states[particle].twist

    def state_array_dth_z_callback(self, states_msg, perturbed_particle):
        if not (perturbed_particle in self.particle_positions_dth_z):
            self.particle_positions_dth_z[perturbed_particle] = {}
            self.particle_orientations_dth_z[perturbed_particle] = {}
            self.particle_twists_dth_z[perturbed_particle] = {}

        for particle in (self.custom_static_particles + self.tip_particles):
            self.particle_positions_dth_z[perturbed_particle][particle] = states_msg.states[particle].pose.position
            self.particle_orientations_dth_z[perturbed_particle][particle] = states_msg.states[particle].pose.orientation
            self.particle_twists_dth_z[perturbed_particle][particle] = states_msg.states[particle].twist

    def calculate_jacobian_tip(self):
        """
        Calculates the Jacobian matrix that defines the relation btw. 
        the robot hold points (custom_static_particles) 6DoF poses and
        the tent pole tip points (tip_particles) 6 DoF poses 
        """
        J = np.zeros((6*len(self.tip_particles),6*len(self.custom_static_particles)))

        for idx_tip, tip in enumerate(self.tip_particles):
            for idx_particle, particle in enumerate(self.custom_static_particles):
                
                # Do not proceed until the initial values have been set
                if ((not self.is_perturbed_states_set_for_particle(particle))):
                    rospy.logwarn("[calculate_jacobian func.] particle:" + str(particle) + " state is not published yet or it does not have for perturbed states.")
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

        for idx_tip, tip in enumerate(self.tip_particles):
            # Do not proceed until the initial values have been set
            if not (tip in self.particle_positions):
                rospy.logwarn("Tip particle:" + str(tip) + " state is not obtained yet.")
                continue

            current_pose = Pose()
            current_pose.position = self.particle_positions[tip]
            current_pose.orientation = self.particle_orientations[tip]

            target_pose = self.target_poses[tip]

            err[(6*idx_tip) : (6*(idx_tip+1)), 0] = self.calculate_pose_target_error(current_pose,target_pose)

        return err

    def calculate_control_outputs_timer_callback(self,event):
        # Only publish if enabled
        if self.enabled:
            J_tip = self.calculate_jacobian_tip() # 12x12
            # pretty_print_array(J_tip)
            # print("---------------------------")

            err_tip = self.calculate_error_tip() # 12x1
            # pretty_print_array(err_tip)
            # print("---------------------------")

            # control_output = self.kp*np.dot(np.linalg.pinv(J_tip), err_tip)
            control_output = np.dot(np.linalg.pinv(J_tip), err_tip)

            # print(control_output)
            # pretty_print_array(control_output)
            # print("---------------------------")
            

            for idx_particle, particle in enumerate(self.custom_static_particles):

                self.control_outputs[particle] = self.kp * control_output[6*idx_particle:6*(idx_particle+1),0]
                # print(self.control_outputs[particle])


                # # error is 3D np.array, publish its norm for information
                # pos_error_norm = np.linalg.norm(pos_error)
                # self.info_pos_error_norm_publishers[particle].publish(Float32(data=pos_error_norm))

                # # Update the controller terms with the current error and the last executed command
                # # Get control output from the nominal controller
                # control_output = self.nominal_controllers[particle].output(pos_error, self.control_outputs[particle]) # nominal
                # # control_output = np.zeros(3) # disable nominal controller to test only the safe controller

                # # # init_t = time.time()

                
                # # # rospy.logwarn("QP solver calculation time: " + str(1000*(time.time() - init_t)) + " ms.")
                # self.control_outputs[particle] = control_output # to test nominal controller only

                # if self.control_outputs[particle] is not None:
                #     pass

                #     # print("Particle " + str(particle) + " u: " + str(self.control_outputs[particle]))

                #     # rospy.logwarn("control_output:" + str(control_output))
  
            
            
    def odom_pub_timer_callback(self,event):
        # Only publish if enabled
        if self.enabled:
            for particle in self.custom_static_particles:
                
                # Do not proceed until the initial values have been set
                if ((not (particle in self.particle_positions)) or \
                    (not self.is_perturbed_states_set_for_particle(particle))):                    
                    continue
                    
                if self.control_outputs[particle] is not None:
                    # Prepare Odometry message
                    odom = Odometry()
                    odom.header.stamp = rospy.Time.now()
                    odom.header.frame_id = "map"

                    # dt_check = self.nominal_controllers[particle].get_dt() # 
                    dt = 1./self.pub_rate_odom

                    # Control output is the new position
                    odom.pose.pose.position.x =  self.particle_positions[particle].x + self.control_outputs[particle][0]*dt
                    odom.pose.pose.position.y =  self.particle_positions[particle].y + self.control_outputs[particle][1]*dt
                    odom.pose.pose.position.z =  self.particle_positions[particle].z + self.control_outputs[particle][2]*dt

                    # The new orientation
                    axis_angle = np.array(self.control_outputs[particle][3:6])
                    angle = np.linalg.norm(axis_angle)
                    axis = axis_angle / angle if (angle > 1e-9) else np.array([0, 0, 1])

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

                    # Update the pose of the particle 
                    self.particle_positions[particle] = odom.pose.pose.position
                    self.particle_orientations[particle] = odom.pose.pose.orientation

                    # Publish
                    self.odom_publishers[particle].publish(odom)
                else:
                    self.control_outputs[particle] = np.zeros(6)



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
        

if __name__ == "__main__":
    rospy.init_node('velocity_controller_node', anonymous=False)

    node = VelocityControllerNode()

    rospy.spin()
