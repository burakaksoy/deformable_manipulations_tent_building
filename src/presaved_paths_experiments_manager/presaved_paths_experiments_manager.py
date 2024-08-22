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
from .rosbag_controlled_recording import RosbagControlledRecorder

from geometry_msgs.msg import Twist, Point, PointStamped, Quaternion, Pose, PoseStamped, Wrench, Vector3
from nav_msgs.msg import Odometry, Path

class PresavedPathsExperimentsManager(object):
    def __init__(self, velocity_controller_node, scene_id, experiments_range, saved_paths_dir):
        
        self.velocity_controller_node = velocity_controller_node
        self.scene_id = scene_id
        self.experiments_range = experiments_range
        self.experiment_number = self.experiments_range[0] - 1 # Start from the first experiment number
        self.saved_paths_dir = saved_paths_dir
        
        self.is_experiments_completed = False
        
        self.rosbag_file = None # The rosbag file name with full path
        self.odom_topics = None # List of odom topics to record and reverse the executed path
        self.rosbag_recorder = None # RosbagControlledRecorder object, None if not recording
        
    def start_next_experiment(self):
        if self.experiment_number >= self.experiments_range[1]:            
            self.is_experiments_completed = True
            rospy.loginfo("No more experiments to run.")
            
        if not self.is_experiments_completed:
            self.experiment_number += 1
            self.run_experiment(self.experiment_number)
        else:
            rospy.logwarn("Experiments are completed. Plese restart the node to run again.")
        
    def run_experiment(self, experiment_number):
        rospy.loginfo("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        rospy.loginfo("Running experiment number %d" % experiment_number)
        
        # Read presaved path for the experiment
        presaved_path_variables = self.load_presaved_path(experiment_number)
        
        # Assign the path to the controller
        self.velocity_controller_node.set_path(presaved_path_variables)
        
        # Start rosbag recording
        self.start_rosbag_recording(experiment_number)
        
        # Call the controller_enabler(true)
        self.velocity_controller_node.controller_enabler(True)
        
    def end_experiment(self, cause="manual", execution_results=None):
        if not self.is_experiments_completed:
            rospy.loginfo("Experiment ended")
            rospy.loginfo("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            
            # End rosbag recording
            self.stop_rosbag_recording()
            
            # Save results to csv file
            self.save_experiment_results(execution_results, self.experiment_number)
            
            # Reverse the executed path
            self.reverse_executed_path(speed_multiplier=2.0)
            
            # Re-apply the initial state
            self.apply_initial_state()
            
            if cause == "manual":
                # self.is_experiments_completed = True
                rospy.loginfo("Experiment ended manually")
                if not self.is_experiments_completed:
                    time_to_wait = 2
                    rospy.loginfo("Starting next experiment in %d seconds.." % time_to_wait)
                    time.sleep(time_to_wait)
                    self.start_next_experiment()
                
            else:
                if not self.is_experiments_completed:
                    time_to_wait = 2
                    rospy.loginfo("Starting next experiment in %d seconds.." % time_to_wait)
                    time.sleep(time_to_wait)
                    self.start_next_experiment()
        else:
            rospy.logwarn("Cannot end the experiment since Experiments are already completed. Plese restart the node to run again.")
                
    def load_presaved_path(self, experiment_number):
        rospy.loginfo("Loading presaved path for experiment number %d" % experiment_number)
        
        scene_dir = f"scene_{self.scene_id}"
        formatted_experiment_id = f"{experiment_number:03d}"
        file_name = f"scene_{self.scene_id}_experiment_{formatted_experiment_id}_data.pkl"  # e.g. "scene_1_experiment_001_data.pkl"
        
        path_file = os.path.expanduser(os.path.join(self.saved_paths_dir, scene_dir, file_name))
        rospy.loginfo("File: " + path_file)
        
        with open(path_file, 'rb') as inp:
            plan_data = pickle.load(inp)
        
            plan_data_ompl = plan_data[0]
            plan_data_trajopt = plan_data[1] 
            performance_data = plan_data[2]
            initial_n_target_states = plan_data[3]

        return plan_data_trajopt
        
    def start_rosbag_recording(self, experiment_number, compress=True):
        rospy.loginfo("Starting rosbag recording")
        
        if self.rosbag_recorder is not None:
            rospy.logwarn("Rosbag recorder is already running. Stopping the current recording and starting a new one.")
            self.stop_rosbag_recording()
        
        # Create the rosbag file name
        scene_dir = f"scene_{self.scene_id}"
        formatted_experiment_id = f"{experiment_number:03d}"
        file_name = f"scene_{self.scene_id}_experiment_{formatted_experiment_id}.bag"  # e.g. "scene_1_experiment_001.bag"
        self.rosbag_file = os.path.expanduser(os.path.join(self.saved_paths_dir, scene_dir, file_name))
        
        # Topics to record
        topics = ["/dlo_markers", "/dlo_state", "/min_dist_markers", 
                    "/min_dist_to_rigid_bodies", "/rigid_body_markers", 
                    "/spacenav/twist", "/rosout", "/rosout_agg"]
        
        # All velocity controller specific topics
        vel_controller_topics = ["-e \"/tent_building_velocity_controller.*\""]
        topics.extend(vel_controller_topics)
        
        # All odom particles topics
        self.odom_topics = [f"/odom_particle_{i}" for i in self.velocity_controller_node.custom_static_particles]
        # e.g. ["/odom_particle_1", "/odom_particle_2"]
        topics.extend(self.odom_topics)
        
        # Convert to a single string with spaces
        topics_str = " ".join(topics)
        
        # Compress flag
        compress_str = "--bz2" if compress else ""
        
        # Create the rosbag command
        rosbag_command = f"rosbag record --output-name={self.rosbag_file} {compress_str} {topics_str}"
        
        # ROSbag recorder object
        self.rosbag_recorder = RosbagControlledRecorder(rosbag_command)
        
        # Start recording
        self.rosbag_recorder.start_recording_srv()
        
        # Wait for a few seconds to make sure the recording has started
        time_to_wait = 2
        rospy.loginfo(f"Waiting for {time_to_wait} seconds for the rosbag recording to start completely..")
        time.sleep(time_to_wait)
        return
        
    def stop_rosbag_recording(self):
        rospy.loginfo("Stopping rosbag recording")
        
        if self.rosbag_recorder is not None:
            self.rosbag_recorder.stop_recording_srv()
            self.rosbag_recorder = None
        else:
            rospy.logwarn("Rosbag recorder is not running. Nothing to stop.")
        return
        
    def save_experiment_results(self,execution_results, experiment_id):
        rospy.loginfo("Saving experiment results")
        
        # Create the results csv file name
        scene_dir = f"scene_{self.scene_id}"
        file_name = f"scene_{self.scene_id}_experiment_execution_results.csv"  # e.g. "scene_1_experiment_001_execution_results.csv"
        self.csv_file = os.path.expanduser(os.path.join(self.saved_paths_dir, scene_dir, file_name))
        
        row_title = ["experiment_id", "ft_on", "collision_on", "success", "min_distance", "rate", "duration", "stress_avoidance_performance_avr", "stress_avoidance_performance_ever_zero", "start_time", "final_task_error"]
        
        # If the file does not exist, create it and write the header
        if not os.path.exists(self.csv_file):
            with open(self.csv_file, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(row_title)
                rospy.loginfo(f"File '{self.csv_file}' created and header written.")
                
        # Append the results to the csv file
        with open(self.csv_file, 'a', newline='') as file:
            writer = csv.writer(file)
            if execution_results:
                writer.writerow([str(int(experiment_id))] + execution_results)
            else:
                writer.writerow([str(int(experiment_id))] + ["Nan" for _ in range(len(row_title)-1)])
                rospy.logwarn("No execution results to save, writing 'Nan' values.")                
            rospy.loginfo(f"Results appended to the CSV file: '{self.csv_file}'.")
            
    def reverse_executed_path(self, speed_multiplier=1.0):
        rospy.loginfo("Reversing executed path")
        
        # Timed history of Odometry topics (self.odom_topics) stores the executed path for each particle
        
        # Read the recorded rosbag file
        bag = rosbag.Bag(self.rosbag_file)
        
        # Dictionaries to store commands and time deltas for each particle
        commands = {particle: [] for particle in self.velocity_controller_node.custom_static_particles}
        time_deltas = {particle: [] for particle in self.velocity_controller_node.custom_static_particles}
        last_timestamps = {particle: None for particle in self.velocity_controller_node.custom_static_particles}
        
        # Read and store commands with time differences for each topic
        for topic, msg, t in bag.read_messages(topics=self.odom_topics):
            # Extract the particle number from the topic name
            particle = int(topic.split('_')[-1])
            
            if last_timestamps[particle] is not None:
                time_deltas[particle].insert(0, (1.0/speed_multiplier)*(t - last_timestamps[particle]).to_sec())
            last_timestamps[particle] = t
            commands[particle].insert(0, self.reverse_odom_msg(msg,speed_multiplier))  # Reverse the order of commands as we store them
        
        bag.close()
        
        # Determine the minimum length of command sequences to handle synchronized publishing
        min_len = min(len(commands[particle]) for particle in self.velocity_controller_node.custom_static_particles)
        
        for i in range(min_len):
            # Publish commands to all particles simultaneously using pre-existing publishers
            for particle in self.velocity_controller_node.custom_static_particles:
                self.velocity_controller_node.odom_publishers[particle].publish(commands[particle][i])
            
            # Filter out empty sequences before calling min()
            valid_time_deltas = [time_deltas[particle][i] for particle in self.velocity_controller_node.custom_static_particles if i < len(time_deltas[particle])]
            
            if valid_time_deltas:  # Check if the list is not empty
                sleep_time = min(valid_time_deltas)
                time.sleep(sleep_time)
        
        # Handle any remaining commands for each particle
        for particle in self.velocity_controller_node.custom_static_particles:
            for i in range(min_len, len(commands[particle])):
                self.velocity_controller_node.odom_publishers[particle].publish(commands[particle][i])
                if i < len(time_deltas[particle]):
                    time.sleep(time_deltas[particle][i])
        
    def reverse_odom_msg(self, odom_msg, speed_multiplier=1.0):
        odom = Odometry()
        odom.header = odom_msg.header # No need to reverse the header
        odom.child_frame_id = odom_msg.child_frame_id # No need to reverse the child_frame_id
        odom.pose = odom_msg.pose # No need to reverse the pose
        # Reverse the twist
        odom.twist.twist.linear.x = -odom_msg.twist.twist.linear.x*speed_multiplier
        odom.twist.twist.linear.y = -odom_msg.twist.twist.linear.y*speed_multiplier
        odom.twist.twist.linear.z = -odom_msg.twist.twist.linear.z*speed_multiplier
        odom.twist.twist.angular.x = -odom_msg.twist.twist.angular.x*speed_multiplier
        odom.twist.twist.angular.y = -odom_msg.twist.twist.angular.y*speed_multiplier
        odom.twist.twist.angular.z = -odom_msg.twist.twist.angular.z*speed_multiplier
        return odom 
        
    def apply_initial_state(self):
        rospy.loginfo("Applying initial state")
        
        # Read the initial "/dlo_state" topic from the rosbag file
        bag = rosbag.Bag(self.rosbag_file)
        initial_state = None
        for topic, msg, t in bag.read_messages(topics="/dlo_state"):
            initial_state = msg
            break
        bag.close()
        
        for particle in self.velocity_controller_node.custom_static_particles:
            odom = Odometry()
            odom.header.stamp = rospy.Time.now()  # Corrected to use stamp
            odom.header.frame_id = "map"
            
            # Assign the pose from the initial state to the odometry message
            odom.pose.pose = initial_state.states[particle].pose
            
            # Twist remains zero as we are setting the initial state
            
            # Publish the initial state to the corresponding odom topic
            self.velocity_controller_node.odom_publishers[particle].publish(odom)
        
        return
        