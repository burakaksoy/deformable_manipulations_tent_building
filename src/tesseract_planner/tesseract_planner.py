import os
import re
import traceback
import numpy as np
import time
import sys
from pathlib import Path

import rospy

from geometry_msgs.msg import Twist, Point, PointStamped, Quaternion, Pose, PoseStamped, Wrench, Vector3

from tesseract_robotics.tesseract_common import FilesystemPath, \
                                                Isometry3d, \
                                                Translation3d, \
                                                Quaterniond, \
                                                ManipulatorInfo, \
                                                GeneralResourceLocator, \
                                                CollisionMarginData, \
                                                AnyPoly, \
                                                AnyPoly_wrap_double, \
                                                ResourceLocator, \
                                                SimpleLocatedResource, \
                                                TransformMap, \
                                                CONSOLE_BRIDGE_LOG_DEBUG, \
                                                CONSOLE_BRIDGE_LOG_INFO, \
                                                CONSOLE_BRIDGE_LOG_WARN, \
                                                CONSOLE_BRIDGE_LOG_ERROR, \
                                                CONSOLE_BRIDGE_LOG_NONE, \
                                                setLogLevel, \
                                                Timer, \
                                                AngleAxisd

from tesseract_robotics.tesseract_environment import Environment, \
                                                     AddLinkCommand, \
                                                     AddSceneGraphCommand, \
                                                     Commands
                                                    

from tesseract_robotics.tesseract_scene_graph import Joint, \
                                                     Link, \
                                                     Visual, \
                                                     Collision, \
                                                     JointType_FIXED, \
                                                     Material

from tesseract_robotics.tesseract_geometry import Sphere, \
                                                    Box, \
                                                    Cylinder, \
                                                    ConvexMesh, \
                                                    Mesh, \
                                                    Plane, \
                                                    MeshMaterial

from tesseract_robotics.tesseract_command_language import CartesianWaypoint, \
                                                          CartesianWaypointPoly, \
                                                          CartesianWaypointPoly_wrap_CartesianWaypoint, \
                                                          WaypointPoly, \
                                                          JointWaypoint, \
                                                          JointWaypointPoly, \
                                                          StateWaypointPoly_wrap_StateWaypoint, \
                                                          JointWaypointPoly_wrap_JointWaypoint, \
                                                          InstructionPoly, \
                                                          MoveInstruction, \
                                                          MoveInstructionPoly, \
                                                          MoveInstructionType_FREESPACE, \
                                                          MoveInstructionType_LINEAR, \
                                                          MoveInstructionType_CIRCULAR, \
                                                          MoveInstructionPoly_wrap_MoveInstruction, \
                                                          InstructionPoly_as_MoveInstructionPoly, \
                                                          ProfileDictionary, \
                                                          WaypointPoly_as_StateWaypointPoly, \
                                                          StateWaypoint, \
                                                          StateWaypointPoly, \
                                                          CompositeInstruction, \
                                                          AnyPoly_as_CompositeInstruction, \
                                                          AnyPoly_wrap_CompositeInstruction, \
                                                          CompositeInstructionOrder_ORDERED, \
                                                          DEFAULT_PROFILE_KEY, \
                                                          toJointTrajectory

from tesseract_robotics_viewer import TesseractViewer
                                      
from tesseract_robotics.tesseract_task_composer import  TaskComposerPluginFactory, \
                                                        PlanningTaskComposerProblem, \
                                                        PlanningTaskComposerProblemUPtr, \
                                                        PlanningTaskComposerProblemUPtr_as_TaskComposerProblemUPtr, \
                                                        TaskComposerDataStorage, \
                                                        TaskComposerContext, \
                                                        TaskComposerFuture, \
                                                        TaskComposerFutureUPtr, \
                                                        MinLengthProfile, \
                                                        ProfileDictionary_addProfile_MinLengthProfile
                                                                                                                
from tesseract_robotics.tesseract_motion_planners_simple import generateInterpolatedProgram

from tesseract_robotics.tesseract_motion_planners_ompl import OMPLDefaultPlanProfile, \
                                                              RRTConnectConfigurator, \
                                                              OMPLProblemGeneratorFn, \
                                                              OMPLMotionPlanner, \
                                                              ProfileDictionary_addProfile_OMPLPlanProfile

from tesseract_robotics.tesseract_motion_planners_trajopt import TrajOptDefaultPlanProfile,\
                                                                 TrajOptPlanProfile, \
                                                                 ProfileDictionary_addProfile_TrajOptPlanProfile, \
                                                                 TrajOptDefaultCompositeProfile, \
                                                                 TrajOptCompositeProfile, \
                                                                 ProfileDictionary_addProfile_TrajOptCompositeProfile, \
                                                                 TrajOptDefaultSolverProfile, \
                                                                 TrajOptSolverProfile, \
                                                                 ProfileDictionary_addProfile_TrajOptSolverProfile, \
                                                                 TrajOptProblemGeneratorFn, \
                                                                 TrajOptMotionPlanner, \
                                                                 CollisionEvaluatorType_SINGLE_TIMESTEP, \
                                                                 CollisionEvaluatorType_DISCRETE_CONTINUOUS, \
                                                                 CollisionEvaluatorType_CAST_CONTINUOUS, \
                                                                 ModelType, \
                                                                 BasicTrustRegionSQPParameters
                                                                 
from tesseract_robotics.tesseract_collision import ContactTestType_ALL, \
                                                   ContactTestType_FIRST, \
                                                   ContactTestType_CLOSEST
                                                   
from tesseract_robotics.tesseract_time_parameterization import TimeOptimalTrajectoryGeneration, \
                                                               InstructionsTrajectory
                                                               
from tesseract_robotics.tesseract_urdf import parseURDFString, \
                                              parseURDFFile, \
                                              writeURDFFile

import tf.transformations as tf_trans

from .utils.add_env_obstacles import add_environment_obstacles, \
                                             add_environment_obstacles_l_shape_corridor, \
                                             add_environment_obstacles_from_urdf
                                             
from .utils.add_profiles import add_MinLengthProfile, \
                                        add_TrajOptPlanProfile, \
                                        add_OMPLDefaultPlanProfile

from deformable_simulator_scene_utilities import json_str_to_urdf, json_to_urdf

## Set the log level 
# setLogLevel(CONSOLE_BRIDGE_LOG_DEBUG)
# setLogLevel(CONSOLE_BRIDGE_LOG_INFO)
# setLogLevel(CONSOLE_BRIDGE_LOG_WARN)
# setLogLevel(CONSOLE_BRIDGE_LOG_ERROR)
# setLogLevel(CONSOLE_BRIDGE_LOG_NONE)

class TesseractPlanner(object):
    def __init__(self, 
                 collision_scene_json_file,
                 tesseract_resource_path,
                 urdf_url_or_path,
                 srdf_url_or_path,
                 tcp_frame,
                 manipulator,
                 working_frame  ,
                 viewer_enabled = True):
        rospy.loginfo("Initializing TesseractPlanner..")
        
        # Set the resource path for tesseract
        os.environ["TESSERACT_RESOURCE_PATH"] = os.path.expanduser(tesseract_resource_path)
        
        # -----------------------------------------------------------------------
        # Get the directory of the current script
        self.current_file_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Path to the config files
        self.config_path = os.path.join(self.current_file_dir, 'config')
        
        # Path to the task composer config file
        # self.task_composer_filename = os.environ["TESSERACT_TASK_COMPOSER_CONFIG_FILE"]        
        self.task_composer_filename = os.path.join(self.config_path, 'task_composer_plugins_no_trajopt_ifopt.yaml')
        
        # Create the task composer plugin factory and load the plugins
        self.factory = TaskComposerPluginFactory(FilesystemPath(self.task_composer_filename))    
        # -----------------------------------------------------------------------
        
        # --------------------------------------------------------------------------------------------
        # Create the task composer node. In this case the FreespacePipeline is used. Many other are available.
        # self.task = self.factory.createTaskComposerNode("FreespacePipeline")
        # self.task = self.factory.createTaskComposerNode("TrajOptPipeline")
        # self.task = self.factory.createTaskComposerNode("OMPLPipeline")

        # # Disabled DiscreteContactCheckTask so that it moves through obstacles in the animation
        self.task = self.factory.createTaskComposerNode("FreespacePipeline2") 
        # # Disabled DiscreteContactCheckTask so that it moves through obstacles in the animation
        # self.task = self.factory.createTaskComposerNode("TrajOptPipeline2") 

        # Get the output keys for the task
        self.task_output_key = self.task.getOutputKeys()[0]

        # Create an executor to run the task
        self.task_executor = self.factory.createTaskComposerExecutor("TaskflowExecutor")
        # --------------------------------------------------------------------------------------------

        # -----------------------------------------------------------------------
        # Initialize the resource locator and environment
        self.locator = GeneralResourceLocator() # locator_fn must be kept alive by maintaining a reference
        self.urdf_fname = FilesystemPath(self.locator.locateResource(urdf_url_or_path).getFilePath())
        self.srdf_fname = FilesystemPath(self.locator.locateResource(srdf_url_or_path).getFilePath())
        self.env = Environment()
        assert self.env.init(self.urdf_fname, self.srdf_fname, self.locator)
        # -----------------------------------------------------------------------
        
        # -----------------------------------------------------------------------
        # Fill in the manipulator information. This is used to find the kinematic chain for the manipulator. This must
        # match the SRDF, although the exact tcp_frame can differ if a tool is used.
        self.tcp_frame = tcp_frame
        
        self.manip_info = ManipulatorInfo()
        self.manip_info.tcp_frame = self.tcp_frame
        self.manip_info.manipulator = manipulator
        self.manip_info.working_frame = working_frame
        # -----------------------------------------------------------------------
        
        # -----------------------------------------------------------------------
        # Create a viewer and set the environment so the results can be displayed later
        self.viewer_enabled = viewer_enabled
        self.viewer = None
        if self.viewer_enabled:
            # Create a viewer and set the environment so the results can be displayed later
            self.viewer = TesseractViewer()
            
            # Show the world coordinate frame
            self.viewer.add_axes_marker(position=[0,0,0], quaternion=[1,0,0,0], 
                                        size=1.0, parent_link="base_link", name="world_frame")
            self.viewer.update_environment(self.env, [0,0,0])
            
            # Start the viewer
            self.viewer.start_serve_background()
        # -----------------------------------------------------------------------

        # -----------------------------------------------------------------------
        ## Set the initial state of the robot
        # Get the joint names from the environment:
        self.joint_group = self.env.getJointGroup(self.manip_info.manipulator)
        self.joint_names = list(self.joint_group.getJointNames())
        
        # Another way to get the joint names:
        # self.joint_names = list(self.env.getGroupJointNames("manipulator"))
        
        # print("joint_names: ", joint_names)
        # print("")
        # -----------------------------------------------------------------------
        
        # -----------------------------------------------------------------------
        self.load_collision_scene(collision_scene_json_file)
        # -----------------------------------------------------------------------
        
        # -----------------------------------------------------------------------
        # ## Set the collision margin for check. Objects with closer than the specified margin will be returned
        # # Tesseract reports all GJK/EPA distance within contact_distance threshold
        
        # margin_distance = 0.01 # 3cm margin
        
        # env_link_names = self.env.getLinkNames()
        
        # # Get the discrete contact manager. This must be called again after the environment is updated
        # self.contact_manager_discrete = self.env.getDiscreteContactManager()
        # self.contact_manager_discrete.setActiveCollisionObjects(env_link_names)
        # self.contact_manager_discrete.setCollisionMarginData(CollisionMarginData(margin_distance))
        
        # # Get the continuous contact manager. 
        # self.contact_manager_continuous = self.env.getContinuousContactManager()
        # self.contact_manager_continuous.setActiveCollisionObjects(env_link_names)
        # self.contact_manager_continuous.setCollisionMarginData(CollisionMarginData(margin_distance))
        # -----------------------------------------------------------------------
        
        # # ------------
        # # Get the state solver. This must be called again after environment is updated
        # solver = env.getStateSolver()

        # # Set the transform of the active collision objects from SceneState
        # solver.setState(joint_names, initial_joint_positions)
        # scene_state = solver.getState()
        # manager.setCollisionObjectsTransform(scene_state.link_transforms)
        # # ------------
        rospy.loginfo("TesseractPlanner initialized.")
  
    def load_collision_scene(self, json_file_path):
        # locator = GeneralResourceLocator()
        # locator = TesseractSupportResourceLocator()
        
        scene_urdf_str = json_to_urdf(input_file_path=json_file_path, 
                                      visualize=False, 
                                      save_output=False,
                                      output_file_path="./tesseract_scene.urdf")
        # print("scene_urdf_str: ")
        # print(scene_urdf_str)
        
        scene_graph_to_add = parseURDFString(scene_urdf_str, self.locator).release()
        add_scene_graph_command = AddSceneGraphCommand(scene_graph_to_add)
    
        cmds = Commands()
        cmds.push_back(add_scene_graph_command)
        self.env.applyCommands(cmds)
        
        if self.viewer_enabled:
            self.viewer.update_environment(self.env, [0,0,0])
           
    def set_current_state(self, centroid_pose):
        """
        centroid_pose is a ROS Pose message
        but we assume the values of the pose are the joint values of the robot of the 
        tessaract environment
        """
        
        print("Setting current state..")
        
        if self.viewer_enabled:
            # Clear all markers in the viewer
            self.viewer.clear_all_markers()
        
        init_x = centroid_pose.position.x
        init_y = centroid_pose.position.y
        init_z = centroid_pose.position.z
        
        init_q = [centroid_pose.orientation.w,
                  centroid_pose.orientation.x,
                  centroid_pose.orientation.y,
                  centroid_pose.orientation.z] # convert to wxyz format
        
        init_rpy = tf_trans.euler_from_quaternion([centroid_pose.orientation.x,
                                                    centroid_pose.orientation.y,
                                                    centroid_pose.orientation.z,
                                                    centroid_pose.orientation.w], 'sxyz')
        
        init_roll = init_rpy[0]
        init_pitch = init_rpy[1]
        init_yaw = init_rpy[2]
        
        if self.viewer_enabled:
            # Add the initial pose to the viewer
            self.viewer.add_axes_marker(position=[init_x, init_y, init_z], 
                                          quaternion=init_q, 
                                          size=0.5, 
                                          parent_link=self.manip_info.working_frame, 
                                          name="init_frame")

            
        initial_joint_positions = np.zeros(len(self.joint_names))
        initial_joint_positions[:6] = np.array([init_x, init_y, init_z, 
                                                     init_yaw, init_pitch, init_roll])
        
        # Set the initial state of the robot in the environment
        self.env.setState(self.joint_names, initial_joint_positions)
        
        if self.viewer_enabled:
            # self.viewer.update_environment(self.env, [0,0,0])
            self.viewer.update_joint_positions(self.joint_names, initial_joint_positions)
            
        return initial_joint_positions
              
    def set_goal_state(self, goal_pose):
        """
        goal_pose is a ROS Pose message
        but we assume the values of the pose are the joint values of the robot of the 
        tessaract environment
        """
        
        goal_x = goal_pose.position.x
        goal_y = goal_pose.position.y
        goal_z = goal_pose.position.z
        
        goal_q = [goal_pose.orientation.w,
                  goal_pose.orientation.x,
                  goal_pose.orientation.y,
                  goal_pose.orientation.z]
        
        goal_rpy = tf_trans.euler_from_quaternion([goal_pose.orientation.x,
                                                    goal_pose.orientation.y,
                                                    goal_pose.orientation.z,
                                                    goal_pose.orientation.w], 'sxyz')
        
        goal_roll = goal_rpy[0]
        goal_pitch = goal_rpy[1]
        goal_yaw = goal_rpy[2]
        
        if self.viewer_enabled:
            # Add the goal pose to the viewer
            self.viewer.add_axes_marker(position=[goal_x, goal_y, goal_z], 
                                          quaternion=goal_q, 
                                          size=0.5, 
                                          parent_link=self.manip_info.working_frame, 
                                          name="goal_frame")
            
        goal_joint_positions = np.zeros(len(self.joint_names))
        goal_joint_positions[:6] = np.array([goal_x, goal_y, goal_z,
                                                    goal_yaw, goal_pitch, goal_roll])
        
        return goal_joint_positions
        
    def set_state_waypoints(self, initial_joint_positions ,waypoints, goal_joint_positions):
        """
        waypoints: A list of Pose messages representing the waypoints.
        """
        
        # Create a list of StateWaypointPoly
        state_waypoints = []
        
        # Add the initial state waypoint
        initial_state_waypoint = StateWaypointPoly_wrap_StateWaypoint(StateWaypoint(self.joint_names, initial_joint_positions))
        state_waypoints.append(initial_state_waypoint)
        
        # Add the intermediate waypoints
        itr = 1
        for waypoint in waypoints:
            x = waypoint.position.x
            y = waypoint.position.y
            z = waypoint.position.z
            
            q = [waypoint.orientation.w,
                 waypoint.orientation.x,
                 waypoint.orientation.y,
                 waypoint.orientation.z]
            
            rpy = tf_trans.euler_from_quaternion([waypoint.orientation.x,
                                                  waypoint.orientation.y,
                                                  waypoint.orientation.z,
                                                  waypoint.orientation.w], 'sxyz')
            roll = rpy[0]
            pitch = rpy[1]
            yaw = rpy[2]
            
            if self.viewer_enabled:
                # Add the waypoint pose to the viewer
                self.viewer.add_axes_marker(position=[x, y, z], 
                                            quaternion=q, 
                                            size=0.5,
                                            parent_link= self.manip_info.working_frame,
                                            name="wp_" + str(itr))
                
            joint_positions = np.zeros(len(self.joint_names))
            joint_positions[:6] = np.array([x, y, z, yaw, pitch, roll])
            
            wp_state_waypoint = StateWaypointPoly_wrap_StateWaypoint(StateWaypoint(self.joint_names, joint_positions))
            state_waypoints.append(wp_state_waypoint)
            
            itr += 1
        
        # Add the goal state waypoint
        goal_state_waypoint = StateWaypointPoly_wrap_StateWaypoint(StateWaypoint(self.joint_names, goal_joint_positions))
        state_waypoints.append(goal_state_waypoint)
        
        return state_waypoints
            
    def set_move_instructions(self, state_waypoints):
        """
        waypoints: A list of Tessearct StateWaypointPoly objects
        
        Note the use of *_wrap_MoveInstruction functions. This is required because the
        Python bindings do not support implicit conversion from the MoveInstruction to the MoveInstructionPoly.
        """
        
        move_instructions = []
        
        for state_waypoint in state_waypoints:
            move_instruction = MoveInstructionPoly_wrap_MoveInstruction(MoveInstruction(state_waypoint, 
                                                                                        MoveInstructionType_FREESPACE, 
                                                                                        "DEFAULT"))
            move_instructions.append(move_instruction)
        
        return move_instructions
    
    def create_input_command_program(self, manip_info, move_instructions):
        """
        Create an input command program for the Tesseract planner
        
        Args:
            manip_info: ManipulatorInfo object
            move_instructions: A list of MoveInstructionPoly objects
        """
    
        # Create the input command program using CompositeInstruction
        program = CompositeInstruction("DEFAULT")
        # program = CompositeInstruction("freespace_profile")
        program.setManipulatorInfo(manip_info)
         
        # Add the MoveInstructionPoly objects to the CompositeInstruction
        for move_instruction in move_instructions:
            program.appendMoveInstruction(move_instruction)
            
        # Print diagnosics
        program._print("Program: ")
        
        ## Create an AnyPoly containing the program. 
        # This explicit step is required because the Python bindings do not
        # support implicit conversion from the CompositeInstruction to the AnyPoly.
        program_anypoly = AnyPoly_wrap_CompositeInstruction(program)
        
        rospy.loginfo("Program created.")
        
        return program_anypoly
        
    def set_profile_dictionary(self):        
        # Create a profile dictionary. Profiles can be customized by adding to this dictionary and setting the profiles
        # in the instructions.
        rospy.loginfo("Setting profile dictionary..")
        
        profiles = ProfileDictionary()
        
        add_MinLengthProfile(profiles, "DEFAULT")

        add_TrajOptPlanProfile(profiles, "DEFAULT")

        add_OMPLDefaultPlanProfile(profiles, "DEFAULT")
        
        rospy.loginfo("Profile dictionary set.")
        return profiles
        
    def plan(self, current_pose, waypoints, custom_static_particles):
        """
        Args:
            current_pose: Pose message as the current centroid of the custom static particles of the deformable object
            waypoints: A list of Pose messages representing the waypoints.
            custom_static_particles: List of ids of the custom static particles of the deformable object segments
            
        Returns:
            planned_path: A list of PoseStamped() msgs
            planned_path_points: path 3D xyz points as numpy array
            planned_path_cumulative_lengths: cumulative lengths of the path segments
            planned_path_cumulative_rotations: cumulative rotations of the path segments obtained from the angles in axis-angle representation consecutive rotations in radians
            planned_path_direction_vectors: directions of the path segments as unit vectors
            planned_path_rotation_vectors: rotation axes of the path segments as unit vectors
            
            planned_path_of_particles: planned path of the particles as a list of Pose() msgs
            planned_path_points_of_particles: path 3D xyz points of the particles as numpy array
            planned_path_cumulative_lengths_of_particles: cumulative lengths of the path segments of the particles
            planned_path_cumulative_rotations_of_particles: cumulative rotations of the path segments of the particles obtained from the angles in axis-angle representation consecutive rotations in radians
            planned_path_direction_vectors_of_particles: directions of the path segments of the particles as unit vectors (list of n elements with each element is a 3D vector for each particle)
            planned_path_rotation_vectors_of_particles: rotation axes of the path segments of the particles as unit vectors (list of n elements with each element is a 3D vector for each particle)
        """
        
        print("Planning path..")
        # Set the current state of the robot as the initial state and
        # Send the simplified object to the center of the custom static particles
        initial_joint_positions = self.set_current_state(current_pose)
        
        # Set the goal state
        try:
            goal_pose = waypoints[-1] # Note: The last waypoint is the goal pose
            goal_joint_positions = self.set_goal_state(goal_pose)
        except:
            print("Error setting goal state: waypoints list is empty")
            traceback.print_exc()
            return None
        
        # The remaining waypoints are the intermediate waypoints
        intermediate_waypoints = waypoints[:-1]
        
        state_waypoints = self.set_state_waypoints(initial_joint_positions, 
                                                   intermediate_waypoints,
                                                   goal_joint_positions) 
        
        move_instructions = self.set_move_instructions(state_waypoints)
        
        program_anypoly = self.create_input_command_program(self.manip_info, move_instructions)
        
        profiles = self.set_profile_dictionary()
        
        # Create the task problem and 
        rospy.loginfo("Creating the task planning problem..")
        task_planning_problem = PlanningTaskComposerProblem(self.env, profiles)
        task_planning_problem.input = program_anypoly
        rospy.loginfo("Task planning problem created.")

        # --------------------------------------------------------------------------------------------        
        # Solve task
        rospy.loginfo("Planning the task..")
        stopwatch = Timer()
        stopwatch.start()

        # Run the task and wait for completion
        future = self.task_executor.run(self.task.get(), task_planning_problem)
        future.wait()

        stopwatch.stop()
        rospy.loginfo(f"PLANNING TOOK {stopwatch.elapsedSeconds()} SECONDS.")
        # --------------------------------------------------------------------------------------------

        try:
            # Retrieve the output, converting the AnyPoly back to a CompositeInstruction
            results = AnyPoly_as_CompositeInstruction(future.context.data_storage.getData(self.task_output_key))
        
            # Display the output
            # Print out the resulting waypoints
            for instr in results:
                assert instr.isMoveInstruction()
                move_instr1 = InstructionPoly_as_MoveInstructionPoly(instr)
                wp1 = move_instr1.getWaypoint()
                assert wp1.isStateWaypoint()
                wp = WaypointPoly_as_StateWaypointPoly(wp1)
                # print("-------------------------------------------------------------")
                # print(f"Joint Time: {wp.getTime()}")
                # print(f"Joint Positions: {wp.getPosition().flatten()} time: {wp.getTime()}")
                # print("Joint Names: " + str(list(wp.getNames())))
                # print(f"Joint Velocities: {wp.getVelocity().flatten()}")
                # print(f"Joint Accelerations: {wp.getAcceleration().flatten()}")
                # print(f"Joint Efforts: {wp.getEffort().flatten()}")
                # print("-------------------------------------------------------------")
        except:
            ## Assertions failed, fall back to the fallback plan
            rospy.logwarn("Error getting the results of the global plan, Using the fallback plan")
            # results = results_fallback # TODO
            
        if self.viewer_enabled:
            # Update the viewer with the results to animate the trajectory
            # Open web browser to http://localhost:8000 to view the results
            # self.viewer.clear_all_markers()
            self.viewer.update_trajectory(results)
            self.viewer.plot_trajectory(results, self.manip_info)

        # --------------------------------------------------------------------------------------------
        ## Convert the results to ROS Path message and corresponding data
        
        # Get the joint values from the results
        joint_values = self.tesseract_trajectory_to_joint_values_list(results)
        
        ## Frame names that we are interested in
        frame_names = [self.tcp_frame] # This is the default frame name for the end-effector 
        # and also the centroid of the deformable object model in the tesseract environment
        
        # Here we assume the frame names of the particles are "pole_holder_link_1", "pole_holder_link_2", ...
        particle_frame_name_prefix = "pole_holder_link_"
        for particle_id in custom_static_particles:
            frame_names.append(particle_frame_name_prefix + str(particle_id))
        
        ## Convert the joint values to the planned path data
        (planned_path_of_frames, 
        planned_path_points_of_frames,
        planned_path_cumulative_lengths_of_frames,
        planned_path_cumulative_rotations_of_frames,
        planned_path_direction_vectors_of_frames,
        planned_path_rotation_vectors_of_frames) = self.joint_values_to_planned_path_data(joint_values, frame_names)
        
        ## Convert the joint values to the planned path data for the particles
        frame_names.remove(self.tcp_frame)
        planned_path = planned_path_of_frames.pop(self.tcp_frame)
        planned_path_points = planned_path_points_of_frames.pop(self.tcp_frame)
        planned_path_cumulative_lengths = planned_path_cumulative_lengths_of_frames.pop(self.tcp_frame)
        planned_path_cumulative_rotations = planned_path_cumulative_rotations_of_frames.pop(self.tcp_frame)
        planned_path_direction_vectors = planned_path_direction_vectors_of_frames.pop(self.tcp_frame)
        planned_path_rotation_vectors = planned_path_rotation_vectors_of_frames.pop(self.tcp_frame)

        ## Assign the keys of the dictionaries to the custom static particle ids
        planned_path_of_particles = {p_id: planned_path_of_frames.get(particle_frame_name_prefix + str(p_id), None) 
                                        for p_id in custom_static_particles}
        planned_path_points_of_particles = {p_id: planned_path_points_of_frames.get(particle_frame_name_prefix + str(p_id), None) 
                                        for p_id in custom_static_particles}
        planned_path_cumulative_lengths_of_particles = {p_id: planned_path_cumulative_lengths_of_frames.get(particle_frame_name_prefix + str(p_id), None) 
                                        for p_id in custom_static_particles}
        planned_path_cumulative_rotations_of_particles = {p_id: planned_path_cumulative_rotations_of_frames.get(particle_frame_name_prefix + str(p_id), None) 
                                        for p_id in custom_static_particles}
        planned_path_direction_vectors_of_particles = {p_id: planned_path_direction_vectors_of_frames.get(particle_frame_name_prefix + str(p_id), None) 
                                        for p_id in custom_static_particles}
        planned_path_rotation_vectors_of_particles = {p_id: planned_path_rotation_vectors_of_frames.get(particle_frame_name_prefix + str(p_id), None) 
                                        for p_id in custom_static_particles}
        
        
        return (planned_path,
        planned_path_points,
        planned_path_cumulative_lengths,
        planned_path_cumulative_rotations,
        planned_path_direction_vectors,
        planned_path_rotation_vectors, 
        planned_path_of_particles,
        planned_path_points_of_particles,
        planned_path_cumulative_lengths_of_particles,
        planned_path_cumulative_rotations_of_particles,
        planned_path_direction_vectors_of_particles,
        planned_path_rotation_vectors_of_particles)
        
    def tesseract_trajectory_to_joint_values_list(self, tesseract_trajectory):
        """
            Convert the tesseract trajectory to a list of joint values
        """    
        joint_values = []
        for instr in tesseract_trajectory: 
            # assert instr.isMoveInstruction() # each instr is assumed return true for instr.isMoveInstruction()
            instr_m = InstructionPoly_as_MoveInstructionPoly(instr)
            wp = instr_m.getWaypoint() 
            # assert wp.isStateWaypoint() # each wp is assumed return true for wp.isStateWaypoint()
            state_wp = WaypointPoly_as_StateWaypointPoly(wp)
            joint_values.append(state_wp.getPosition().flatten().tolist())
            
            # Other information that can be also extracted from each state waypoint
            # Joint Names:          list(state_wp.getNames())
            # Joint Times:          state_wp.getTime()
            # Joint Velocities:     state_wp.getVelocity().flatten()
            # Joint Accelerations:  state_wp.getAcceleration().flatten()
            # Joint Efforts:        state_wp.getEffort().flatten()
        return joint_values
    
    def joint_values_to_planned_path_data(self, joint_values, frame_names):
        """
        Convert the joint values to the frame poses using the forward kinematics

        Args:
            joint_values (List[List[float]]): List of joint values to get the poses using the forward kinematics
            frame_names (List[str]): List of frame names to get the poses of
            
        Returns:
            planned_path_of_frames: A dictionary keyed by the frame names with the values as a list of PoseStamped() messages
            planned_path_points_of_frames: A dictionary keyed by the frame names with the values as a numpy array of 3D points
            planned_path_cumulative_lengths_of_frames: A dictionary keyed by the frame names with the values as a list of cumulative lengths of the path segments
            planned_path_cumulative_rotations_of_frames: A dictionary keyed by the frame names with the values as a list of cumulative rotations of the path segments obtained from the angles in axis-angle representation consecutive rotations in radians
            planned_path_direction_vectors_of_frames: A dictionary keyed by the frame names with the values as a list of directions of the path segments as unit vectors
            planned_path_rotation_vectors_of_frames: A dictionary keyed by the frame names with the values as a list of rotation axes of the path segments as unit vectors
        """
        # Initialize the dictionaries
        planned_path_of_frames = {}
        planned_path_points_of_frames = {}
        planned_path_cumulative_lengths_of_frames = {}
        planned_path_cumulative_rotations_of_frames = {}
        planned_path_direction_vectors_of_frames = {}
        planned_path_rotation_vectors_of_frames = {}
        
        for frame_name in frame_names:
            # Initialize the list of poses (PoseStamped() msgs) for each frame
            planned_path_of_frames[frame_name] = []
            # Initialize the numpy array of 3D points for each frame
            planned_path_points_of_frames[frame_name] = []
            # Initialize the list of cumulative lengths of the path segments for each frame
            planned_path_cumulative_lengths_of_frames[frame_name] = [0.0] # Start with 0.0
            # Initialize the list of cumulative rotations of the path segments for each frame
            planned_path_cumulative_rotations_of_frames[frame_name] = [0.0] # Start with 0.0 
            # Initialize the list of directions of the path segments as unit vectors for each frame
            planned_path_direction_vectors_of_frames[frame_name] = []
            # Initialize the list of rotation axes of the path segments as unit vectors for each frame
            planned_path_rotation_vectors_of_frames[frame_name] = []
        
        kin = self.env.getKinematicGroup(self.manip_info.manipulator)
        
        for i in range(len(joint_values)):
            frames = kin.calcFwdKin(np.asarray(joint_values[i], dtype=np.float64))
            # # print the keys of the frames
            # print("frames.keys(): ", frames.keys())
            for frame_name in frame_names:
                if frame_name not in frames:
                    print(f"Frame name: {frame_name} not found in the frames")
                    continue
                
                frame_pose = frames[frame_name]
                
                new_point = frame_pose.translation().flatten()
                q = Quaterniond(frame_pose.rotation())
                new_orientation = [q.x(), q.y(), q.z(), q.w()]
                
                pose_stamped = PoseStamped()
                pose_stamped.pose.position.x = new_point[0]
                pose_stamped.pose.position.y = new_point[1]
                pose_stamped.pose.position.z = new_point[2]
                pose_stamped.pose.orientation.x = new_orientation[0]
                pose_stamped.pose.orientation.y = new_orientation[1]
                pose_stamped.pose.orientation.z = new_orientation[2]
                pose_stamped.pose.orientation.w = new_orientation[3]
                
                planned_path_of_frames[frame_name].append(pose_stamped)
                planned_path_points_of_frames[frame_name].append(new_point)
                
                if i > 0:
                    last_length = planned_path_cumulative_lengths_of_frames[frame_name][-1]
                    
                    last_point = planned_path_points_of_frames[frame_name][-2]
                    
                    direction_vector = np.array(new_point) - np.array(last_point)
                    
                    # Calculate the length of the path segment
                    length = np.linalg.norm(direction_vector)
                    planned_path_cumulative_lengths_of_frames[frame_name].append(last_length + length)
                    
                    direction_vector = direction_vector / np.linalg.norm(direction_vector)
                    planned_path_direction_vectors_of_frames[frame_name].append(direction_vector)
                    
                    last_pose = planned_path_of_frames[frame_name][-2].pose
                    
                    last_orientation = [last_pose.orientation.x,
                                        last_pose.orientation.y,
                                        last_pose.orientation.z,
                                        last_pose.orientation.w]
                    
                    # Relative rotation quaternion from last to new
                    quaternion_difference = tf_trans.quaternion_multiply(new_orientation, tf_trans.quaternion_inverse(last_orientation))

                    # Normalize the quaternion to avoid numerical issues
                    quaternion_difference = self.normalize_quaternion(quaternion_difference)
                    
                    # AXIS-ANGLE ORIENTATION ERROR/DIFFERENCE DEFINITION
                    # Convert quaternion difference to rotation vector (axis-angle representation)
                    rotation_vector = self.quaternion_to_rotation_vec(quaternion_difference)
                    rotation_angle = np.linalg.norm(rotation_vector)

                    if rotation_angle > 0:
                        # Normalize the rotation vector to get the rotation axis
                        planned_path_rotation_vectors_of_frames[frame_name].append(rotation_vector/rotation_angle)
                    else:
                        # If the rotation angle is zero, add a unit vector along the z-axis
                        planned_path_rotation_vectors_of_frames[frame_name].append(np.array([0.0, 0.0, 1.0]))
                    
                    last_rotation = planned_path_cumulative_rotations_of_frames[frame_name][-1]
                    new_rotation = last_rotation + rotation_angle
                    planned_path_cumulative_rotations_of_frames[frame_name].append(new_rotation)

        # Convert the list of 3D points to numpy array
        for frame_name in frame_names:
            planned_path_points_of_frames[frame_name] = np.array(planned_path_points_of_frames[frame_name])                 
                
        return (planned_path_of_frames, 
                planned_path_points_of_frames,
                planned_path_cumulative_lengths_of_frames,
                planned_path_cumulative_rotations_of_frames,
                planned_path_direction_vectors_of_frames,
                planned_path_rotation_vectors_of_frames)

    def normalize_quaternion(self, quaternion):
        norm = np.linalg.norm(quaternion)
        if norm == 0:
            raise ValueError("Cannot normalize a quaternion with zero norm.")
        return quaternion / norm

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
        else:
            # Regular calculation for larger angles
            axis = quaternion[:3] / np.sin(angle/2.0)
            # Normalize the axis
            axis = axis / np.linalg.norm(axis)
            rotation_vector = angle * axis 
        return rotation_vector
    
