import csv
import itertools
import os
import pickle
from pathlib import Path
import re
import sys
import time
import traceback

import numpy as np
import pandas as pd
import rospy
import tf.transformations as tf_trans

from geometry_msgs.msg import (
    Point,
    PointStamped,
    Pose,
    PoseStamped,
    Quaternion,
    Twist,
    Vector3,
    Wrench,
)

from tesseract_robotics.tesseract_collision import (
    CollisionCheckProgramType_ALL,
    CollisionCheckProgramType_ALL_EXCEPT_END,
    CollisionCheckProgramType_ALL_EXCEPT_START,
    CollisionCheckProgramType_END_ONLY,
    CollisionCheckProgramType_INTERMEDIATE_ONLY,
    CollisionCheckProgramType_START_ONLY,
    CollisionEvaluatorType_CONTINUOUS,
    CollisionEvaluatorType_DISCRETE,
    CollisionEvaluatorType_LVS_CONTINUOUS,
    CollisionEvaluatorType_LVS_DISCRETE,
    CollisionEvaluatorType_NONE,
    ContactRequest,
    ContactResultMap,
    ContactResultVector,
    ContactTestType_ALL,
    ContactTestType_CLOSEST,
    ContactTestType_FIRST,
)

from tesseract_robotics.tesseract_command_language import (
    AnyPoly_as_CompositeInstruction,
    AnyPoly_wrap_CompositeInstruction,
    CartesianWaypoint,
    CartesianWaypointPoly,
    CartesianWaypointPoly_wrap_CartesianWaypoint,
    CompositeInstruction,
    CompositeInstructionOrder_ORDERED,
    DEFAULT_PROFILE_KEY,
    InstructionPoly,
    InstructionPoly_as_MoveInstructionPoly,
    JointWaypoint,
    JointWaypointPoly,
    JointWaypointPoly_wrap_JointWaypoint,
    MoveInstruction,
    MoveInstructionPoly,
    MoveInstructionPoly_wrap_MoveInstruction,
    MoveInstructionType_CIRCULAR,
    MoveInstructionType_FREESPACE,
    MoveInstructionType_LINEAR,
    ProfileDictionary,
    StateWaypoint,
    StateWaypointPoly,
    StateWaypointPoly_wrap_StateWaypoint,
    toJointTrajectory,
    WaypointPoly,
    WaypointPoly_as_StateWaypointPoly,
)

from tesseract_robotics.tesseract_common import (
    AllowedCollisionMatrix,
    AngleAxisd,
    AnyPoly,
    AnyPoly_wrap_double,
    CollisionMarginData,
    CollisionMarginOverrideType_REPLACE,
    CONSOLE_BRIDGE_LOG_DEBUG,
    CONSOLE_BRIDGE_LOG_ERROR,
    CONSOLE_BRIDGE_LOG_INFO,
    CONSOLE_BRIDGE_LOG_NONE,
    CONSOLE_BRIDGE_LOG_WARN,
    FilesystemPath,
    GeneralResourceLocator,
    Isometry3d,
    ManipulatorInfo,
    Quaterniond,
    ResourceLocator,
    setLogLevel,
    SimpleLocatedResource,
    Timer,
    TransformMap,
    Translation3d,
)

from tesseract_robotics.tesseract_environment import (
    AddContactManagersPluginInfoCommand,
    AddKinematicsInformationCommand,
    AddLinkCommand,
    AddSceneGraphCommand,
    ChangeCollisionMarginsCommand,
    ChangeJointOriginCommand,
    CommandType_MODIFY_ALLOWED_COLLISIONS,
    Commands,
    Environment,
    ModifyAllowedCollisionsCommand,
    ModifyAllowedCollisionsType_ADD,
)

from tesseract_robotics.tesseract_geometry import (
    Box,
    ConvexMesh,
    Cylinder,
    Mesh,
    MeshMaterial,
    Plane,
    Sphere,
)

from tesseract_robotics.tesseract_motion_planners import PlannerRequest, PlannerResponse

from tesseract_robotics.tesseract_motion_planners_ompl import (
    OMPLDefaultPlanProfile,
    OMPLMotionPlanner,
    OMPLProblemGeneratorFn,
    ProfileDictionary_addProfile_OMPLPlanProfile,
    RRTConnectConfigurator,
)

from tesseract_robotics.tesseract_motion_planners_simple import generateInterpolatedProgram

from tesseract_robotics.tesseract_motion_planners_trajopt import (
    BasicTrustRegionSQPParameters,
    CollisionEvaluatorType_CAST_CONTINUOUS,
    CollisionEvaluatorType_DISCRETE_CONTINUOUS,
    CollisionEvaluatorType_SINGLE_TIMESTEP,
    ModelType,
    ProfileDictionary_addProfile_TrajOptCompositeProfile,
    ProfileDictionary_addProfile_TrajOptPlanProfile,
    ProfileDictionary_addProfile_TrajOptSolverProfile,
    TrajOptCompositeProfile,
    TrajOptDefaultCompositeProfile,
    TrajOptDefaultPlanProfile,
    TrajOptDefaultSolverProfile,
    TrajOptMotionPlanner,
    TrajOptPlanProfile,
    TrajOptProblemGeneratorFn,
    TrajOptSolverProfile,
)

from tesseract_robotics.tesseract_scene_graph import (
    Collision,
    Joint,
    JointType_FIXED,
    Link,
    Material,
    SceneGraph,
    Visual,
)

from tesseract_robotics.tesseract_srdf import SRDFModel, processSRDFAllowedCollisions

from tesseract_robotics.tesseract_task_composer import (
    MinLengthProfile,
    PlanningTaskComposerProblem,
    PlanningTaskComposerProblemUPtr,
    PlanningTaskComposerProblemUPtr_as_TaskComposerProblemUPtr,
    ProfileDictionary_addProfile_MinLengthProfile,
    TaskComposerContext,
    TaskComposerDataStorage,
    TaskComposerFuture,
    TaskComposerFutureUPtr,
    TaskComposerPluginFactory,
)

from tesseract_robotics.tesseract_time_parameterization import (
    InstructionsTrajectory,
    TimeOptimalTrajectoryGeneration,
)

from tesseract_robotics.tesseract_urdf import parseURDFFile, parseURDFString, writeURDFFile

from tesseract_robotics_viewer import TesseractViewer

# from tesseract_planner.utils.add_env_obstacles import (
#     add_environment_obstacles,
#     add_environment_obstacles_from_urdf,
#     add_environment_obstacles_l_shape_corridor,
# )

from tesseract_planner.utils.add_profiles import (
    add_MinLengthProfile,
    add_OMPLDefaultPlanProfile,
    add_TrajOptPlanProfile,
)

from deformable_simulator_scene_utilities import json_str_to_urdf, json_to_urdf

from dlo_state_approximator import dlo_state_approximator, dlo_fwd_kin

from dlo_urdf_creator import DloURDFCreator

# -----------------------------------------------------------------------
# OPTIONAL
import matplotlib.pyplot as plt 
import matplotlib.cm as cm # Colormaps

# Set the default DPI for all images
plt.rcParams['figure.dpi'] = 100  # e.g. 300 dpi
# Set the default figure size
# plt.rcParams['figure.figsize'] = [25.6, 19.2]  # e.g. 6x4 inches
plt.rcParams['figure.figsize'] = [12.8, 9.6]  # e.g. 6x4 inches
# plt.rcParams['figure.figsize'] = [6.4, 4.8]  # e.g. 6x4 inches

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
# -----------------------------------------------------------------------

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
        
        self.collision_scene_json_file = collision_scene_json_file
        self.tesseract_resource_path = tesseract_resource_path
        self.urdf_url_or_path = urdf_url_or_path
        self.srdf_url_or_path = srdf_url_or_path
        self.tcp_frame = tcp_frame
        self.manipulator = manipulator
        self.working_frame = working_frame  
        self.viewer_enabled = viewer_enabled 
        
        self.re_init(self.collision_scene_json_file,
                     self.tesseract_resource_path,
                     self.urdf_url_or_path,
                     self.srdf_url_or_path,
                     self.tcp_frame,
                     self.manipulator,
                     self.working_frame,
                     self.viewer_enabled,
                     is_re_init=False)
        
        
    def re_init(self, 
                collision_scene_json_file,
                tesseract_resource_path,
                urdf_url_or_path,
                srdf_url_or_path,
                tcp_frame,
                manipulator,
                working_frame  ,
                viewer_enabled = True,
                is_re_init = False):
        if not is_re_init:
            rospy.loginfo("Initializing TesseractPlanner..")
        else:
            rospy.loginfo("Re-initializing TesseractPlanner..")
        
        if not is_re_init:
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
            # self.task_composer_filename = os.path.join(self.config_path, 'task_composer_plugins_no_trajopt_ifopt_TEST.yaml')
        
        # Create the task composer plugin factory and load the plugins
        self.factory = TaskComposerPluginFactory(FilesystemPath(self.task_composer_filename))    
        # -----------------------------------------------------------------------
        
        # -----------------------------------------------------------------------
        # Create AND INITIALIZE the environment FROM THE COLLISION SCENE
        if not is_re_init:
            # Initialize the resource locator and environment
            self.locator = GeneralResourceLocator() # locator_fn must be kept alive by maintaining a reference
        
        self.env = Environment()
        
        # -----------------------------------------------------------------------
        self.load_collision_scene(collision_scene_json_file)
        # -----------------------------------------------------------------------
        
        # -----------------------------------------------------------------------
        if not is_re_init:
            # Create a viewer and set the environment so the results can be displayed later
            self.viewer_enabled = viewer_enabled
            self.viewer = None
            if self.viewer_enabled:
                # Create a viewer and set the environment so the results can be displayed later
                self.viewer = TesseractViewer()
                
                self.viewer.clear_all_markers()
                
                # Show the world coordinate frame
                self.viewer.add_axes_marker(position=[0,0,0], quaternion=[1,0,0,0], 
                                            size=1.0, parent_link=self.env.getSceneGraph().getRoot(), name="world_frame")
                
                self.viewer.update_environment(self.env, [0,0,0])
                
                # Start the viewer
                self.viewer.start_serve_background()
        # -----------------------------------------------------------------------
        
        # -----------------------------------------------------------------------
        if not is_re_init:
            # self.urdf_fname = FilesystemPath(self.locator.locateResource(urdf_url_or_path).getFilePath())
            self.srdf_fname = FilesystemPath(self.locator.locateResource(srdf_url_or_path).getFilePath())
        # -----------------------------------------------------------------------
        
        # -----------------------------------------------------------------------
        # Fill in the manipulator information. This is used to find the kinematic chain for the manipulator. This must
        # match the SRDF, although the exact tcp_frame can differ if a tool is used.
        
        self.manip_info = ManipulatorInfo()
        self.manip_info.tcp_frame = tcp_frame
        self.manip_info.manipulator = manipulator
        self.manip_info.working_frame = working_frame
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
                
    def plan(self, initial_full_state_dict, 
                    target_full_state_dict,
                    dlo_length=0.5,
                    dlo_radius=0.035,
                    full_dlo_holding_segment_ids=[],
                    max_simplified_dlo_num_segments=None,
                    min_simplified_dlo_num_segments=1,
                    environment_limits_xyz=[-1, 1, -1, 1, -1, 1],
                    joint_angle_limits_xyz_deg=[-90, 90, -180, 180, -10, 10],
                    approximation_error_threshold=0.02,
                    return_all_data=False,
                    plot_for_debugging=False):
        """
        Args:
            
            
        Returns:
        If return_all_data is True:
            plan_data: A dictionary containing ALL the data
        If return_all_data is False:
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
        
        # Convert the state dictionaries to numpy arrays
        initial_full_state = self.convert_state_dict_to_numpy(initial_full_state_dict) # Nx7, N is the number of particles
        target_full_state = self.convert_state_dict_to_numpy(target_full_state_dict) # Nx7, N is the number of particles
        
        # We can figure out the full dlo num segments from the initial state 
        full_dlo_num_segments = initial_full_state.shape[0] # N 
        # full_dlo_num_segments = 40 # Example
        
        # Handle the cases where max_simplified_dlo_num_segments is not given or is not valid
        if (max_simplified_dlo_num_segments is None):
            max_simplified_dlo_num_segments = full_dlo_num_segments 
            print("max_simplified_dlo_num_segments is not given. Setting it to full_dlo_num_segments: ", 
                max_simplified_dlo_num_segments)
            
        if (max_simplified_dlo_num_segments > full_dlo_num_segments):
            max_simplified_dlo_num_segments = full_dlo_num_segments 
            print("max_simplified_dlo_num_segments is greater than full_dlo_num_segments. Setting it to full_dlo_num_segments: ",
                max_simplified_dlo_num_segments)
            
        if (max_simplified_dlo_num_segments < 1):
            max_simplified_dlo_num_segments = 1 
            print("max_simplified_dlo_num_segments is less than 1. Setting it to 1: ", 
                max_simplified_dlo_num_segments)
            
        if min_simplified_dlo_num_segments is None:
            min_simplified_dlo_num_segments = 1
        else:
            min_simplified_dlo_num_segments = min(max(min_simplified_dlo_num_segments, 1), max_simplified_dlo_num_segments)


        simplified_dlo_num_segments = min_simplified_dlo_num_segments - 1
        is_num_segments_validated = False
        dlo_simplification_time = 0.0

        while (simplified_dlo_num_segments <= max_simplified_dlo_num_segments
            and not is_num_segments_validated):
            planning_success = 1
            simplified_dlo_num_segments += 1
            # -----------------------------------------------------------------------
            print("--------------------------------------------------------------------")
            print("Simplified DLO Num Segments: ", simplified_dlo_num_segments)
            print("--------------------------------------------------------------------")
            
            # input("Press enter to reset the environment to add the robot")

            # Reset the environment to make sure the robot is not added to the scene
            self.env.reset()
            # -----------------------------------------------------------------------
            
            # -----------------------------------------------------------------------
            # Approximate the initial and target states of the dlo with the simplified dlo
            
            # INFO: approximated_state_pos:
            # # The approximated DLO state as a list of points (x, y, z) with length equal to num_seg_d + 1. 
            
            # INFO: approximated_state_joint_pos: 
            # # The joint positions of the modeled DLO as a (3+3N) x 3 numpy array. The first 3 elements are the translational joint 
            # # angles (x, y, z) and the last 3N elements are the rotational joint angles around x, y, and z axes for each segment respectively.
            
            # INFO: approximated_state_avg_error:
            # # The average distance error between the original and approximated positions per original segment.
            
            stopwatch = Timer()
            stopwatch.start()
            
            (
            initial_approximated_state_pos,
            initial_approximated_state_joint_pos,
            initial_approximated_state_max_angle, 
            initial_approximated_state_avg_error ) = dlo_state_approximator(dlo_length,
                                                                            initial_full_state,
                                                                            simplified_dlo_num_segments,
                                                                            start_from_beginning=True)
            
            (
            target_approximated_state_pos, 
            target_approximated_state_joint_pos,
            target_approximated_state_max_angle,
            target_approximated_state_avg_error ) = dlo_state_approximator(dlo_length,
                                                                        target_full_state,
                                                                        simplified_dlo_num_segments,
                                                                        start_from_beginning=True)
            
            stopwatch.stop()
            dlo_simplification_time += stopwatch.elapsedSeconds()
            # -----------------------------------------------------------------------

            # -----------------------------------------------------------------------
            # OPTIONAL: Plot the approximated states
            if plot_for_debugging:            
                ax = plt.figure().add_subplot(projection='3d')

                # Add title with the number of segments
                ax.set_title("Initial and Target State Approximations \n w/ Number of Segments = " + str(simplified_dlo_num_segments), fontsize=30)

                ax.plot(initial_full_state[:,0], # x
                        initial_full_state[:,1], # y
                        initial_full_state[:,2], # z
                        'og', label='Initial: Original', markersize=6)
                
                ax.plot(initial_approximated_state_pos[:,0], # x
                        initial_approximated_state_pos[:,1], # y
                        initial_approximated_state_pos[:,2], # z
                        '-g', label='Initial: Approx.', markersize=12, linewidth=3)

                ax.plot(target_full_state[:,0], # x
                        target_full_state[:,1], # y
                        target_full_state[:,2], # z
                        'or', label='Target: Original', markersize=6)
                
                ax.plot(target_approximated_state_pos[:,0], # x
                        target_approximated_state_pos[:,1], # y
                        target_approximated_state_pos[:,2], # z
                        '-r', label='Target: Approx.', markersize=12, linewidth=3)

                ax.legend(fontsize=20)
                ax.tick_params(axis='both', which='major', labelsize=20)
                # ax.set_aspect('equal')
                set_axes_equal(ax)
                plt.show()
            # -----------------------------------------------------------------------
            
            # -----------------------------------------------------------------------
            # Check the approximation error thresholds
            print("--------------------------------------------------------------------")
            print("Approximation error thresholds: ", approximation_error_threshold)
            print("Initial state approximation error: ", initial_approximated_state_avg_error)
            print("Target state approximation error: ", target_approximated_state_avg_error)
            
            if (initial_approximated_state_avg_error > approximation_error_threshold):
                print("\nSkipping the current simplified_dlo_num_segments: ", simplified_dlo_num_segments)
                print("Reason: Initial state approximation error is greater than the threshold: ", initial_approximated_state_avg_error)
                continue
            if (target_approximated_state_avg_error > approximation_error_threshold):
                print("\nSkipping the current simplified_dlo_num_segments: ", simplified_dlo_num_segments)
                print("Reason: Target state approximation error is greater than the threshold: ", target_approximated_state_avg_error)
                continue
            # -----------------------------------------------------------------------
            
            # -----------------------------------------------------------------------
            # Check the environment limits for the initial and target states
            print("--------------------------------------------------------------------")
            print("Environment limits: ", environment_limits_xyz)
            print("Approximated initial state pos: \n", initial_approximated_state_pos)
            print("Approximated target state pos: \n", initial_approximated_state_pos)
            
            if (initial_approximated_state_pos[:,0].min() < environment_limits_xyz[0] or
                initial_approximated_state_pos[:,0].max() > environment_limits_xyz[1] or
                initial_approximated_state_pos[:,1].min() < environment_limits_xyz[2] or
                initial_approximated_state_pos[:,1].max() > environment_limits_xyz[3] or
                initial_approximated_state_pos[:,2].min() < environment_limits_xyz[4] or
                initial_approximated_state_pos[:,2].max() > environment_limits_xyz[5]):
                print("\nSkipping the current simplified_dlo_num_segments: ", simplified_dlo_num_segments)
                print("Reason: Initial state is out of the environment limits")
                continue
            if (target_approximated_state_pos[:,0].min() < environment_limits_xyz[0] or
                target_approximated_state_pos[:,0].max() > environment_limits_xyz[1] or
                target_approximated_state_pos[:,1].min() < environment_limits_xyz[2] or
                target_approximated_state_pos[:,1].max() > environment_limits_xyz[3] or
                target_approximated_state_pos[:,2].min() < environment_limits_xyz[4] or
                target_approximated_state_pos[:,2].max() > environment_limits_xyz[5]):
                print("\nSkipping the current simplified_dlo_num_segments: ", simplified_dlo_num_segments)
                print("Reason: Target state is out of the environment limits")
                continue
            # -----------------------------------------------------------------------
                
            # -----------------------------------------------------------------------
            # Check the joint angle limits for the initial and target states
            # INFO: Note that in joint pos, the first 3 elements are the translational joint angles (x, y, z)
            # # and the last 3N elements are the rotational joint angles around x, y and z axes for each segment respectively.
            # # But the first 3 angular joint angles are free and not limited.
            # # Therfore, 
            # # Every 3 indexes after the 6th index are the x joint angles
            # # Every 3 indexes after the 7th index are the y joint angles
            # # Every 3 indexes after the 8th index are the z joint angles
            print("--------------------------------------------------------------------")
            print("Joint angle limits [x_min, x_max] (deg): ", joint_angle_limits_xyz_deg[0], joint_angle_limits_xyz_deg[1])
            print("Joint angle limits [y_min, y_max] (deg): ", joint_angle_limits_xyz_deg[2], joint_angle_limits_xyz_deg[3])
            print("Joint angle limits [z_min, z_max] (deg): ", joint_angle_limits_xyz_deg[4], joint_angle_limits_xyz_deg[5])
            print("Approximated initial state X joint pos (deg): ", np.rad2deg(initial_approximated_state_joint_pos[6::3]))
            print("Approximated initial state Y joint pos (deg): ", np.rad2deg(initial_approximated_state_joint_pos[7::3]))
            print("Approximated initial state Z joint pos (deg): ", np.rad2deg(initial_approximated_state_joint_pos[8::3]))
            print("Approximated target state X joint pos (deg): ", np.rad2deg(target_approximated_state_joint_pos[6::3]))
            print("Approximated target state Y joint pos (deg): ", np.rad2deg(target_approximated_state_joint_pos[7::3]))
            print("Approximated target state Z joint pos (deg): ", np.rad2deg(target_approximated_state_joint_pos[8::3]))
            
            if len(initial_approximated_state_joint_pos) > 6: 
                if (initial_approximated_state_joint_pos[6::3].min() < np.deg2rad(joint_angle_limits_xyz_deg[0]) or
                    initial_approximated_state_joint_pos[6::3].max() > np.deg2rad(joint_angle_limits_xyz_deg[1]) or
                    initial_approximated_state_joint_pos[7::3].min() < np.deg2rad(joint_angle_limits_xyz_deg[2]) or
                    initial_approximated_state_joint_pos[7::3].max() > np.deg2rad(joint_angle_limits_xyz_deg[3]) or
                    initial_approximated_state_joint_pos[8::3].min() < np.deg2rad(joint_angle_limits_xyz_deg[4]) or
                    initial_approximated_state_joint_pos[8::3].max() > np.deg2rad(joint_angle_limits_xyz_deg[5])):
                    print("\nSkipping the current simplified_dlo_num_segments: ", simplified_dlo_num_segments)
                    print("Reason: Initial state is out of the joint angle limits")
                    continue
                
            if len(target_approximated_state_joint_pos) > 6:
                if (target_approximated_state_joint_pos[6::3].min() < np.deg2rad(joint_angle_limits_xyz_deg[0]) or
                    target_approximated_state_joint_pos[6::3].max() > np.deg2rad(joint_angle_limits_xyz_deg[1]) or
                    target_approximated_state_joint_pos[7::3].min() < np.deg2rad(joint_angle_limits_xyz_deg[2]) or
                    target_approximated_state_joint_pos[7::3].max() > np.deg2rad(joint_angle_limits_xyz_deg[3]) or
                    target_approximated_state_joint_pos[8::3].min() < np.deg2rad(joint_angle_limits_xyz_deg[4]) or
                    target_approximated_state_joint_pos[8::3].max() > np.deg2rad(joint_angle_limits_xyz_deg[5])):
                    print("\nSkipping the current simplified_dlo_num_segments: ", simplified_dlo_num_segments)
                    print("Reason: Target state is out of the joint angle limits")
                    continue
            # -----------------------------------------------------------------------

            # Now we need to find a way to add the robot and its SRDF to the environment

            # -----------------------------------------------------------------------
            # First let's create the URDF for the robot using the DloURDFCreator

            print("--------------------------------------------------------------------")
            
            # Parameters for the DLO URDF creator
            model_name="pole"
            # base_link_name="base_link"
            # tcp_link_name="tool0"
            center_link_name="center_link"
            holding_points_link_name_prefix= "pole_holder_link_"
            prism_joint_effort = 1
            rev_joint_effort = 1
            prism_joint_max_velocity = 1.5
            rev_joint_max_velocity = 1.0
            visualize=False

            dlo_urdf_creator = DloURDFCreator()
            
            stopwatch = Timer()
            stopwatch.start()
            # Create the URDF string for the robot
            urdf_str, is_valid_urdf, allowed_collision_pairs = dlo_urdf_creator.create_dlo_urdf_equal_segment_length(
                                                                    length = dlo_length,
                                                                    radius = dlo_radius,
                                                                    simplified_dlo_num_segments = simplified_dlo_num_segments,
                                                                    full_dlo_num_segments = full_dlo_num_segments,
                                                                    full_dlo_holding_segment_ids = full_dlo_holding_segment_ids,
                                                                    environment_limits_xyz = environment_limits_xyz,
                                                                    joint_angle_limits_xyz_deg = joint_angle_limits_xyz_deg,
                                                                    model_name = model_name,
                                                                    base_link_name = self.manip_info.working_frame,
                                                                    tcp_link_name = self.manip_info.tcp_frame,
                                                                    center_link_name = center_link_name,
                                                                    holding_points_link_name_prefix = holding_points_link_name_prefix,
                                                                    prism_joint_effort = prism_joint_effort,
                                                                    rev_joint_effort = rev_joint_effort,
                                                                    prism_joint_max_velocity = prism_joint_max_velocity,
                                                                    rev_joint_max_velocity = rev_joint_max_velocity,
                                                                    visualize = visualize)

            stopwatch.stop()
            urdf_generation_time = stopwatch.elapsedSeconds()
            print(f"URDF GENERATION TOOK {urdf_generation_time} SECONDS.")

            # print("URDF string from equal segment length:\n")
            # print(urdf_str + "\n")

            # print("Allowed collision pairs: ")
            # print(allowed_collision_pairs)

            # Next, We need to convert the URDF string into a scene graph
            scene_graph = parseURDFString(urdf_str, self.locator).release()

            # Then, we need to parse the SRDF file and create an SRDF model


            srdf_model = SRDFModel()
            srdf_model.initFile(scene_graph, str(self.srdf_fname), self.locator)

            # Process the SRDF allowed collisions in the robot scene graph 
            # (for a generic pole srdf it should be empty,)
            # (and we will process them manually based on the pairs obtained from the dlo urdf creator.)
            # (but let's keep it just in case we need to add some allowed collisions IN THE SRDF in the future)
            processSRDFAllowedCollisions(scene_graph, srdf_model)

            # Finally we need to create the commands to apply the robot to the environment
            cmds = Commands()

            add_scene_graph_command = AddSceneGraphCommand(scene_graph)
            cmds.push_back(add_scene_graph_command)

            # Add other SRDF related commands as well
            add_contact_managers_plugin_info_cmd = AddContactManagersPluginInfoCommand(srdf_model.contact_managers_plugin_info)
            cmds.push_back(add_contact_managers_plugin_info_cmd)

            add_kinematics_information_cmd = AddKinematicsInformationCommand(srdf_model.kinematics_information)
            cmds.push_back(add_kinematics_information_cmd)

            # Add the commands for the calibration information in the SRDF
            for cal in srdf_model.calibration_info.joints:
                change_joint_origin_cmd = ChangeJointOriginCommand(cal.first, cal.second)
                cmds.push_back(change_joint_origin_cmd)

            # Check srdf for collision margin data 
            if isinstance(srdf_model.collision_margin_data, CollisionMarginData):
                change_collision_margins_cmd = ChangeCollisionMarginsCommand(srdf_model.collision_margin_data, CollisionMarginOverrideType_REPLACE)
                cmds.push_back(change_collision_margins_cmd)

            # Apply the commands to the environment
            if not self.env.isInitialized():
                print("Environment is not initialized yet, let's initialize it")
                assert self.env.init(cmds)
            else:
                print("Environment is already initialized, let's apply the commands")
                self.env.applyCommands(cmds)
                
            # Update the viewer
            if self.viewer_enabled:
                self.viewer.update_environment(self.env, [0,0,0])
            # -----------------------------------------------------------------------

            # -----------------------------------------------------------------------
            # Add allowed collision pairs to the environment obtained from the dlo urdf creator
            acm = AllowedCollisionMatrix()

            for pair in allowed_collision_pairs:
                acm.addAllowedCollision(pair[0], pair[1], pair[2]) # link1, link2, reason
                
            modify_allowed_collisions_command = ModifyAllowedCollisionsCommand(acm, ModifyAllowedCollisionsType_ADD)

            cmds = Commands()
            cmds.push_back(modify_allowed_collisions_command)
            self.env.applyCommands(cmds)

            # May not be necessary but:
            if self.viewer_enabled:
                self.viewer.update_environment(self.env, [0,0,0])
            # -----------------------------------------------------------------------

            # -----------------------------------------------------------------------
            # Get the kinematic group from the environment, needed for forward kinematics
            kin_group = self.env.getKinematicGroup(self.manip_info.manipulator)
            # -----------------------------------------------------------------------
            
            # -----------------------------------------------------------------------
            # Get the joint names from the environment:
            joint_group = self.env.getJointGroup(self.manip_info.manipulator)
            joint_names = list(joint_group.getJointNames())

            # Another way to get the joint names:
            # self.joint_names = list(self.env.getGroupJointNames("manipulator"))

            # print("joint_names: ", joint_names)
            # print("")
            # -----------------------------------------------------------------------

            # -----------------------------------------------------------------------
            # Check the goal state for collisions
            goal_joint_positions = target_approximated_state_joint_pos.flatten()
            
            is_collision_free, new_joint_pos = self.check_and_try_remove_collision_state(joint_names, goal_joint_positions)
            
            if not is_collision_free:
                print("The goal state is in collision. Skipping the current simplified_dlo_num_segments.")
                continue
            else:
                print("The goal state is collision free.")
                goal_joint_positions = new_joint_pos                    
            # -----------------------------------------------------------------------
            
            # -----------------------------------------------------------------------
            # Set GOAL STATE of the robot in the environment
            # This state is obtained from the dlo simulator and passed to the planner the controller
            
            # Set the initial state of the robot in the environment
            self.env.setState(joint_names, goal_joint_positions)

            if self.viewer_enabled:
                # viewer.update_environment(self.env, [0,0,0])
                self.viewer.update_joint_positions(joint_names, goal_joint_positions)
            # -----------------------------------------------------------------------    
                    
            # input("Press enter to confirm the goal state")

            # -----------------------------------------------------------------------
            # Check the initial state for collisions
            initial_joint_positions = initial_approximated_state_joint_pos.flatten()

            is_collision_free, new_joint_pos = self.check_and_try_remove_collision_state(joint_names, initial_joint_positions)
            
            if not is_collision_free:
                print("The initial state is in collision. Skipping the current simplified_dlo_num_segments.")
                continue
            else:
                print("The initial state is collision free.")
                initial_joint_positions = new_joint_pos
            # -----------------------------------------------------------------------
            
            # -----------------------------------------------------------------------
            # Set INITIAL STATE of the robot in the environment
            # This state is obtained from the dlo simulator and passed to the planner the controller
            
            # Set the initial state of the robot in the environment
            self.env.setState(joint_names, initial_joint_positions)

            if self.viewer_enabled:
                # viewer.update_environment(self.env, [0,0,0])
                self.viewer.update_joint_positions(joint_names, initial_joint_positions)
            # -----------------------------------------------------------------------
                
            # input("Press enter to confirm the Start state and initiate the planning")

            # -----------------------------------------------------------------------

            # Print a message that says the the approximation with the simplified dlo num segments is valid within the thresholds
            print("The approximation with ", simplified_dlo_num_segments, " segments is valid within the thresholds.")
            print(f"DLO SIMPLIFICATION TOOK {dlo_simplification_time} SECONDS IN TOTAL.")
            print("Creating the URDF for the robot with the DloURDFCreator")

            # We will now start the planning from the initial state to the goal state!!!    

            # -----------------------------------------------------------------------
            # Create a list of StateWaypointPoly
            state_waypoints = []

            # Add the initial state waypoint
            initial_state_waypoint = StateWaypointPoly_wrap_StateWaypoint(StateWaypoint(joint_names, initial_joint_positions))
            state_waypoints.append(initial_state_waypoint)

            # # Add the intermediate waypoints 
            # intermediate_state_waypoint = StateWaypointPoly_wrap_StateWaypoint(StateWaypoint(joint_names, intermediate_joint_positions))
            # state_waypoints.append(intermediate_state_waypoint)

            # Add the goal state waypoint
            goal_state_waypoint = StateWaypointPoly_wrap_StateWaypoint(StateWaypoint(joint_names, goal_joint_positions))
            state_waypoints.append(goal_state_waypoint)
            # -----------------------------------------------------------------------

            # -----------------------------------------------------------------------
            # Set move instructions from the state waypoints
            move_instructions = []
                    
            for state_waypoint in state_waypoints:
                move_instruction = MoveInstructionPoly_wrap_MoveInstruction(MoveInstruction(state_waypoint, 
                                                                                            MoveInstructionType_FREESPACE, 
                                                                                            "DEFAULT"))
                move_instructions.append(move_instruction)
            # -----------------------------------------------------------------------

            # -----------------------------------------------------------------------
            # Create input command program

            # Create the input command program using CompositeInstruction
            program = CompositeInstruction("DEFAULT")
            # program = CompositeInstruction("freespace_profile")
            program.setManipulatorInfo(self.manip_info)
                
            # Add the MoveInstructionPoly objects to the CompositeInstruction
            for move_instruction in move_instructions:
                program.appendMoveInstruction(move_instruction)
                
            # Print diagnosics
            program._print("Program: ")

            ## Create an AnyPoly containing the program. 
            # This explicit step is required because the Python bindings do not
            # support implicit conversion from the CompositeInstruction to the AnyPoly.
            program_anypoly = AnyPoly_wrap_CompositeInstruction(program)
            # -----------------------------------------------------------------------

            # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            # ++++++++++++++++++++++++++++++++++++++++++++ OMPL PLANNING ++++++++++++++++++++++++++++++++++++++++++++++++++++++
            # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

            # --------------------------------------------------------------------------------------------
            # Create the task composer node. In this case the FreespacePipeline is used. Many other are available.
            task = self.factory.createTaskComposerNode("OMPLPipelineAlone")
            
            # Get the output keys for the task
            task_output_key = task.getOutputKeys()[0]

            # Create an executor to run the task
            task_executor = self.factory.createTaskComposerExecutor("TaskflowExecutor")
            # --------------------------------------------------------------------------------------------

            # -----------------------------------------------------------------------
            # Set profile dictionary 
            # Create a profile dictionary. Profiles can be customized by adding to this dictionary and setting the profiles
            # in the instructions.

            profiles = ProfileDictionary()

            add_MinLengthProfile(profiles, "DEFAULT", length=40)
            add_OMPLDefaultPlanProfile(profiles, "DEFAULT")
            # add_TrajOptPlanProfile(profiles, "DEFAULT", len(initial_joint_positions))
            # -----------------------------------------------------------------------

            # -----------------------------------------------------------------------
            # Create the task problem and 
            print("Creating the task planning problem..")

            task_planning_problem = PlanningTaskComposerProblem(self.env, profiles)        
            task_planning_problem.input = program_anypoly
            # task_planning_problem.input = program

            print("Task planning problem created.")
            # -----------------------------------------------------------------------
            
            # -----------------------------------------------------------------------
            # Solve task
            print("Planning the task..")
            stopwatch = Timer()
            stopwatch.start()

            # Run the task and wait for completion
            future = task_executor.run(task.get(), task_planning_problem)
            future.wait()

            stopwatch.stop()
            planning_time_ompl = stopwatch.elapsedSeconds()
            print(f"OMPL PLANNING TOOK {planning_time_ompl} SECONDS.")

            # -----------------------------------------------------------------------
            
            # --------------------------------------------------------------------------------------------
            # Get the results of the global plan
            try:
                # Retrieve the output, converting the AnyPoly back to a CompositeInstruction
                results_as_any_poly = future.context.data_storage.getData(task_output_key)
                results = AnyPoly_as_CompositeInstruction(results_as_any_poly)
            except Exception:
                ## Assertions failed, fall back to the fallback plan
                print("Error getting the results of the OMPL plan")
                print("{}".format(traceback.format_exc()))
            #     planning_success = 0
                
            if len(results) < 2:
                planning_success = 0
                print("OMPL: Path length is less than 2. Planning failed!!")
                print("Try the next simplified_dlo_num_segments.")
                continue
            # --------------------------------------------------------------------------------------------

            # --------------------------------------------------------------------------------------------
            # Plan sanity check
            try:
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
            except Exception:
                planning_success = 0
                ## Assertions failed, fall back to the fallback plan
                print("Error getting the results of the global plan")
                print("{}".format(traceback.format_exc()))
                # print("Using the fallback plan")
                # # results = results_fallback # TODO
                print("Try the next simplified_dlo_num_segments.")
                continue
            # --------------------------------------------------------------------------------------------
            
            # --------------------------------------------------------------------------------------------
            # Visualize the results in the viewer
            if self.viewer_enabled:
                try:
                    # Update the viewer with the results to animate the trajectory
                    # Open web browser to http://localhost:8000 to view the results
                    self.viewer.clear_all_markers()
                    self.viewer.update_trajectory(results)
                    self.viewer.plot_trajectory(results, self.manip_info)
                except Exception:
                    print("Error updating the viewer with the results:")
                    print("{}".format(traceback.format_exc()))
            # # --------------------------------------------------------------------------------------------
            
            # --------------------------------------------------------------------------------------------
            # Process the results of the OMPL planner, e.g. check the path length, etc.
            try:
                (
                ompl_path,
                ompl_path_points,
                ompl_path_cumulative_lengths,
                ompl_path_cumulative_rotations,
                ompl_path_direction_vectors,
                ompl_path_rotation_vectors, 
                ompl_path_of_particles,
                ompl_path_points_of_particles,
                ompl_path_cumulative_lengths_of_particles,
                ompl_path_cumulative_rotations_of_particles,
                ompl_path_direction_vectors_of_particles,
                ompl_path_rotation_vectors_of_particles,
                ompl_path_approximated_dlo_joint_values
                ) = self.process_planner_results(results, 
                                                center_link_name, 
                                                holding_points_link_name_prefix, 
                                                full_dlo_holding_segment_ids,
                                                kin_group)
                    
                # Calculate the ompl_path length as average of the ompl_path lengths of the centroid and the holding points
                
                ompl_path_length = ompl_path_cumulative_lengths[-1] # Use this if the centroid is also included
                i = 1.0 # Use this if the centroid is also included 
                
                # ompl_path_length = 0.0 # Initialize the ompl_path length # Use this if the centroid is NOT included
                # i = 0.0 # Use this if the centroid is NOT included
                
                for id in full_dlo_holding_segment_ids:
                    ompl_path_length += ompl_path_cumulative_lengths_of_particles[id][-1]
                    i += 1.0
                ompl_path_length = ompl_path_length / i # Average ompl_path length of the centroid and the holding points
            except Exception:
                print("Error processing the results of the OMPL planner")
                # print("{}".format(traceback.format_exc()))
                print("Try the next simplified_dlo_num_segments.")
                continue
            # --------------------------------------------------------------------------------------------
                
            # input("Press enter to continue to the TrajOpt planner for smoothing")
    
            # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            # +++++++++++++++++++++++++++++++++++++++++++ TRAJOPT PLANNING ++++++++++++++++++++++++++++++++++++++++++++++++++++
            # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

            # --------------------------------------------------------------------------------------------
            # Prepare the previous results for the TrajOpt planner
            joint_waypoints = []
            try:
                for i, instr in enumerate(results):
                    assert instr.isMoveInstruction()
                    move_instr1 = InstructionPoly_as_MoveInstructionPoly(instr)
                    wp1 = move_instr1.getWaypoint()
                    assert wp1.isStateWaypoint()
                    wp = WaypointPoly_as_StateWaypointPoly(wp1)
                    # print("-------------------------------------------------------------")
                    # print(f"Joint Time: {wp.getTime()}")
                    # print(f"Joint Positions: {wp.getPosition().flatten()}")
                    # print("Joint Names: " + str(list(wp.getNames())))
                    # print(f"Joint Velocities: {wp.getVelocity().flatten()}")
                    # print(f"Joint Accelerations: {wp.getAcceleration().flatten()}")
                    # print(f"Joint Efforts: {wp.getEffort().flatten()}")
                    # print("-------------------------------------------------------------")
                    
                    # Joint waypoint
                    jw = JointWaypoint()
                    jw.setNames(wp.getNames())
                    # jw.setNames(list(wp.getNames()))
                    
                    jw.setPosition(wp.getPosition())
                    # jw.setPosition(wp.getPosition().flatten())
                    
                    if i == 0 or i == len(results)-1:
                        jw.setIsConstrained(True) # Constrain the start and end waypoints
                    else:
                        jw.setIsConstrained(False) # Free waypoints
                    
                    # Joint waypoint poly
                    jwp = JointWaypointPoly_wrap_JointWaypoint(jw)
                    joint_waypoints.append(jwp)
                    
            except Exception:
                planning_success = 0
                print("Error preparing the OMPL results for the TrajOpt plan")
                print("{}".format(traceback.format_exc()))
                print("Try the next simplified_dlo_num_segments.")
                continue
                
            # Set move instructions from the joint waypoints
            move_instructions = []
            for jwp in joint_waypoints:
                mi = MoveInstruction(jwp, MoveInstructionType_FREESPACE, "DEFAULT")
                move_instructions.append(MoveInstructionPoly_wrap_MoveInstruction(mi))
                
            # Create the input command program using CompositeInstruction
            program = CompositeInstruction("DEFAULT")
            program.setManipulatorInfo(self.manip_info)
            
            # Add the MoveInstructionPoly objects to the CompositeInstruction
            for move_instruction in move_instructions:
                program.appendMoveInstruction(move_instruction)
                
            # # Print diagnosics
            # program._print("Program for TrajOpt: ")
            
            ## Create an AnyPoly containing the program.
            program_anypoly = AnyPoly_wrap_CompositeInstruction(program)
            # --------------------------------------------------------------------------------------------

            # --------------------------------------------------------------------------------------------
            # Create the task composer node. In this case the FreespacePipeline is used. Many other are available.
            # task = factory.createTaskComposerNode("OMPLPipelineAlone")
            task = self.factory.createTaskComposerNode("TrajOptPipelineAlone")

            # Get the output keys for the task
            task_output_key = task.getOutputKeys()[0]

            # Create an executor to run the task
            task_executor = self.factory.createTaskComposerExecutor("TaskflowExecutor")
            # --------------------------------------------------------------------------------------------

            # -----------------------------------------------------------------------
            # Set profile dictionary 
            # Create a profile dictionary. Profiles can be customized by adding to this dictionary and setting the profiles
            # in the instructions.

            profiles = ProfileDictionary()

            add_MinLengthProfile(profiles, "DEFAULT", length=70)
            # add_OMPLDefaultPlanProfile(profiles, "DEFAULT")
            add_TrajOptPlanProfile(profiles, "DEFAULT", len(initial_joint_positions))
            # -----------------------------------------------------------------------

            # -----------------------------------------------------------------------
            # Create the task problem and 
            print("Creating the task planning problem..")

            task_planning_problem = PlanningTaskComposerProblem(self.env, profiles)        
            task_planning_problem.input = program_anypoly        
            # task_planning_problem.input = results_as_any_poly        

            print("Task planning problem created.")
            # -----------------------------------------------------------------------

            # -----------------------------------------------------------------------
            # Solve task
            print("Planning the TRAJOPT task..")
            stopwatch = Timer()
            stopwatch.start()

            # Run the task and wait for completion
            future = task_executor.run(task.get(), task_planning_problem)
            future.wait()

            stopwatch.stop()
            planning_time_trajopt = stopwatch.elapsedSeconds()
            print(f"TRAJOPT PLANNING TOOK {planning_time_trajopt} SECONDS.")
            # -----------------------------------------------------------------------

            # --------------------------------------------------------------------------------------------
            # Get the results of the global plan
            try:
                # Retrieve the output, converting the AnyPoly back to a CompositeInstruction
                results_as_any_poly = future.context.data_storage.getData(task_output_key)
                results = AnyPoly_as_CompositeInstruction(results_as_any_poly)
            except Exception:
                planning_success = 0
                ## Assertions failed, fall back to the fallback plan
                print("Error getting the results of the TRAJOPT plan")
                print("{}".format(traceback.format_exc()))
                print("Try the next simplified_dlo_num_segments.")
                continue
                
            if len(results) < 2:
                planning_success = 0
                print("TrajOpt: Path length is less than 2. Planning failed!!")
                print("Try the next simplified_dlo_num_segments.")
                continue
            # --------------------------------------------------------------------------------------------

            # --------------------------------------------------------------------------------------------
            # Plan sanity check
            try:
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
            except Exception:
                planning_success = 0
                ## Assertions failed, fall back to the fallback plan
                print("Error getting the results of the global plan")
                print("{}".format(traceback.format_exc()))
                # print("Using the fallback plan")
                # results = results_fallback # TODO
                print("Try the next simplified_dlo_num_segments.")
                continue
            # --------------------------------------------------------------------------------------------
            
            # --------------------------------------------------------------------------------------------
            # Visualize the results in the viewer
            if self.viewer_enabled:
                try:
                    # Update the viewer with the results to animate the trajectory
                    # Open web browser to http://localhost:8000 to view the results
                    self.viewer.clear_all_markers()
                    self.viewer.update_trajectory(results)
                    self.viewer.plot_trajectory(results, self.manip_info)
                except Exception:
                    print("Error updating the viewer with the results:")
                    print("{}".format(traceback.format_exc()))
            # # --------------------------------------------------------------------------------------------
            
            # --------------------------------------------------------------------------------------------
            # Process the results of the TrajOpt planner, e.g. check the path length, etc.
            try:
                (
                trajopt_path,
                trajopt_path_points,
                trajopt_path_cumulative_lengths,
                trajopt_path_cumulative_rotations,
                trajopt_path_direction_vectors,
                trajopt_path_rotation_vectors, 
                trajopt_path_of_particles,
                trajopt_path_points_of_particles,
                trajopt_path_cumulative_lengths_of_particles,
                trajopt_path_cumulative_rotations_of_particles,
                trajopt_path_direction_vectors_of_particles,
                trajopt_path_rotation_vectors_of_particles,
                trajopt_path_approximated_dlo_joint_values
                ) = self.process_planner_results(results, 
                                                center_link_name, 
                                                holding_points_link_name_prefix, 
                                                full_dlo_holding_segment_ids,
                                                kin_group)
                    
                # Calculate the trajopt_path length as average of the trajopt_path lengths of the centroid and the holding points
                
                trajopt_path_length = trajopt_path_cumulative_lengths[-1] # Use this if the centroid is also included
                i = 1.0 # Use this if the centroid is also included
                
                # trajopt_path_length = 0.0 # Initialize the ompl_path length # Use this if the centroid is NOT included
                # i = 0.0 # Use this if the centroid is NOT included
                
                for id in full_dlo_holding_segment_ids:
                    trajopt_path_length += trajopt_path_cumulative_lengths_of_particles[id][-1]
                    i += 1.0
                trajopt_path_length = trajopt_path_length / i # Average path length of the centroid and the holding points
            except Exception:
                print("Error processing the results of the TrajOpt planner")
                # print("{}".format(traceback.format_exc()))
                print("Try the next simplified_dlo_num_segments.")
                continue
            # --------------------------------------------------------------------------------------------
            
            # --------------------------------------------------------------------------------------------
            # OPTIONAL: Plot the path points
            if plot_for_debugging:
                try:
                    ax = plt.figure().add_subplot(projection='3d')

                    # Add title with the number of segments
                    ax.set_title("Generated Paths of Centroid and Holding Points\n w/ Number of Segments = " + str(simplified_dlo_num_segments), fontsize=30)

                    ax.plot(initial_full_state[:,0], # x
                            initial_full_state[:,1], # y
                            initial_full_state[:,2], # z
                            # 'og', label='Initial State: Original', markersize=10, fillstyle='none')
                            'Xg', label='Initial State: Original Centers', markersize=10,  mec = 'k', alpha=.5)
                    
                    ax.plot(initial_approximated_state_pos[:,0], # x
                            initial_approximated_state_pos[:,1], # y
                            initial_approximated_state_pos[:,2], # z
                            '-g', label='Initial State: Approximation Line', markersize=12, linewidth=8)
                            # '-g', label='Initial State: Approximation', markersize=12, linewidth=6, alpha=.5)

                    ax.plot(target_full_state[:,0], # x
                            target_full_state[:,1], # y
                            target_full_state[:,2], # z
                            # 'or', label='Target State: Original', markersize=10, fillstyle='none')
                            'Xr', label='Target State: Original Centers', markersize=10,  mec = 'k', alpha=.5)
                    
                    ax.plot(target_approximated_state_pos[:,0], # x
                            target_approximated_state_pos[:,1], # y
                            target_approximated_state_pos[:,2], # z
                            '-r', label='Target State: Approximation Line', markersize=12, linewidth=8)
                            # '-r', label='Target State: Approximation', markersize=12, linewidth=6, alpha=.5)
                    
                    # # Plot the centroid path points before smoothing
                    # ax.plot(ompl_path_points[:,0], # x
                    #         ompl_path_points[:,1], # y
                    #         ompl_path_points[:,2], # z
                    #         ':ok', label='Centroid Path (before smoothing)', markersize=2, linewidth=1)
                    
                    path_colors = ['b', 'm', 'c', 'y', 'k', 'g', 'r']
                    
                    # # Plot the holding points path points before smoothing
                    # i = 0
                    # for id in full_dlo_holding_segment_ids:
                    #     ax.plot(ompl_path_points_of_particles[id][:,0], # x
                    #             ompl_path_points_of_particles[id][:,1], # y
                    #             ompl_path_points_of_particles[id][:,2], # z
                    #             ':o'+path_colors[i], label='Point ' + str(id) + ' Path (before smoothing)', markersize=2, linewidth=1)
                    #     i += 1
                        
                    # # Plot the centroid path points after smoothing
                    # ax.plot(trajopt_path_points[:,0], # x
                    #         trajopt_path_points[:,1], # y
                    #         trajopt_path_points[:,2], # z
                    #         '--^k', label='Centroid Path (after smoothing)', markersize=4, linewidth=2)
                    
                    # Plot the holding points path points after smoothing
                    i = 0
                    for id in full_dlo_holding_segment_ids:
                        ax.plot(trajopt_path_points_of_particles[id][:,0], # x
                                trajopt_path_points_of_particles[id][:,1], # y
                                trajopt_path_points_of_particles[id][:,2], # z
                                '--^'+path_colors[i], label='Point ' + str(id) + ' Path (after smoothing)', markersize=4, linewidth=2)
                        i += 1
                        
                    # Gradient from green (initial approximation) to red (target approximation)
                    num_intermediate_lines = len(trajopt_path_approximated_dlo_joint_values)
                    color_gradient = cm.RdYlGn(np.linspace(1, 0, num_intermediate_lines))  # Colormap from green to red

                    # Plot the approximated DLO segments during the path using the joint values (after smoothing, ie. trajopt)
                    for i, joint_values in enumerate(trajopt_path_approximated_dlo_joint_values):
                        polyline = dlo_fwd_kin(joint_pos=joint_values, dlo_l=dlo_length, return_rot_matrices=False)  # intermediate approximated state
                        ax.plot(polyline[:,0], # x
                                polyline[:,1], # y
                                polyline[:,2], # z
                                '-', color=color_gradient[i], markersize=6, linewidth=4, alpha=.2)
                            
                    ax.legend(fontsize=20)
                    ax.tick_params(axis='both', which='major', labelsize=20)
                    # ax.set_aspect('equal')
                    set_axes_equal(ax)
                    plt.show()
                except Exception:
                    print("Error plotting the path points")
                    # print("{}".format(traceback.format_exc()))
            # --------------------------------------------------------------------------------------------
            
            # --------------------------------------------------------------------------------------------
            print("Planning with ", simplified_dlo_num_segments, " segments is completed.")    
    
            # Calculate the total planning time
            total_planning_time = 0.0
            if planning_time_ompl:
                total_planning_time += planning_time_ompl
            if planning_time_trajopt:
                total_planning_time += planning_time_trajopt
            
            print("Total planning time: ", total_planning_time, " seconds.")
            print("--------------------------------------------------------------------")
    
            # Calculate max of initial_approximated_state_avg_error and target_approximated_state_avg_error
            avr_state_approx_error = max(initial_approximated_state_avg_error, target_approximated_state_avg_error)
            
            # Data
            # Variables to save to the pickle file:
            plan_data_ompl = (ompl_path, ompl_path_points, ompl_path_cumulative_lengths, ompl_path_cumulative_rotations,
                            ompl_path_direction_vectors, ompl_path_rotation_vectors, ompl_path_of_particles,
                            ompl_path_points_of_particles, ompl_path_cumulative_lengths_of_particles,
                            ompl_path_cumulative_rotations_of_particles, ompl_path_direction_vectors_of_particles,
                            ompl_path_rotation_vectors_of_particles, ompl_path_approximated_dlo_joint_values)
            
            plan_data_trajopt = (trajopt_path, trajopt_path_points, trajopt_path_cumulative_lengths, trajopt_path_cumulative_rotations,
                                trajopt_path_direction_vectors, trajopt_path_rotation_vectors, trajopt_path_of_particles,
                                trajopt_path_points_of_particles, trajopt_path_cumulative_lengths_of_particles,
                                trajopt_path_cumulative_rotations_of_particles, trajopt_path_direction_vectors_of_particles,
                                trajopt_path_rotation_vectors_of_particles, trajopt_path_approximated_dlo_joint_values)
            
            # Note: Below the first element is a placeholder for the experiment id
            performance_data = (np.NaN, simplified_dlo_num_segments, avr_state_approx_error, planning_success,
                                planning_time_ompl, planning_time_trajopt, total_planning_time, ompl_path_length, trajopt_path_length) 
            
            initial_n_target_states = (initial_full_state, initial_approximated_state_pos, initial_approximated_state_joint_pos, 
                                    target_full_state, target_approximated_state_pos, target_approximated_state_joint_pos)
            
            performance_data_extra = (dlo_simplification_time, urdf_generation_time)
            
            # Create object to save to the pickle file
            plan_data = [plan_data_ompl, plan_data_trajopt, performance_data, initial_n_target_states, performance_data_extra]
            
            if return_all_data:
                return plan_data            
            else:
                return (trajopt_path, trajopt_path_points, trajopt_path_cumulative_lengths, trajopt_path_cumulative_rotations,
                        trajopt_path_direction_vectors, trajopt_path_rotation_vectors, trajopt_path_of_particles,
                        trajopt_path_points_of_particles, trajopt_path_cumulative_lengths_of_particles,
                        trajopt_path_cumulative_rotations_of_particles, trajopt_path_direction_vectors_of_particles,
                        trajopt_path_rotation_vectors_of_particles)
    
    # -----------------------------------------------------------------------
    def load_collision_scene(self, json_file_path):
        cmds = Commands()
        if (not json_file_path or 
            not os.path.exists(json_file_path) or 
            not os.path.isfile(json_file_path) or 
            not json_file_path.endswith('.json') or 
            not os.path.getsize(json_file_path) > 0):
            
            # Create an empty scene graph
            scene_graph_to_add = SceneGraph()
            scene_graph_to_add.addLink(Link("world_frame"))
            
            add_scene_graph_command = AddSceneGraphCommand(scene_graph_to_add)
            
            cmds.push_back(add_scene_graph_command)
        else:
            scene_urdf_str = json_to_urdf(input_file_path=json_file_path, 
                                            visualize=False, 
                                            save_output=False,
                                            output_file_path="./tesseract_scene.urdf")
            # print("scene_urdf_str: ")
            # print(scene_urdf_str)

            scene_graph_to_add = parseURDFString(scene_urdf_str, self.locator).release()
            add_scene_graph_command = AddSceneGraphCommand(scene_graph_to_add)

            cmds.push_back(add_scene_graph_command)

        if not self.env.isInitialized():
            print("Environment is not initialized yet, let's initialize it")
            assert self.env.init(cmds)
            
        else:
            print("Environment is already initialized, let's apply the commands")
            self.env.applyCommands(cmds)
            
        # Get the root link name
        print("Scene graph root name: ", self.env.getSceneGraph().getRoot())
    # -----------------------------------------------------------------------
    
    # -----------------------------------------------------------------------
    def read_state_dict_from_csv(self, file):
        # Load the data from the CSV file
        df = pd.read_csv(file)
        
        # Convert DataFrame to dictionary
        state_dict = df.to_dict(orient='list')
        return state_dict
        
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
    
    def check_and_try_remove_collision_state(self,joint_names, joint_positions, coll_depth_to_try_remove=0.007):
        new_joint_positions = joint_positions
        
        result_vector = self.check_for_collisions(joint_names, joint_positions)

        # Print the number of contacts
        if len(result_vector) > 0:
            print("Number of contacts: ", len(result_vector))
            print("Distances of the contacts: ")
            
            # print all distances and check if the state is in deep collision
            for contact in result_vector:
                print("Distance: ", contact.distance)
                if contact.distance < -coll_depth_to_try_remove:
                    # If the penetration depth is greater than the threshold, then the goal state is not valid
                    print("The state is in deep collision.")
                    # return is_collision_free, new_joint_positions
                    return False, new_joint_positions
                
            # Otherwise as a quick fix, we will move the goal state up by collision depth along the normal of the contact
            # This is a quick fix and may not be the best solution
            print("The goal state is in collision by a small margin. Moving the goal state up by the penetration depth.")
            
            for contact in result_vector:
                # print("Normal of the contact: ", result_vector[0].normal.flatten())
                new_joint_positions[0:3] += contact.normal.flatten() * (contact.distance*1.02)
                
                # Check if the new state is collision free
                inner_result_vector = self.check_for_collisions(joint_names, new_joint_positions)
                if len(inner_result_vector) == 0:
                    print("Collision is removal is successful.")
                    # return is_collision_free, new_joint_positions
                    return True, new_joint_positions
                else:
                    # continue to the next contact
                    continue
            
            # If the new state is still in collision, then the goal state is not valid
            print("The state is still in collision after moving up by the penetration depth of each contact.")
            return False, new_joint_positions
        else:
            # return is_collision_free, new_joint_positions
            return True, new_joint_positions
        
    def check_for_collisions(self, joint_names, joint_positions):
        # Get the state solver. This must be called again after environment is updated
        solver = self.env.getStateSolver()
        
        # Get the discrete contact manager. This must be called again after the environment is updated
        manager = self.env.getDiscreteContactManager()
        
        manager.setActiveCollisionObjects(self.env.getActiveLinkNames())
        
        manager.setCollisionMarginData(CollisionMarginData(0.0)) # Does not replace the environment's collision margin data, so it is safe to use
        
        # # Print the max collision margin of the contact manager 
        # print("Max collision margin of the contact manager:", 
        #         manager.getCollisionMarginData().getMaxCollisionMargin())
        
        # Set the transform of the active collision objects from SceneState
        solver.setState(joint_names, joint_positions)
        scene_state = solver.getState()
        manager.setCollisionObjectsTransform(scene_state.link_transforms)
        
        # Perform collision check
        result = ContactResultMap()
        manager.contactTest(result, ContactRequest(ContactTestType_CLOSEST))
        result_vector = ContactResultVector()
        result.flattenMoveResults(result_vector)
        
        return result_vector
    # -----------------------------------------------------------------------
    
    # -----------------------------------------------------------------------
    # Functions related to processing the planner results
    def tesseract_trajectory_to_joint_values_list(self, tesseract_trajectory):
        """
            Convert the tesseract trajectory to a list of joint values
        
        Args:
            tesseract_trajectory (CompositeInstruction): The results of the planner
        
        Returns:
            joint_values (list): A list of lists where each inner list contains the joint values for a waypoint
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

    def joint_values_to_planned_path_data(self, joint_values, frame_names, kin_group):
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
        
        # kin = self.env.getKinematicGroup(self.manip_info.manipulator)
        
        for i in range(len(joint_values)):
            frames = kin_group.calcFwdKin(np.asarray(joint_values[i], dtype=np.float64))
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
                    
                    # Wrap the angle to [-pi, pi]
                    rotation_angle = np.arctan2(np.sin(rotation_angle), np.cos(rotation_angle))

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
        
        # Wrap the angle to [-pi, pi]
        angle = np.arctan2(np.sin(angle), np.cos(angle))

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
        
    def process_planner_results(self, 
                                results, 
                                center_link_name, 
                                holding_points_link_name_prefix, 
                                full_dlo_holding_segment_ids,
                                kin_group):
        """
        Centroid Related:
        - path ([])                                   : a list of PoseStamped() msgs 
        
        - path_points ([])                            : planned path of centroid as Nx3 xyz points (N: num of waypoints in the path),
                                                        as numpy array 
        - path_cumulative_lengths ([0.0])             : cumulative lengths of the centroid path segments as list of floats
        
        - path_cumulative_rotations ([0.0])           : cumulative rotations of the centroid path segments obtained from the angles 
                                                        in axis-angle representation consecutive rotations in radians
        - path_direction_vectors ([])                 : directions of the centroid path segments as list of np array unit vectors
        
        - path_rotation_vectors ([])                  : rotation axes of the centroid path segments as list of np array unit vectors
        
        Holding Points Related:
        - path_of_particles ({})                      : dict of planned path of the particles as a list of Pose() msgs, 
                                                        keyed by the particle ids
        - path_points_of_particles ({})               : dict of planned path of the particles each as a Nx3 xyz points numpy array
                                                        (N: num of waypoints in the path), keyed by the particle ids
        - path_cumulative_lengths_of_particles ({})   : dict of cumulative lengths of the path segments of the particles as list of floats,
                                                        keyed by the particle ids
        - path_cumulative_rotations_of_particles ({}) : dict of cumulative rotations of the path segments of the particles as list of floats,
                                                        obtained from the angles in axis-angle representation consecutive rotations in radians
        - path_direction_vectors_of_particles ({})    : dict of directions of the path segments of the particles as unit vectors 
                                                        (list of N elements with each element is a 3D unit vector for each particle)
        - path_rotation_vectors_of_particles ({})     : dict of rotation axes of the path segments of the particles as unit vectors 
                                                        (list of N elements with each element is a 3D unit vector for each particle)
                                                        
        - path_approximated_dlo_joint_values ([])      : a list of joint values of the approximated dlo for each waypoint
                                                        
        Args:
            - results (CompositeInstruction): The results of the planner
            - center_link_name (str): The name of the center link of the dlo
            - holding_points_link_name_prefix (str): The prefix of the link names of the holding points
            - full_dlo_holding_segment_ids (list): The full dlo holding segment ids from the full_dlo_holding_segment_ids
            - kin_group (KinematicGroup): The kinematic group of the manipulator, used for forward kinematics
        """
        # To shorten the variable names
        c_link = center_link_name
        h_pts_pre = holding_points_link_name_prefix
        h_seg_ids = full_dlo_holding_segment_ids
        
        # Get the joint values of each waypoint from the results
        path_approximated_dlo_joint_values = self.tesseract_trajectory_to_joint_values_list(results)
        
        # Frame names that we are interested in as the path points
        frame_names = [c_link] + [h_pts_pre + str(i) for i in h_seg_ids]
        
        ## Convert the joint values to the planned path data
        (
        path_of_frames, 
        path_points_of_frames,
        path_cumulative_lengths_of_frames,
        path_cumulative_rotations_of_frames,
        path_direction_vectors_of_frames,
        path_rotation_vectors_of_frames
        ) = self.joint_values_to_planned_path_data(path_approximated_dlo_joint_values, 
                                            frame_names,
                                            kin_group)
        
        ## Convert the joint values to the planned path data for the particles
        frame_names.remove(c_link)
        path = path_of_frames.pop(c_link)
        path_points = path_points_of_frames.pop(c_link)
        path_cumulative_lengths = path_cumulative_lengths_of_frames.pop(c_link)
        path_cumulative_rotations = path_cumulative_rotations_of_frames.pop(c_link)
        path_direction_vectors = path_direction_vectors_of_frames.pop(c_link)
        path_rotation_vectors = path_rotation_vectors_of_frames.pop(c_link)
        
        ## Assign the keys of the dictionaries to the custom static particle ids
        path_of_particles = {p_id: path_of_frames.get(h_pts_pre + str(p_id), None) 
                                        for p_id in h_seg_ids}
        path_points_of_particles = {p_id: path_points_of_frames.get(h_pts_pre + str(p_id), None) 
                                        for p_id in h_seg_ids}
        path_cumulative_lengths_of_particles = {p_id: path_cumulative_lengths_of_frames.get(h_pts_pre + str(p_id), None) 
                                        for p_id in h_seg_ids}
        path_cumulative_rotations_of_particles = {p_id: path_cumulative_rotations_of_frames.get(h_pts_pre + str(p_id), None) 
                                        for p_id in h_seg_ids}
        path_direction_vectors_of_particles = {p_id: path_direction_vectors_of_frames.get(h_pts_pre + str(p_id), None) 
                                        for p_id in h_seg_ids}
        path_rotation_vectors_of_particles = {p_id: path_rotation_vectors_of_frames.get(h_pts_pre + str(p_id), None) 
                                        for p_id in h_seg_ids}
        
        return (path,
        path_points,
        path_cumulative_lengths,
        path_cumulative_rotations,
        path_direction_vectors,
        path_rotation_vectors, 
        path_of_particles,
        path_points_of_particles,
        path_cumulative_lengths_of_particles,
        path_cumulative_rotations_of_particles,
        path_direction_vectors_of_particles,
        path_rotation_vectors_of_particles,
        path_approximated_dlo_joint_values)
    # -----------------------------------------------------------------------
    
    # -----------------------------------------------------------------------
    # Functions related to saving and exporting the results
    def save_performance_results(self, 
                                performance_data, 
                                performance_data_extra, 
                                experiment_id, 
                                mingruiyu_scene_id, 
                                dlo_type, 
                                saving_folder_name,
                                is_to_current_file_dir=True):
        """
        Saves the performance results to a CSV file.

        Parameters:
        - performance_data: tuple containing performance data
        - performance_data_extra: tuple containing extra performance data
        - experiment_id: int, experiment ID
        - mingruiyu_scene_id: int, scene ID
        - dlo_type: int or None, DLO type
        - saving_folder_name: str, folder where to save the results

        """
        # Unpack performance data
        (_, simplified_dlo_num_segments, avr_state_approx_error, 
        planning_success, planning_time_ompl, planning_time_trajopt, 
        total_planning_time, ompl_path_length, trajopt_path_length) = performance_data

        (dlo_simplification_time, urdf_generation_time) = performance_data_extra

        TOTAL_PROCESS_TIME = dlo_simplification_time + urdf_generation_time + total_planning_time

        print("- Experiment ID: ", experiment_id)
        print("- Simplified DLO Num Segments: ", simplified_dlo_num_segments)
        print("- Average State Approximation Error: ", avr_state_approx_error)
        print("- Planning Success: ", planning_success)
        print("- DLO simplification time: ", dlo_simplification_time)
        print("- URDF generation time: ", urdf_generation_time)
        print("- Planning Time OMPL: ", planning_time_ompl)
        print("- Planning Time Trajopt: ", planning_time_trajopt)
        print("- Total Planning Time: ", total_planning_time)
        print("- TOTAL PROCESS TIME: ", TOTAL_PROCESS_TIME)
        print("- OMPL Path Length: ", ompl_path_length)
        print("- Trajopt Path Length: ", trajopt_path_length)
        print("")

        if is_to_current_file_dir:
            # File path and name:
            current_file_dir = os.path.dirname(os.path.abspath(__file__))

            # Create a folder for the results
            results_folder = os.path.join(current_file_dir, saving_folder_name)
        else:
            results_folder = os.path.expanduser(saving_folder_name)

        if not os.path.exists(results_folder):
            os.makedirs(results_folder)
            print(f"Directory '{results_folder}' created.")

        # File name for the results
        if mingruiyu_scene_id is None:
            perf_results_csv_file = f"planning_results.csv"
        else:
            if dlo_type is None:
                perf_results_csv_file = f"scene_{mingruiyu_scene_id}_experiment_results.csv"
            else:
                perf_results_csv_file = f"scene_{mingruiyu_scene_id}_dlo_{dlo_type}_experiment_results.csv"

        # If the file does not exist, create it and write the header
        if not os.path.exists(os.path.join(results_folder, perf_results_csv_file)):
            with open(os.path.join(results_folder, perf_results_csv_file), 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["experiment_id", "num_segments", "avr_state_approx_error",
                                "planning_success", "dlo_simplification_time", "urdf_generation_time",
                                "planning_time_ompl", "planning_time_trajopt",
                                "total_planning_time", "total_process_time",
                                "ompl_path_length", "trajopt_path_length"])

                print(f"File '{perf_results_csv_file}' created and header written.")

        # Append the results to the csv file
        with open(os.path.join(results_folder, perf_results_csv_file), 'a', newline='') as file:
            writer = csv.writer(file)

            if planning_success:
                writer.writerow([experiment_id, simplified_dlo_num_segments, avr_state_approx_error,
                                planning_success, dlo_simplification_time, urdf_generation_time,
                                planning_time_ompl, planning_time_trajopt,
                                total_planning_time, TOTAL_PROCESS_TIME,
                                ompl_path_length, trajopt_path_length])
            else:
                writer.writerow([experiment_id, simplified_dlo_num_segments, avr_state_approx_error,
                                planning_success, dlo_simplification_time, urdf_generation_time,
                                planning_time_ompl, planning_time_trajopt,
                                total_planning_time, TOTAL_PROCESS_TIME,
                                0.0, 0.0])

            print(f"Results appended to the file '{perf_results_csv_file}'.")

    def save_generated_paths(self, 
                            plan_data, 
                            experiment_id, 
                            mingruiyu_scene_id, 
                            dlo_type, 
                            saving_folder_name,
                            is_to_current_file_dir=True):
        """
        Saves the generated paths to a pickle file.

        Parameters:
        - plan_data: the plan data to be saved
        - experiment_id: int, experiment ID
        - mingruiyu_scene_id: int, scene ID
        - dlo_type: int or None, DLO type
        - saving_folder_name: str, folder where to save the results

        """
        # File path and name:
        if is_to_current_file_dir:
            current_file_dir = os.path.dirname(os.path.abspath(__file__))

            # Create a folder for the results
            results_folder = os.path.join(current_file_dir, saving_folder_name)
        else:
            results_folder = os.path.expanduser(saving_folder_name)

        if not os.path.exists(results_folder):
            os.makedirs(results_folder)
            print(f"Directory '{results_folder}' created.")

        # File name for the results
        if mingruiyu_scene_id is None:
            perf_results_pkl_file = f"{experiment_id}_planning_data.pkl"    
            
            # While the file exists add a suffix to the file name
            i = 0
            while os.path.exists(os.path.join(results_folder, perf_results_pkl_file)):
                i += 1
                perf_results_pkl_file = f"{experiment_id}_planning_data_{i}.pkl"
                
        else: 
            formatted_experiment_id = f"{experiment_id:03d}"
            if dlo_type is None:
                perf_results_pkl_file = f"scene_{mingruiyu_scene_id}_experiment_{formatted_experiment_id}_data.pkl"
            else:
                perf_results_pkl_file = f"scene_{mingruiyu_scene_id}_dlo_{dlo_type}_experiment_{formatted_experiment_id}_data.pkl"
                

        # Save the plan data to the pickle file
        with open(os.path.join(results_folder, perf_results_pkl_file), 'wb') as outp:  # Overwrites any existing file.
            pickle.dump(plan_data, outp, pickle.HIGHEST_PROTOCOL)

        print(f"Paths are saved to the file '{perf_results_pkl_file}'.")
    
    # -----------------------------------------------------------------------
    