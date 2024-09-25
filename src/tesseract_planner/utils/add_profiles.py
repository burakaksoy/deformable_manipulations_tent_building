import os
import re
import traceback
import numpy as np
import time
import sys
from pathlib import Path

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
                                                setLogLevel, \
                                                Timer, \
                                                AngleAxisd

from tesseract_robotics.tesseract_environment import Environment, \
                                                     AddLinkCommand, \
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
                                                    
# ----------------------------------------------------------------------------------------------------------------
 
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
                                                   ContactTestType_CLOSEST, \
                                                    CollisionEvaluatorType_NONE, \
                                                    CollisionEvaluatorType_DISCRETE, \
                                                    CollisionEvaluatorType_LVS_DISCRETE, \
                                                    CollisionEvaluatorType_CONTINUOUS, \
                                                    CollisionEvaluatorType_LVS_CONTINUOUS, \
                                                    CollisionCheckProgramType_ALL, \
                                                    CollisionCheckProgramType_ALL_EXCEPT_START,\
                                                    CollisionCheckProgramType_ALL_EXCEPT_END,\
                                                    CollisionCheckProgramType_START_ONLY,\
                                                    CollisionCheckProgramType_END_ONLY,\
                                                    CollisionCheckProgramType_INTERMEDIATE_ONLY
                                                   
                                                   
from tesseract_robotics.tesseract_motion_planners_ompl import OMPLDefaultPlanProfile, \
                                                              RRTConnectConfigurator, \
                                                              OMPLProblemGeneratorFn, \
                                                              OMPLMotionPlanner, \
                                                              ProfileDictionary_addProfile_OMPLPlanProfile, \
                                                              OMPLProblemStateSpace_REAL_STATE_SPACE, \
                                                              OMPLProblemStateSpace_SE3_STATE_SPACE, \
                                                              OMPLProblemStateSpace_REAL_CONSTRAINTED_STATE_SPACE
                                                            
                                                                                                    
def add_MinLengthProfile(profiles, name, length=100):
    """Add a MinLengthProfile to the ProfileDictionary

    Args:
        profiles (ProfileDictionary): The ProfileDictionary to add the MinLengthProfile to 
        length (int): The length of the trajectory
        name (str): The name of the Profile e.g. "DEFAULT"
    """

    MINLENGTH_DEFAULT_NAMESPACE = "MinLengthTask"

    # Set the number of steps to use for the trajectory for the MinLengthProfile
    min_length_profile = MinLengthProfile(length)

    ProfileDictionary_addProfile_MinLengthProfile(profiles, MINLENGTH_DEFAULT_NAMESPACE, name, min_length_profile)
    
def add_TrajOptPlanProfile(profiles, name, num_joints):
    """Add a TrajOptPlanProfile to the ProfileDictionary

    Args:
        profiles (ProfileDictionary): The ProfileDictionary to add the TrajOptPlanProfile to
        name (str): The name of the Profile e.g. "DEFAULT"
    """
    
    TRAJOPT_DEFAULT_NAMESPACE = "TrajOptMotionPlannerTask"

    ####################
    ## Plan Profile BEGIN

    trajopt_plan_profile = TrajOptDefaultPlanProfile()
    

    # trajopt_plan_profile.cartesian_coeff = np.array([1, 1, 1, 1, 1, 1], dtype=np.float64)
    # trajopt_plan_profile.joint_coeff = np.array([10, 10, 1, 100, 100, 100, 1], dtype=np.float64)
    # trajopt_plan_profile.joint_coeff = np.array([1000, 1000, 0, 100, 100, 100, 0], dtype=np.float64)
    # trajopt_plan_profile.joint_coeff = np.array([0,0,0,0,0,0,0], dtype=np.float64)
    
    joint_coeff = np.ones(num_joints, dtype=np.float64)
    # joint_coeff[:2] = 2 # x, y
    # joint_coeff[2] = 1 # z
    # joint_coeff[3:6] = 1 # rx, ry, rz of the initial segment orientation
    # if num_joints > 6:
    #     joint_coeff[6::3] = 10 # rx of the internal joints
    #     joint_coeff[7::3] = 10 # ry of the internal joints
    #     joint_coeff[8::3] = 100 # rz of the internal joints
    trajopt_plan_profile.joint_coeff = joint_coeff

    # trajopt_plan_profile.constraint_error_functions # ???

    # trajopt_plan_profile.term_type # ???

    # Arguments: (profile_dictionary, ns, profile_name, profile)
    # ProfileDictionary_addProfile_TrajOptPlanProfile(profiles, TRAJOPT_DEFAULT_NAMESPACE, "TEST_PROFILE", trajopt_plan_profile)
    # ProfileDictionary_addProfile_TrajOptPlanProfile(profiles, TRAJOPT_DEFAULT_NAMESPACE, "DEFAULT", trajopt_plan_profile)
    # ProfileDictionary_addProfile_TrajOptPlanProfile(profiles, TRAJOPT_DEFAULT_NAMESPACE, "freespace_profile", trajopt_plan_profile)
    ProfileDictionary_addProfile_TrajOptPlanProfile(profiles, TRAJOPT_DEFAULT_NAMESPACE, name, trajopt_plan_profile)

    ## Plan Profile END
    ####################


    ####################
    ## Composite Profile BEGIN 

    trajopt_composite_profile = TrajOptDefaultCompositeProfile()

    trajopt_composite_profile.collision_cost_config.enabled = True # If true, a collision cost term will be added to the problem. Default: true*/
    trajopt_composite_profile.collision_cost_config.use_weighted_sum = True # Use the weighted sum for each link pair. This reduces the number equations added to the problem. If set to true, it is recommended to start with the coeff set to one Default: false*/
    trajopt_composite_profile.collision_cost_config.safety_margin = 0.015 # 0.005 # 0.025 # 0.0150 # 2.5cm #  Max distance in which collision costs will be evaluated. Default: 0.025*/
    trajopt_composite_profile.collision_cost_config.safety_margin_buffer = 0.0 # 0.01 # Distance beyond buffer_margin in which collision optimization will be evaluated. This is set to 0 by default (effectively disabled) for collision costs.
    trajopt_composite_profile.collision_cost_config.type = CollisionEvaluatorType_DISCRETE_CONTINUOUS # The evaluator type that will be used for collision checking. # SINGLE_TIMESTEP, DISCRETE_CONTINUOUS, CAST_CONTINUOUS. Default: DISCRETE_CONTINUOUS
    # trajopt_composite_profile.collision_cost_config.type = CollisionEvaluatorType_SINGLE_TIMESTEP # The evaluator type that will be used for collision checking. # SINGLE_TIMESTEP, DISCRETE_CONTINUOUS, CAST_CONTINUOUS. Default: DISCRETE_CONTINUOUS
    trajopt_composite_profile.collision_cost_config.coeff = 10 #0.1 # The collision coeff/weight. Default: 20*/

    trajopt_composite_profile.collision_constraint_config.enabled = True # If true, a collision cost term will be added to the problem. Default: true
    trajopt_composite_profile.collision_constraint_config.use_weighted_sum = True # Use the weighted sum for each link pair. This reduces the number equations added to the problem. If set to true, it is recommended to start with the coeff set to one. Default: false
    trajopt_composite_profile.collision_constraint_config.safety_margin = 0.001 # 0.01 # 0.016 # Max distance in which collision costs will be evaluated. Default: 0.01
    trajopt_composite_profile.collision_constraint_config.safety_margin_buffer = 0.001 # 0.051 # Distance beyond buffer_margin in which collision optimization will be evaluated. Default: 0.05
    trajopt_composite_profile.collision_constraint_config.type = CollisionEvaluatorType_DISCRETE_CONTINUOUS # The evaluator type that will be used for collision checking. # SINGLE_TIMESTEP, DISCRETE_CONTINUOUS, CAST_CONTINUOUS. Default: DISCRETE_CONTINUOUS
    # trajopt_composite_profile.collision_constraint_config.type = CollisionEvaluatorType_SINGLE_TIMESTEP # The evaluator type that will be used for collision checking. # SINGLE_TIMESTEP, DISCRETE_CONTINUOUS, CAST_CONTINUOUS. Default: DISCRETE_CONTINUOUS
    trajopt_composite_profile.collision_constraint_config.coeff = 50 #20 # The collision coeff/weight. Default: 20

    # The type of contact test to perform: FIRST, CLOSEST, ALL. Default: ALL
    # trajopt_composite_profile.contact_test_type = ContactTestType_ALL # ContactTestType_CLOSEST 
    # trajopt_composite_profile.contact_test_type = ContactTestType_ALL

    trajopt_composite_profile.smooth_velocities = True # If true, a joint velocity cost with a target of 0 will be applied for all timesteps Default: true
    # trajopt_composite_profile.velocity_coeff = np.array([10, 10, 1, 100, 100, 100, 1], dtype=np.float64) # This default to all ones, but allows you to weight different joints differently. Default: Eigen::VectorXd::Ones(num_joints)

    trajopt_composite_profile.smooth_accelerations = True # If true, a joint acceleration cost with a target of 0 will be applied for all timesteps Default: false
    # trajopt_composite_profile.acceleration_coeff = np.array([1], dtype=np.float64) # This default to all ones, but allows you to weight different joints differently. Default: Eigen::VectorXd::Ones(num_joints)

    trajopt_composite_profile.smooth_jerks = False # If true, a joint jerk cost with a target of 0 will be applied for all timesteps Default: false
    # trajopt_composite_profile.jerk_coeff = np.array([1], dtype=np.float64) # This default to all ones, but allows you to weight different joints differently. Default: Eigen::VectorXd::Ones(num_joints)

    trajopt_composite_profile.avoid_singularity = False #  If true, applies a cost to avoid kinematic singularities. Default: false
    trajopt_composite_profile.avoid_singularity_coeff = 5.0 # Optimization weight associated with kinematic singularity avoidance. Default: 5.0

    trajopt_composite_profile.longest_valid_segment_fraction = 0.5 # Set the resolution at which state validity needs to be verified in order for a motion between two states to be considered valid in post checking of trajectory returned by trajopt. The resolution is equal to longest_valid_segment_fraction * state_space.getMaximumExtent(). Default: 0.01
    # Note: The planner takes the conservative of either longest_valid_segment_fraction or longest_valid_segment_length.

    trajopt_composite_profile.longest_valid_segment_length = 0.05 # Set the resolution at which state validity needs to be verified in order for a motion between two states to be considered valid. If norm(state1 - state0) > longest_valid_segment_length. Default: 0.1
    # Note: This gets converted to longest_valid_segment_fraction. longest_valid_segment_fraction = longest_valid_segment_length / state_space.getMaximumExtent()

    # Arguments: (profile_dictionary, ns, profile_name, profile)
    # ProfileDictionary_addProfile_TrajOptCompositeProfile(profiles, TRAJOPT_DEFAULT_NAMESPACE, "TEST_PROFILE", trajopt_composite_profile)
    # ProfileDictionary_addProfile_TrajOptCompositeProfile(profiles, TRAJOPT_DEFAULT_NAMESPACE, "DEFAULT", trajopt_composite_profile)
    # ProfileDictionary_addProfile_TrajOptCompositeProfile(profiles, TRAJOPT_DEFAULT_NAMESPACE, "freespace_profile", trajopt_composite_profile)
    ProfileDictionary_addProfile_TrajOptCompositeProfile(profiles, TRAJOPT_DEFAULT_NAMESPACE, name, trajopt_composite_profile)

    ## Composite Profile END
    ####################
    
    ####################
    ## Solver Profile BEGIN 
    
    trajopt_solver_profile = TrajOptDefaultSolverProfile()
    
    # trajopt_solver_profile.convex_solver =  # The Convex solver to use. Default: sco::ModelType::OSQP
    
    # trajopt_solver_profile.convex_solver_config = # The convex solver config to use. sco::ModelConfig::Ptr, default: nullptr, which uses the default settings
    # eg. from C++
    # auto convex_solver_config = std::make_shared<sco::OSQPModelConfig>();
    # convex_solver_config->settings.adaptive_rho = 0;
    # trajopt_solver_profile->convex_solver_config = convex_solver_config;
    
    # trajopt_solver_profile.opt_info = # Optimization paramters. sco::BasicTrustRegionSQPParameters
    # trajopt_solver_profile.opt_info.num_threads = 0 # If greater than one, multi threaded functions are called. Default: 0
    
    trajopt_solver_profile.opt_info.max_iter = 200 # The maximum number of iterations to run the optimization. Default: 40 or 50
    # trajopt_solver_profile.opt_info.min_approx_improve = 1e-5 # If model improves less than this, exit and report convergence. Default: 1e-4
    # trajopt_solver_profile.opt_info.min_trust_box_size = 1e-5 # If trust region gets any smaller, exit and report convergence. Default: 1e-4
    
    # trajopt_solver_profile.opt_info.max_qp_solver_failures = 3 # Max number of times the QP solver can fail before optimization is aborted. Default: 3, DON'T USE
    
    
    ProfileDictionary_addProfile_TrajOptSolverProfile(profiles, TRAJOPT_DEFAULT_NAMESPACE, name, trajopt_solver_profile)
    
    ## Solver Profile END
    ####################
    
def add_OMPLDefaultPlanProfile(profiles, name):
    """Add a OMPLDefaultPlanProfile to the ProfileDictionary

    Args:
        profiles (ProfileDictionary): The ProfileDictionary to add the OMPLDefaultPlanProfile to
        name (str): The name of the Profile e.g. "DEFAULT"
    """
    
    """
    NOTE: OMPL does not support the concept of multi waypoint planning like descartes and trajopt. Because of this
    every plan instruction will be its a seperate ompl motion plan and therefore planning information is relevent
    for this motion planner in the profile.
    """
    
    OMPL_DEFAULT_NAMESPACE = "OMPLMotionPlannerTask"

    """
    C++ EXAMPLE:

    // Create OMPL Profile
    auto ompl_profile = std::make_shared<OMPLDefaultPlanProfile>();
    
    auto ompl_planner_config = std::make_shared<RRTConnectConfigurator>();
    
    ompl_planner_config->range = range_; // the maximum distance the tree can extend towards a randomly selected sample in the configuration space during each iteration.
    
    ompl_profile->planning_time = planning_time_; DEFAULT 60.0 ??????  
    ompl_profile->planners = { ompl_planner_config, ompl_planner_config };

    // Create profile dictionary
    auto profiles = std::make_shared<ProfileDictionary>();
    profiles->addProfile<OMPLPlanProfile>(OMPL_DEFAULT_NAMESPACE, "FREESPACE", ompl_profile);
    
    ------------------------------------------------------------------------------------------
    ANOTHER EXAMPLE:
    // Setup Problem
    tesseract_motion_planners::OMPLMotionPlanner ompl_planner;

    auto ompl_config =
        std::make_shared<tesseract_motion_planners::OMPLPlannerConstrainedConfig>(tesseract_, "manipulator");

    ompl_config->start_waypoint = std::make_shared<tesseract_motion_planners::JointWaypoint>(swp, kin->getJointNames());
    ompl_config->end_waypoint = std::make_shared<tesseract_motion_planners::JointWaypoint>(ewp, kin->getJointNames());
    ompl_config->collision_safety_margin = 0.01;
    ompl_config->planning_time = planning_time_;
    ompl_config->max_solutions = 2;
    ompl_config->longest_valid_segment_fraction = 0.01;

    ompl_config->collision_continuous = false;
    ompl_config->collision_check = false;
    ompl_config->simplify = false;
    ompl_config->n_output_states = 50;

    if (use_trajopt_constraint_)
    {
        if (plotting_)
        ompl_config->constraint =
            std::make_shared<TrajOptGlassUprightConstraint>(tesseract_, kin, "manipulator", "tool0", plotter);
        else
        ompl_config->constraint =
            std::make_shared<TrajOptGlassUprightConstraint>(tesseract_, kin, "manipulator", "tool0", nullptr);
    }
    else
    {
        Eigen::Vector3d normal = -1.0 * Eigen::Vector3d::UnitZ();
        ompl_config->constraint = std::make_shared<GlassUprightConstraint>(normal, kin);
    }

    for (int i = 0; i < 4; ++i)
    {
        auto rrtconnect_planner = std::make_shared<tesseract_motion_planners::ESTConfigurator>();
        rrtconnect_planner->range = range_;
        ompl_config->planners.push_back(rrtconnect_planner);
    }

    // Set the planner configuration
    ompl_planner.setConfiguration(ompl_config);

    // Solve Trajectory
    CONSOLE_BRIDGE_logInform("glass upright plan OMPL example");

    ros::Time tStart = ros::Time::now();
    tesseract_motion_planners::PlannerResponse ompl_planning_responseb = ompl_planner.solve(ompl_planning_request);
    CONSOLE_BRIDGE_logError("planning time: %.3f", (ros::Time::now() - tStart).toSec());
    """

    ####################
    ## Plan Profile BEGIN

    ompl_plan_profile = OMPLDefaultPlanProfile()
    
    """
    #* state_space
        The state space to use when planning.
        C++ Type: OMPLProblemStateSpace state_space{ OMPLProblemStateSpace::REAL_STATE_SPACE };
        Default: REAL_STATE_SPACE
        Other Options: REAL_CONSTRAINED_STATE_SPACE, SE3_STATE_SPACE
    """
    # ompl_plan_profile.state_space
    
    """
    #* planning_time
        Max planning time allowed in seconds. default: 5.0 seconds
    """
    ompl_plan_profile.planning_time = 30.0
    
    """
    #* max_solutions (default: 10)
        The max number of solutions. If max solutions are hit it will exit even if other threads are running.
    """
    ompl_plan_profile.max_solutions = 10 # 10
    
    """
    #* simplify (default: False)
        Simplify trajectory. If set to true it ignores n_output_states and returns the simplest trajectory.
    """
    ompl_plan_profile.simplify = False 
    
    """
    #* optimize (default: True)
        This uses all available planning time to create the most optimized trajectory given the objective function.
        This is required because not all OMPL planners are optimize graph planners. 
        If the planner you choose is an optimize graph planner then setting this to true has no affect. 
        In the case of non-optimize planners they still use the OptimizeObjective function 
        but only when searching the graph to find the most optimize solution based on the 
        provided optimize objective function. 
        In the case of these type of planners like RRT and RRTConnect if set to true,
        it will leverage all planning time to keep finding solutions up to your max solutions count 
        to find the most optimal solution.
    """
    ompl_plan_profile.optimize = True
    
    """
    #* planners
        Vector of planner configurators (OMPLPlannerConfigurator).
        Default: *TWO* RRTConnectConfigurator's
        This will create a new thread for each planner configurator provided.
        
        Other Options:
        - SBLConfigurator
        - ESTConfigurator
        - LBKPIECE1Configurator
        - BKPIECE1Configurator
        - KPIECE1Configurator
        - BiTRRTConfigurator
        - RRTConfigurator
        - RRTConnectConfigurator
        - RRTstarConfigurator
        - TRRTConfigurator
        - PRMConfigurator
        - PRMstarConfigurator
        - LazyPRMstarConfigurator
        - SPARSConfigurator
    """
    ompl_plan_profile.planners.clear()
    
    # range = 0.15 # the maximum distance the tree can extend towards a randomly selected sample in the configuration space during each iteration.
    range = 0.15 # the maximum distance the tree can extend towards a randomly selected sample in the configuration space during each iteration.
    # range = 0.05 # the maximum distance the tree can extend towards a randomly selected sample in the configuration space during each iteration.
    # range = 0.025 # the maximum distance the tree can extend towards a randomly selected sample in the configuration space during each iteration.
    # Increasing the range may help in reaching the goal faster if the environment has fewer obstacles.
    # Decreasing the range can result in a more detailed path which might be beneficial in cluttered or highly constrained spaces.
    # A large range might cause the algorithm to miss narrow passages, as large steps could overshoot small viable corridors. 
    # A very small range could lead to excessive computation time and a large number of nodes, potentially exhausting memory resources.
    
    # Forward Planner in the case of RRTConnect
    planner1 = RRTConnectConfigurator()
    planner1.range = range
    
    # Backward Planner in the case of RRTConnect
    planner2 = RRTConnectConfigurator()
    planner2.range = range
    
    ompl_plan_profile.planners.append(planner1)
    ompl_plan_profile.planners.append(planner2)
    
    
    """
    ##* collision_check_config
        The collision check configuration. (C++ type: tesseract_collision::CollisionCheckConfig)
    """
    
    # ompl_plan_profile.collision_check_config.contact_manager_config = 
    """
    #* collision_check_config.contact_manager_config (C++ type: tesseract_collision::ContactManagerConfig)
        Used to configure the contact manager prior to a series of checks.
        
        Set the collision margin for check. Objects with closer than the specified margin will be returned
    """
    # ompl_plan_profile.collision_check_config.contact_manager_config.margin_data = CollisionMarginData(0.0)
    
    """
    #* collision_check_config.contact_request (C++ type: ContactRequest)
        used for this check. Default test type: ALL
        
        Options:
        - ContactTestType_ALL: Return all contacts for a pair of objects
        - ContactTestType_FIRST: Return at first contact for any pair of objects
        - ContactTestType_CLOSEST: Return the global minimum for a pair of objects
        - ContactTestType_LIMITED: Return limited set of contacts for a pair of objects
    """
    # ompl_plan_profile.collision_check_config.contact_request = 
    # ompl_plan_profile.collision_check_config.contact_request = ContactRequest(ContactTestType_ALL)
    
    """
    #* collision_check_config.type (C++ type: CollisionEvaluatorType)
        Specifies the type of collision check to be performed.
        This is a High level descriptor used in planners and utilities to specify what kind of collision check is desired.
        
        Options:
        - CollisionEvaluatorType_NONE
        - CollisionEvaluatorType_DISCRETE: Discrete contact manager using only steps specified (DEFAULT)
        - CollisionEvaluatorType_LVS_DISCRETE: Discrete contact manager interpolating using longest valid segment
        - CollisionEvaluatorType_CONTINUOUS: Continuous contact manager using only steps specified
        - CollisionEvaluatorType_LVS_CONTINUOUS: Continuous contact manager interpolating using longest valid segment
    """
    # ompl_plan_profile.collision_check_config.type = CollisionEvaluatorType_DISCRETE
    # ompl_plan_profile.collision_check_config.type = CollisionEvaluatorType_NONE
    
    """
    #* collision_check_config.longest_valid_segment_length
        Longest valid segment to use if type supports lvs. Default: 0.005
    """
    # ompl_plan_profile.collision_check_config.longest_valid_segment_length = 0.010
    
    """
    #* collision_check_config.check_program_mode
        Specifies the mode used when collision checking program/trajectory. Default: ALL
        
        Options:
        - CollisionCheckProgramType_ALL: Check all states
        - CollisionCheckProgramType_ALL_EXCEPT_START: Check all states except the start state
        - CollisionCheckProgramType_ALL_EXCEPT_END: Check all states except the end state
        - CollisionCheckProgramType_START_ONLY: Check only the start state
        - CollisionCheckProgramType_END_ONLY: Check only the end state
        - CollisionCheckProgramType_INTERMEDIATE_ONLY: Check only the intermediate states
    """
    # ompl_plan_profile.collision_check_config.check_program_mode = CollisionCheckProgramType_ALL
    # ompl_plan_profile.collision_check_config.check_program_mode = CollisionCheckProgramType_END_ONLY
    
    """
    #* state_sampler_allocator
        The state sampler allocator. This can be null and it will use Tesseract default state sampler allocator.
    """
    # ompl_plan_profile.state_sampler_allocator

    """
    #* optimization_objective_allocator
        Set the optimization objective function allocator. Default is to minimize path length. 
    """
    # ompl_plan_profile.optimization_objective_allocator
    
    """
    #* svc_allocator
        The ompl state validity checker. If nullptr and collision checking enabled it uses StateCollisionValidator.
    """
    # ompl_plan_profile.state_sampler_allocator

    """
    #* mv_allocator
        The ompl motion validator. If nullptr and continuous collision checking enabled it used ContinuousMotionValidator.
    """
    # ompl_plan_profile.optimization_objective_allocator
    

    
    # ompl_plan_profile.applyStartStates ???
    # ompl_plan_profile.applyGoalStates ???
    """
        void applyGoalStates(OMPLProblem& prob,
                       const Eigen::Isometry3d& cartesian_waypoint,
                       const MoveInstructionPoly& parent_instruction,
                       const tesseract_common::ManipulatorInfo& manip_info,
                       const std::vector<std::string>& active_links,
                       int index) const override;

        void applyGoalStates(OMPLProblem& prob,
                            const Eigen::VectorXd& joint_waypoint,
                            const MoveInstructionPoly& parent_instruction,
                            const tesseract_common::ManipulatorInfo& manip_info,
                            const std::vector<std::string>& active_links,
                            int index) const override;

        void applyStartStates(OMPLProblem& prob,
                                const Eigen::Isometry3d& cartesian_waypoint,
                                const MoveInstructionPoly& parent_instruction,
                                const tesseract_common::ManipulatorInfo& manip_info,
                                const std::vector<std::string>& active_links,
                                int index) const override;

        void applyStartStates(OMPLProblem& prob,
                                const Eigen::VectorXd& joint_waypoint,
                                const MoveInstructionPoly& parent_instruction,
                                const tesseract_common::ManipulatorInfo& manip_info,
                                const std::vector<std::string>& active_links,
                                int index) const override;
    """
    

    # Arguments: (profile_dictionary, ns, profile_name, profile)
    # ProfileDictionary_addProfile_OMPLPlanProfile(profiles, OMPL_DEFAULT_NAMESPACE, "DEFAULT", ompl_plan_profile)
    # ProfileDictionary_addProfile_OMPLPlanProfile(profiles, OMPL_DEFAULT_NAMESPACE, "freespace_profile", ompl_plan_profile)
    ProfileDictionary_addProfile_OMPLPlanProfile(profiles, OMPL_DEFAULT_NAMESPACE, name, ompl_plan_profile)

    ## Plan Profile END
    ####################
    