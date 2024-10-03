import pandas as pd
import itertools
import time

from tesseract_planner.planner import TesseractPlanner

# -----------------------------------------------------------------------
def read_state_dict_from_csv(file):
    # Load the data from the CSV file
    df = pd.read_csv(file)
    
    # Convert DataFrame to dictionary
    state_dict = df.to_dict(orient='list')
    return state_dict
# -----------------------------------------------------------------------

# -----------------------------------------------------------------------
tesseract_resource_path = "~/catkin_ws_deformable/src/"

# Scene for the experiments
# scene_ids = [1, 2, 3, 4]
# scene_ids = [0,2,6]
scene_ids = [7]

# DLO Types, None for the default
# dlo_types = None
# dlo_types = [1,4,5]
dlo_types = [5]

# Number of experiments to run
num_experiments = 3

tesseract_tent_pole_srdf = "/home/burak/catkin_ws_deformable/src/deformable_description/urdf/pole_automatic/pole_automatic.srdf" # TODO: Must be passed

tesseract_tent_pole_tcp_frame = "tool0"  # specified in the vel controller yaml file
tesseract_tent_pole_manipulator_group = "manipulator" # defined in the srdf file and specified in the vel controller yaml file
tesseract_tent_pole_manipulator_working_frame = "base_link" # defined in the srdf file and specified in the vel controller yaml file

viewer_enabled = True

# If dlo_types is None, set it to [None]
if dlo_types is None:
    dlo_types = [None]

# Use itertools.product to get all combinations
for mingruiyu_scene_id, dlo_type in itertools.product(scene_ids, dlo_types):
    print(f"Scene ID: {mingruiyu_scene_id}, DLO Type: {dlo_type}")

    # Create json_file_path from scene_id
    json_file_path = "/home/burak/catkin_ws_deformable/src/dlo_simulator_stiff_rods/config/scenes/scene_mingruiyu_" + str(mingruiyu_scene_id) + ".json"

    # -----------------------------------------------------------------------
    # Create tessaract planner object
    tesseract_planner = TesseractPlanner(collision_scene_json_file = json_file_path,
                                        tesseract_resource_path = tesseract_resource_path,
                                        urdf_url_or_path = "",
                                        srdf_url_or_path = tesseract_tent_pole_srdf,
                                        tcp_frame = tesseract_tent_pole_tcp_frame,
                                        manipulator = tesseract_tent_pole_manipulator_group,
                                        working_frame = tesseract_tent_pole_manipulator_working_frame,
                                        viewer_enabled = viewer_enabled)
    # -----------------------------------------------------------------------

    # -----------------------------------------------------------------------
    # Assume we are given:
    # 1.    the initial and goal states of the dlo as a dictionary
    #       with "id,p_x,p_y,p_z,o_x,o_y,o_z,o_w" are the keys
    #       where id is the particle id and p_x, p_y, p_z are the position components
    #       and o_x, o_y, o_z, o_w are the orientation components.
    # 2.    Lenght and radius of the dlo
    # 3.    The full dlo holding segment ids from the custom_static_particles
    # 4.    Maximum and minimum number of segments that can be used in the simplified dlo urdf.
    #       If not given, we can set it to a default value like to the full dlo num segments.
    # 5.    Environment limits and joint angle limits of the URDF robot
    # 6.    Approximation error threshold for the dlo state approximator

    # 1.
    # initial_full_state_file = "/home/burak/catkin_ws_deformable/src/deformable_manipulations_tent_building/saved_states/mingrui_yu_scene_4_initial_states.csv"
    # target_full_state_file = "/home/burak/catkin_ws_deformable/src/deformable_manipulations_tent_building/saved_states/mingrui_yu_scene_4_target_states.csv"
    if dlo_type is None:
        initial_full_state_file = "/home/burak/catkin_ws_deformable/src/deformable_manipulations_tent_building/saved_states/mingrui_yu_scene_"+ str(mingruiyu_scene_id)+ "_initial_states.csv"
        target_full_state_file = "/home/burak/catkin_ws_deformable/src/deformable_manipulations_tent_building/saved_states/mingrui_yu_scene_" + str(mingruiyu_scene_id)+ "_target_states.csv"
    else:
        initial_full_state_file = "/home/burak/catkin_ws_deformable/src/deformable_manipulations_tent_building/saved_states/mingrui_yu_scene_"+ str(mingruiyu_scene_id)+ "_dlo_" + str(dlo_type) + "_initial_states.csv"
        target_full_state_file = "/home/burak/catkin_ws_deformable/src/deformable_manipulations_tent_building/saved_states/mingrui_yu_scene_" + str(mingruiyu_scene_id)+ "_dlo_" + str(dlo_type) + "_target_states.csv"


    # Read the state dictionaries from the csv files
    initial_full_state_dict = read_state_dict_from_csv(initial_full_state_file) # TODO: Must be passed
    target_full_state_dict = read_state_dict_from_csv(target_full_state_file) # TODO: Must be passed

    # 2.
    length = 0.5 # Example # TODO: Must be passed

    radius = 0.0035 # Example # TODO: Must be passed
    # radius = 0.002 # Example # TODO: Must be passed

    # 3.
    # full_dlo_holding_segment_ids=custom_static_particles
    full_dlo_holding_segment_ids = [] # TODO: Must be passed
    full_dlo_holding_segment_ids = [0,39] # Example # TODO: Must be passed

    # 4.
    max_simplified_dlo_num_segments = 21 # Example (If given) # TODO: Must be passed
    min_simplified_dlo_num_segments = 2 # Example (If given) # TODO: Must be passed

    # 5.
    # environment_limits_xyz=[-1, 1, -1, 1, -1, 1] # Example # TODO: Must be passed,
    # environment_limits_xyz=[-0.5, 0.5, 0, 0.5, -1, 1] # Example # TODO: Must be passed, 
    environment_limits_xyz=[-0.5, 0.5, -0.5, 0.5, 0, 0.6] # Example # TODO: Must be passed, 

    # joint_angle_limits_xyz_deg=[-90, 90, -180, 180, -10, 10] # Example # TODO: Must be passed, 
    joint_angle_limits_xyz_deg=[-45, 45, -45, 45, -20, 20] # Example # TODO: Must be passed
    # joint_angle_limits_xyz_deg=[-30, 30, -30, 30, -10, 10] # Example # TODO: Must be passed

    # 6.
    approximation_error_threshold = 0.02 # Example # TODO: Must be passed

    for experiment_id in range(1, num_experiments+1):
        time.sleep(2) # Sleep for 2 seconds to make sure the previous experiment is finished
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print("Experiment ID:", experiment_id, "started..")
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            
        # --------------------------------------------------------------------------------------------
        # (planned_path,
        # planned_path_points,
        # planned_path_cumulative_lengths,
        # planned_path_cumulative_rotations,
        # planned_path_direction_vectors,
        # planned_path_rotation_vectors,
        # planned_path_of_particles,
        # planned_path_points_of_particles,
        # planned_path_cumulative_lengths_of_particles,
        # planned_path_cumulative_rotations_of_particles,
        # planned_path_direction_vectors_of_particles,
        # planned_path_rotation_vectors_of_particles) = tesseract_planner.plan(initial_full_state_dict, 
        #                                                                         target_full_state_dict,
        #                                                                         length,
        #                                                                         radius,
        #                                                                         full_dlo_holding_segment_ids,
        #                                                                         max_simplified_dlo_num_segments,
        #                                                                         min_simplified_dlo_num_segments,
        #                                                                         environment_limits_xyz,
        #                                                                         joint_angle_limits_xyz_deg,
        #                                                                         approximation_error_threshold,
        #                                                                         return_all_data=False)

        plan_data = tesseract_planner.plan(initial_full_state_dict, 
                                            target_full_state_dict,
                                            length,
                                            radius,
                                            full_dlo_holding_segment_ids,
                                            max_simplified_dlo_num_segments,
                                            min_simplified_dlo_num_segments,
                                            environment_limits_xyz,
                                            joint_angle_limits_xyz_deg,
                                            approximation_error_threshold,
                                            return_all_data=True)

        # Unpack the plan data
        (plan_data_ompl, 
        plan_data_trajopt, 
        performance_data, 
        initial_n_target_states,
        performance_data_extra) = plan_data
        # --------------------------------------------------------------------------------------------
        
        # --------------------------------------------------------------------------------------------
        # Save the performance results to a csv file using the class method
        print("Saving the performance results to a csv file:\n")

        # Set the folder name to save the results and created paths
        if dlo_type is None:
            saving_folder_name = f"generated_plans_test/scene_{mingruiyu_scene_id}"
        else:
            saving_folder_name = f"generated_plans_real_demo_test/scene_{mingruiyu_scene_id}_dlo_{dlo_type}"

        # Call the class method to save performance results
        tesseract_planner.save_performance_results(performance_data, 
                                                    performance_data_extra, 
                                                    experiment_id, 
                                                    mingruiyu_scene_id, 
                                                    dlo_type, 
                                                    saving_folder_name)
        # --------------------------------------------------------------------------------------------

        # --------------------------------------------------------------------------------------------
        # Save the generated paths to a pickle file using the class method
        print("\nSaving the generated paths to a pickle file\n")

        # Call the class method to save generated paths
        tesseract_planner.save_generated_paths(plan_data, 
                                                experiment_id, 
                                                mingruiyu_scene_id, 
                                                dlo_type, 
                                                saving_folder_name)
        # --------------------------------------------------------------------------------------------
        
        if viewer_enabled:
            input("Press enter to continue to the next experiment..")
        
        print("--------------------------------------------------------------------")
        
# wait for user input to keep the viewer alive
input("Experiments are completed, Press enter to exit..")