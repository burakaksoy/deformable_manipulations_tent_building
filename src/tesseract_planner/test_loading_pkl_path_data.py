try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle
    
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm # Colormaps

from dlo_state_approximator import dlo_fwd_kin

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
    

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++    
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++    
scene_id = 4
experiment = 1
filename = "./generated_plans_i9_10885h/scene_" + str(scene_id) + "/scene_" + str(scene_id) + "_experiment_" + str(experiment).zfill(3) + "_data.pkl" 

# filename = "./generated_plans_i9_10885h/scene_4/scene_4_experiment_003_data.pkl" # Change this to the file you want to load


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++    
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++    

def calculate_dlo_length(approximated_state_pos):
    # Link lengths
    link_lengths = np.linalg.norm(approximated_state_pos[1:,:] - approximated_state_pos[:-1,:], axis=1)
    return np.sum(link_lengths)
    
with open(filename, 'rb') as inp:
    plan_data = pickle.load(inp)
    
    plan_data_ompl = plan_data[0]
    plan_data_trajopt = plan_data[1] 
    performance_data = plan_data[2]
    initial_n_target_states = plan_data[3]
    
    (ompl_path, ompl_path_points, ompl_path_cumulative_lengths, ompl_path_cumulative_rotations,
    ompl_path_direction_vectors, ompl_path_rotation_vectors, ompl_path_of_particles,
    ompl_path_points_of_particles, ompl_path_cumulative_lengths_of_particles,
    ompl_path_cumulative_rotations_of_particles, ompl_path_direction_vectors_of_particles,
    ompl_path_rotation_vectors_of_particles, ompl_path_approximated_dlo_joint_values) = plan_data_ompl
    
    (trajopt_path, trajopt_path_points, trajopt_path_cumulative_lengths, trajopt_path_cumulative_rotations,
    trajopt_path_direction_vectors, trajopt_path_rotation_vectors, trajopt_path_of_particles,
    trajopt_path_points_of_particles, trajopt_path_cumulative_lengths_of_particles,
    trajopt_path_cumulative_rotations_of_particles, trajopt_path_direction_vectors_of_particles,
    trajopt_path_rotation_vectors_of_particles, trajopt_path_approximated_dlo_joint_values) = plan_data_trajopt
    
    (experiment_id, simplified_dlo_num_segments, avr_state_approx_error, planning_success,
    planning_time_ompl, planning_time_trajopt, total_planning_time, ompl_path_length, trajopt_path_length) = performance_data
    
    (initial_full_state, initial_approximated_state_pos, initial_approximated_state_joint_pos, 
    target_full_state, target_approximated_state_pos, target_approximated_state_joint_pos) = initial_n_target_states
    
    # As an example print the performance data
    print("Experiment ID: ", experiment_id)
    print("Simplified DLO Num Segments: ", simplified_dlo_num_segments)
    print("Average State Approximation Error: ", avr_state_approx_error)
    print("Planning Success: ", planning_success)
    print("Planning Time OMPL: ", planning_time_ompl)
    print("Planning Time Trajopt: ", planning_time_trajopt)
    print("Total Planning Time: ", total_planning_time)
    print("OMPL Path Length: ", ompl_path_length)
    print("Trajopt Path Length: ", trajopt_path_length)

    # As an example calculate the DLO length from the approximated state positions
    dlo_length = calculate_dlo_length(initial_approximated_state_pos)
    print("DLO Length: ", dlo_length)

    # As an example plot the path of the particles
    # --------------------------------------------------------------------------------------------
    # OPTIONAL: Plot the path points
    try:
        ax = plt.figure().add_subplot(projection='3d')

        # Add title with the number of segments
        ax.set_title("Generated Path with " + str(simplified_dlo_num_segments) + " rigid links\nScene: " + str(scene_id) + ", Experiment: "+ str(experiment_id) , fontsize=30)

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
        # for id in ompl_path_points_of_particles:
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
        for id in trajopt_path_points_of_particles:
            ax.plot(trajopt_path_points_of_particles[id][:,0], # x
                    trajopt_path_points_of_particles[id][:,1], # y
                    trajopt_path_points_of_particles[id][:,2], # z
                    '--^'+path_colors[i], label='Point ' + str(id) + ' Path (after smoothing)', markersize=4, linewidth=2, alpha=.5)
            i += 1
            
        # Plot holding points path (after smoothing) without duplicate labels
        for i, id in enumerate(trajopt_path_points_of_particles):
            ax.plot(trajopt_path_points_of_particles[id][:,0], # x
                    trajopt_path_points_of_particles[id][:,1], # y
                    trajopt_path_points_of_particles[id][:,2], # z
                    '--^'+path_colors[i], markersize=4, linewidth=2)

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
    