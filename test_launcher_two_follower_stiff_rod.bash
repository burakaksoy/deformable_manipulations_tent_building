#!/bin/bash
sleep 1s;

gnome-terminal --tab --title="ROSCORE" --command "bash -c \"source ~/.bashrc; killall gzclient && killall gzserver; roscore; exec bash\"";
sleep 1s;

gnome-terminal --tab --title="All" --command "bash -c \"source ~/.bashrc; roslaunch deformable_manipulations_tent_building main_launcher_two_robot.launch launch_controller:=false; exec bash\"";
sleep 4s;

gnome-terminal --tab --title="RVIZ" --command "bash -c \"source ~/.bashrc; rosrun rviz rviz -d ~/catkin_ws_deformable/src/deformable_manipulations_tent_building/rviz/two_robot.rviz; exec bash\"";
sleep 4s;

gnome-terminal --tab --title="GUI" --command "bash -c \"source ~/.bashrc; rosrun dlo_simulator_stiff_rods test_gui.py; exec bash\"";
sleep 1s;

# rqt_ez_publisher, to publish to "/space_nav/twist"
gnome-terminal --tab --title="RQT_EZ_PUBLSHER" --command "bash -c \"source ~/.bashrc; sleep 2s; source ~/.bashrc; rosrun rosrun rqt_ez_publisher rqt_ez_publisher; exec bash\"";
sleep 1s;

gnome-terminal --tab --title="Controller" --command "bash -c \"source ~/.bashrc; roslaunch deformable_manipulations_tent_building velocity_controller.launch; exec bash\"";
sleep 1s;

# gnome-terminal --tab --title="Controller" --command "bash -c \"source ~/.bashrc; roslaunch deformable_manipulations_tent_building velocity_controller_single_min_dist.launch; exec bash\"";
# sleep 1s;

# To start the controller, call the service with command:
# rosservice call /tent_building_velocity_controller/set_enable_controller "data: true" 