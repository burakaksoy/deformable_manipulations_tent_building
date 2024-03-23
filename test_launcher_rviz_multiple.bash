#!/bin/bash
sleep 1s;

# gnome-terminal --tab --title="ROSCORE" --command "bash -c \"source ~/.bashrc; killall gzclient && killall gzserver; roscore; exec bash\"";
# sleep 1s;

gnome-terminal --tab --title="RVIZ" --command "bash -c \"source ~/.bashrc; rosrun rviz rviz -d ~/catkin_ws_deformable/src/deformable_manipulations_tent_building/rviz/two_robot.rviz; exec bash\"";
sleep 4s;

gnome-terminal --tab --title="RVIZ" --command "bash -c \"source ~/.bashrc; rosrun rviz rviz -d ~/catkin_ws_deformable/src/deformable_manipulations_tent_building/rviz/two_robot_front.rviz; exec bash\"";
sleep 4s;

gnome-terminal --tab --title="RVIZ" --command "bash -c \"source ~/.bashrc; rosrun rviz rviz -d ~/catkin_ws_deformable/src/deformable_manipulations_tent_building/rviz/two_robot_side.rviz; exec bash\"";
sleep 4s;

gnome-terminal --tab --title="RVIZ" --command "bash -c \"source ~/.bashrc; rosrun rviz rviz -d ~/catkin_ws_deformable/src/deformable_manipulations_tent_building/rviz/two_robot_top.rviz; exec bash\"";
sleep 4s;
