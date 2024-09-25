import os
import re
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from tesseract_robotics.tesseract_common import (
    AngleAxisd,
    AnyPoly,
    AnyPoly_wrap_double,
    CollisionMarginData,
    CONSOLE_BRIDGE_LOG_DEBUG,
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
    AddLinkCommand,
    AddSceneGraphCommand,
    Commands,
    Environment,
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

from tesseract_robotics.tesseract_scene_graph import (
    Collision,
    Joint,
    JointType_FIXED,
    Link,
    Material,
    SceneGraph,
    Visual,
)

from tesseract_robotics.tesseract_urdf import (
    parseURDFFile,
    parseURDFString,
    writeURDFFile,
)

                                              
# from .tesseract_support_resource_locator import TesseractSupportResourceLocator

                                                                                                    
                                                
def create_add_wall_command(length, width, height, 
                            px, py, pz, rotation_angle, 
                            color, parent_link,
                            link_name_prefix, link_name_suffix):
    
    wall_link = Link(link_name_prefix + link_name_suffix)
    
    # Only add the visual and collision if the length, width and height are greater than 0
    if (length > 0) and (width > 0) and (height > 0):
        wall_link_visual = Visual()
        wall_link_visual.geometry = Box(length, width, height)
        wall_link_visual.material = Material()
        wall_link_visual.material.color = np.array(color)

        wall_link_collision = Collision()
        wall_link_collision.geometry = Box(length, width, height)

        wall_link.visual.push_back(wall_link_visual)
        wall_link.collision.push_back(wall_link_collision)

    wall_joint = Joint(link_name_prefix + link_name_suffix + "_joint")
    wall_joint.parent_link_name = parent_link
    wall_joint.child_link_name = wall_link.getName()
    wall_joint.type = JointType_FIXED

    # Transformation to set position and rotation
    wall_link_joint_transform = Isometry3d.Identity() * \
                                Translation3d(px, py, pz) * \
                                AngleAxisd(rotation_angle, np.array([0.0,0.0,1.0]))
    
    wall_joint.parent_to_joint_origin_transform = wall_link_joint_transform

    # Apply the wall to the environment
    add_wall_command = AddLinkCommand(wall_link, wall_joint)
    return add_wall_command

def create_l_shape_wall_commands(cmds = Commands(), parent_link = "base_link", 
                        prefix="",
                        l_x=0.5, l_y=0.5,
                        h=2.0, w=0.1, 
                        x=0.0, y=0.0, z=0.0,
                        angle=0.0,
                        color=[0.5, 0.5, 0.5, 1.0]):
    """Creates an L-shaped wall in the environment with two walls at a 90 degree angle to each other at the origin.
    Each wall is a box with dimensions l_x, l_y, h, w. The wall origin is at x, y, z and rotated by angle.
    One wall is along the x-axis and the other is along the y-axis. l_x and l_y are the lengths of the walls along 
    the -x and -y axes in the wall frame respectively. h and w are the height and tihckness of the walls respectively.

    Args:
        cmds (Commands): commands to add the wall to the environment
        parent_link (str): parent link name
        prefix (str): prefix for the wall link names
        l_x (float): length of the wall along the -x axis
        l_y (float): length of the wall along the -y axis
        h (float): height of the wall
        w (float): thickness of the wall
        x (float): origin x coordinate
        y (float): origin y coordinate
        z (float): origin z coordinate
        angle (float, optional): 
    """
    
    if prefix != "":
        prefix = prefix + "_" # Add underscore to the prefix
    
    # Create link for the L-shaped wall origin
    wall_origin_link = Link(prefix + "l_shaped_wall_origin_link")
    
    wall_origin_joint = Joint(prefix + "l_shaped_wall_origin_joint")
    wall_origin_joint.parent_link_name = parent_link
    wall_origin_joint.child_link_name = wall_origin_link.getName()
    wall_origin_joint.type = JointType_FIXED
    
    wall_origin_link_transform = Isometry3d.Identity() * \
                                 Translation3d(x, y, z) * \
                                 AngleAxisd(angle, np.array([0.0,0.0,1.0]))
    
    wall_origin_joint.parent_to_joint_origin_transform = wall_origin_link_transform
    
    wall_add_origin_command = AddLinkCommand(wall_origin_link, wall_origin_joint)
    cmds.push_back(wall_add_origin_command)
    
    wall_x_add_command = create_add_wall_command(l_x, w, h,
                                                -l_x/2.0, -w/2.0, h/2.0, 0.0,
                                                color, wall_origin_link.getName(),
                                                prefix, "along_x_wall_link")
    cmds.push_back(wall_x_add_command)
    
    wall_y_add_command = create_add_wall_command(l_y, w, h,
                                                -w/2.0, -l_y/2.0, h/2.0, -np.pi/2.0,
                                                color, wall_origin_link.getName(),
                                                prefix, "along_y_wall_link")
    cmds.push_back(wall_y_add_command)
                                                
    
    return cmds
    
    # env.applyCommands(cmds)
    # Update the viewer with the new environment
    # viewer.update_environment(env, [0,0,0])
    
    
def add_environment_obstacles_l_shape_corridor(env, viewer, ceiling_active=True,
                                               root_link_name="corridor_base_link",
                                               parent_link_name="base_link",
                                               export_as_urdf=False, 
                                               urdf_path="/tmp/", 
                                               urdf_name="l_shape_corridor"):
    """ Add an L-shaped corridor to the environment with walls along the x and y axes. 

    Args:
        env (environment): Tesseract environment
        viewer (TesseractViewer): Tesseract viewer
        ceiling_active (bool, optional): Add a ceiling to the corridor. Defaults to True.
        root_link_name (str, optional): Root link name of the corridor. Defaults to "corridor_base_link".
        parent_link_name (str, optional): Parent link name of the corridor root. Defaults to "base_link".
        export_as_urdf (bool, optional): Export the environment as a URDF file. Defaults to False.
        urdf_path (str, optional): Path to save the URDF file. Defaults to "/tmp/".
        urdf_name (str, optional): Name of the URDF file. Defaults to "l_shape_corridor". .urdf extension will be added automatically.
    """

    cmds = Commands() # Create a command list to apply all the commands at once

    if env is not None and viewer is not None:
        viewer.update_environment(env, [0,0,0])
    
    corridor_width = 0.7 # 1.1
    
    # GROUND PLANE (Assume square plane)
    plane_w = 6.0
    plane_h = 0.1 # plane thickness
    
    ceiling_height = 2.4
    wall_thickness = 0.1
    wall_height = ceiling_height
    
    
    wall_color = np.array([0.5, 0.5, 0.5, 0.5])
    wall_color_invisible = np.array([0.5, 0.5, 0.5, 0.3])
    floor_color = np.array([0.8, 0.8, 0.5, 1.0])
    ceiling_color = np.array([0.8, 0.8, 0.5, 0.3])

    # Add a box using Environment commands as a ground plane
    plane_link_visual = Visual()
    plane_link_visual.geometry = Box(plane_w,plane_w,plane_h)
    plane_link_visual.material = Material()
    plane_link_visual.material.color = floor_color

    plane_link_collision = Collision()
    plane_link_collision.geometry = Box(plane_w,plane_w,plane_h)

    plane_link = Link("link_ground_plane")
    plane_link.visual.push_back(plane_link_visual)
    plane_link.collision.push_back(plane_link_collision)

    plane_joint = Joint("joint_ground_plane")
    plane_joint.parent_link_name = root_link_name
    plane_joint.child_link_name = plane_link.getName()
    plane_joint.type = JointType_FIXED

    plane_link_joint_transform = Isometry3d.Identity() * Translation3d(0.0, 0.0, -plane_h/2.0)
    plane_joint.parent_to_joint_origin_transform = plane_link_joint_transform

    add_plane_command = AddLinkCommand(plane_link, plane_joint)
    cmds.push_back(add_plane_command)
    # env.applyCommand(add_plane_command)

    # viewer.update_environment(env, [0,0,0])

    if ceiling_active:
        # CEILING PLANE
        # Add a box using Environment commands as a ground plane
        plane_link_visual = Visual()
        plane_link_visual.geometry = Box(plane_w,plane_w,plane_h)
        plane_link_visual.material = Material()
        plane_link_visual.material.color = ceiling_color

        plane_link_collision = Collision()
        plane_link_collision.geometry = Box(plane_w,plane_w,plane_h)

        plane_link = Link("link_ceiling_plane")
        plane_link.visual.push_back(plane_link_visual)
        plane_link.collision.push_back(plane_link_collision)

        plane_joint = Joint("joint_ceiling_plane")
        plane_joint.parent_link_name = root_link_name
        plane_joint.child_link_name = plane_link.getName()
        plane_joint.type = JointType_FIXED

        plane_link_joint_transform = Isometry3d.Identity() * Translation3d(0.0, 0.0, ceiling_height+plane_h/2.0)
        plane_joint.parent_to_joint_origin_transform = plane_link_joint_transform

        add_plane_command = AddLinkCommand(plane_link, plane_joint)
        cmds.push_back(add_plane_command)
        # env.applyCommand(add_plane_command)

        # viewer.update_environment(env, [0,0,0])

    # WALLS
    # Create outer L-shaped wall        
    create_l_shape_wall_commands(cmds, parent_link=root_link_name,
                        prefix="outer",
                        l_x=plane_w, l_y=plane_w/2.0,
                        h=wall_height, w=wall_thickness,
                        x=plane_w/2.0, y=plane_w/2.0, z=0.0,
                        angle=0.0,
                        color=wall_color)    

    # Create inner L-shaped wall
    create_l_shape_wall_commands(cmds, parent_link=root_link_name,
                        prefix="inner",
                        l_x=plane_w - (corridor_width + wall_thickness), l_y=plane_w/2.0 - (corridor_width + wall_thickness),
                        h=wall_height, w=wall_thickness,
                        x=plane_w/2.0 - (corridor_width + wall_thickness) , y=plane_w/2.0 - (corridor_width + wall_thickness) , z=0.0,
                        angle=0.0,
                        color=wall_color)    
    
    # Second half of the outer L-shaped wall
    create_l_shape_wall_commands(cmds, parent_link=root_link_name,
                        prefix="outer2",
                        l_x=plane_w/2.0, l_y=plane_w,
                        h=wall_height, w=wall_thickness,
                        x=plane_w/2.0, y=-plane_w/2.0, z=0.0,
                        angle=np.deg2rad(-90.0),
                        color=wall_color)   
    
    # # Second half of the inner L-shaped wall
    # create_l_shape_wall_commands(cmds, parent_link=root_link_name,
    #                     prefix="inner2",
    #                     l_x=plane_w/2.0 - (corridor_width + wall_thickness), l_y=plane_w - (corridor_width + wall_thickness),
    #                     h=wall_height, w=wall_thickness,
    #                     x=plane_w/2.0 - (corridor_width + wall_thickness), y=-plane_w/2.0 + (corridor_width + wall_thickness), z=0.0,
    #                     angle=np.deg2rad(-90.0),
    #                     color=wall_color_invisible)
    
    # # Second half of the inner L-shaped wall
    create_l_shape_wall_commands(cmds, parent_link=root_link_name,
                        prefix="inner2",
                        l_x=plane_w/2.0 - ( wall_thickness), l_y=0.0,
                        h=wall_height, w=wall_thickness,
                        x=plane_w/2.0 - (corridor_width + wall_thickness), y=-plane_w/2.0 + (wall_thickness), z=0.0,
                        angle=np.deg2rad(-90.0),
                        color=wall_color)
    
    
    if export_as_urdf:
        # Export the environment as a URDF file
        
        # Create an empty scene graph
        scene_graph = SceneGraph()
        
        scene_graph.addLink(Link(root_link_name))
        
        # Set scene graph root
        # scene_graph.setRoot(env.getSceneGraph().getRoot())
        # scene_graph.setRoot("base_link")
        
        add_scene_graph_command = AddSceneGraphCommand(scene_graph)
        
        cmds_urdf = Commands()
        cmds_urdf.push_back(add_scene_graph_command)
        
        # Add each cmds to the cmds_urdf
        for i in range(cmds.size()):
            cmds_urdf.push_back(cmds[i])
        
        # cmds_urdf = cmds
        # cmds_urdf.insert(cmds_urdf.begin(), add_scene_graph_command)
        
        # Create a temporary environment to export the URDF with the cmds
        env_urdf = Environment()
        env_urdf.init(cmds_urdf)
        
        # get the scene graph
        scene_graph = env_urdf.getSceneGraph()
        # # The way of use writeURDFFile will be like this: writeURDFFile(sg, "/tmp/", "urdf0.urdf");
        writeURDFFile(scene_graph, urdf_path, urdf_name)
        
    # Add the connection between the environment root link (as the parent_link) and the root link of the corridor (as root_link_name)    
    root_link = Link(root_link_name)
    
    root_joint = Joint(root_link_name + "_joint")
    root_joint.parent_link_name = parent_link_name
    root_joint.child_link_name = root_link.getName()
    root_joint.type = JointType_FIXED

    # Transformation to set position and rotation
    root_link_joint_transform = Isometry3d.Identity()
    
    root_joint.parent_to_joint_origin_transform = root_link_joint_transform

    # Apply the wall to the environment
    add_root_command = AddLinkCommand(root_link, root_joint)
    
    # cmds.push_back(add_root_command)
    cmds.insert(cmds.begin(), add_root_command)
    
    if env is not None and viewer is not None:
        env.applyCommands(cmds)
        viewer.update_environment(env, [0,0,0])
        
        
def add_environment_obstacles_from_urdf(env, viewer, urdf_path, urdf_name):
    """
    Add/Append obstacles from a URDF file to the environment.
    
    Assume that the URDF file contains the static environment obstacles.
    
    urdf_path: str
        Path to the URDF file
    urdf_name: str
        Name of the URDF file
    """
    
    # tesseract_support = os.environ["TESSERACT_RESOURCE_PATH"]
    # path =  os.path.join(tesseract_support, "urdf/lbr_iiwa_14_r820_fixed.urdf")
    path = os.path.join(urdf_path, urdf_name)
    
    locator = GeneralResourceLocator()
    # locator = TesseractSupportResourceLocator()
        
    scene_graph_to_add = parseURDFFile(path, locator).release()
    
    # # urdf_string = Path(path).read_text()
    # # # print(urdf_string)  
    # # scene_graph_to_add = parseURDFString(urdf_string, locator).release()
        
    add_scene_graph_command = AddSceneGraphCommand(scene_graph_to_add)
    
    cmds = Commands()
    cmds.push_back(add_scene_graph_command)
    
    env.applyCommands(cmds)
    viewer.update_environment(env, [0,0,0])
    
    

def add_environment_obstacles(env, viewer, obstacle_ceiling_active=True):
    cmds = Commands()
    # --------------------------------------------------------------------------------------------
    # ADD ENVIRONMENT OBJECTS HERE

    # # Add a sphere using Environment commands
    # sphere_link_visual = Visual()
    # sphere_link_visual.geometry = Sphere(0.1)

    # sphere_link_collision = Collision()
    # sphere_link_collision.geometry = Sphere(0.1)

    # sphere_link = Link("sphere_link")
    # sphere_link.visual.push_back(sphere_link_visual)
    # sphere_link.collision.push_back(sphere_link_collision)

    # sphere_joint = Joint("sphere_joint")
    # sphere_joint.parent_link_name = "base_link"
    # sphere_joint.child_link_name = sphere_link.getName()
    # sphere_joint.type = JointType_FIXED

    # sphere_link_joint_transform = Isometry3d.Identity() * Translation3d(0.0, 0.0, 1.0)
    # sphere_joint.parent_to_joint_origin_transform = sphere_link_joint_transform

    # add_sphere_command = AddLinkCommand(sphere_link, sphere_joint)
    # env.applyCommand(add_sphere_command)

    viewer.update_environment(env, [0,0,0])

    # OBSTACLE BOX
    # Add a box using Environment commands
    box_link_visual = Visual()
    box_link_visual.geometry = Box(1.5, 1, 1)
    box_link_visual.material = Material()
    box_link_visual.material.color = np.array([0.4, 0.4, 0.4, 0.5])

    box_link_collision = Collision()
    box_link_collision.geometry = Box(1.5, 1, 1)

    box_link = Link("box_link")
    box_link.visual.push_back(box_link_visual)
    box_link.collision.push_back(box_link_collision)

    box_joint = Joint("box_joint")
    box_joint.parent_link_name = "base_link"
    box_joint.child_link_name = box_link.getName()
    box_joint.type = JointType_FIXED

    box_link_joint_transform = Isometry3d.Identity() * Translation3d(0.0, 1.0, 0.5)
    box_joint.parent_to_joint_origin_transform = box_link_joint_transform

    add_box_command = AddLinkCommand(box_link, box_joint)
    cmds.push_back(add_box_command)
    # env.applyCommand(add_box_command)

    viewer.update_environment(env, [0,0,0])

    # GROUND PLANE (Assume square plane)
    plane_w = 6.0
    plane_h = 0.1

    # Add a box using Environment commands as a ground plane
    plane_link_visual = Visual()
    plane_link_visual.geometry = Box(plane_w,plane_w,plane_h)
    plane_link_visual.material = Material()
    plane_link_visual.material.color = np.array([0.5, 0.5, 0.5, 1.0])

    plane_link_collision = Collision()
    plane_link_collision.geometry = Box(plane_w,plane_w,plane_h)

    plane_link = Link("plane_link")
    plane_link.visual.push_back(plane_link_visual)
    plane_link.collision.push_back(plane_link_collision)

    plane_joint = Joint("plane_joint")
    plane_joint.parent_link_name = "base_link"
    plane_joint.child_link_name = plane_link.getName()
    plane_joint.type = JointType_FIXED

    plane_link_joint_transform = Isometry3d.Identity() * Translation3d(0.0, 0.0, -plane_h/2.0)
    plane_joint.parent_to_joint_origin_transform = plane_link_joint_transform

    add_plane_command = AddLinkCommand(plane_link, plane_joint)
    cmds.push_back(add_plane_command)
    # env.applyCommand(add_plane_command)

    viewer.update_environment(env, [0,0,0])

    # OBSTACLE CEILING
    # obstacle_ceiling_active = True
    ceil_angle = np.pi/6.0
    ceil_min_h = 0.0 # 0.2
    ceil_l = (plane_w/2.0)/np.cos(ceil_angle)

    if obstacle_ceiling_active:
        # Add a box using Environment commands as a ceiling
        ceiling_link_visual = Visual()
        ceiling_link_visual.geometry = Box(ceil_l, plane_w, plane_h)
        ceiling_link_visual.material = Material()
        ceiling_link_visual.material.color = np.array([0.5, 0.5, 0.5, 1.0])

        ceiling_link_collision = Collision()
        ceiling_link_collision.geometry = Box(ceil_l, plane_w, plane_h)

        ceiling_link = Link("ceiling_link")
        ceiling_link.visual.push_back(ceiling_link_visual)
        ceiling_link.collision.push_back(ceiling_link_collision)

        ceiling_joint = Joint("ceiling_joint")
        ceiling_joint.parent_link_name = "base_link"
        ceiling_joint.child_link_name = ceiling_link.getName()
        ceiling_joint.type = JointType_FIXED

        x = plane_w/4.0
        z = ceil_min_h + ceil_l*np.sin(ceil_angle)/2.0
        ceiling_link_joint_transform = Isometry3d.Identity() * \
                                    Translation3d(x, 0.0, z) * \
                                    AngleAxisd(ceil_angle, np.array([0.0,1.0,0.0]))
        
        ceiling_joint.parent_to_joint_origin_transform = ceiling_link_joint_transform

        add_ceiling_command = AddLinkCommand(ceiling_link, ceiling_joint)
        cmds.push_back(add_ceiling_command)
        # env.applyCommand(add_ceiling_command)

        viewer.update_environment(env, [0,0,0])

        # Add the other half of the ceiling
        ceiling_link2 = Link("ceiling_link2")
        ceiling_link2.visual.push_back(ceiling_link_visual)
        ceiling_link2.collision.push_back(ceiling_link_collision)

        ceiling_joint2 = Joint("ceiling_joint2")
        ceiling_joint2.parent_link_name = "base_link"
        ceiling_joint2.child_link_name = ceiling_link2.getName()
        ceiling_joint2.type = JointType_FIXED

        ceiling_link_joint_transform2 = Isometry3d.Identity() * \
                                        Translation3d(-x, 0.0, z) * \
                                        AngleAxisd(-ceil_angle, np.array([0.0,1.0,0.0]))
        
        ceiling_joint2.parent_to_joint_origin_transform = ceiling_link_joint_transform2

        add_ceiling_command2 = AddLinkCommand(ceiling_link2, ceiling_joint2)
        cmds.push_back(add_ceiling_command2)
        # env.applyCommand(add_ceiling_command2)

        viewer.update_environment(env, [0,0,0])

    env.applyCommands(cmds)
    viewer.update_environment(env, [0,0,0])

    # --------------------------------------------------------------------------------------------

def main():
    # Find current directory
    current_dir = os.path.dirname(os.path.realpath(__file__))
    
    add_environment_obstacles_l_shape_corridor(env=None, viewer=None, ceiling_active=True,
                                               root_link_name="corridor_base_link",
                                               parent_link_name="base_link",
                                               export_as_urdf=True, 
                                               urdf_path=current_dir, 
                                               urdf_name="l_shape_corridor")
    
if __name__ == "__main__":
    # Call the main function
    main()