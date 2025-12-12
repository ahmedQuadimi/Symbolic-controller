import pybullet as p
import pybullet_data
import time
import math
import numpy as np
import ctypes
import pickle
import os

# ---------------------------------------
# 1) Initialize PyBullet
# ---------------------------------------
user32 = ctypes.windll.user32
hWnd = user32.FindWindowW(None, "PyBullet")
user32.MoveWindow(hWnd, 0, 0, 1920, 1080, True)
camera_distance = 7 / (2 * math.tan(math.radians(60)/2))
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.resetDebugVisualizerCamera(
    cameraDistance=camera_distance,
    cameraYaw=0,
    cameraPitch=-89.999,
    cameraTargetPosition=[5, 5, 0]
)
p.setGravity(0, 0, -9.81)

p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
p.configureDebugVisualizer(p.COV_ENABLE_WIREFRAME, 0)
p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
p.configureDebugVisualizer(p.COV_ENABLE_PLANAR_REFLECTION, 0)
p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
# Load plane
plane = p.loadURDF("plane.urdf")
texture_id = p.loadTexture("white.jpg")
p.changeVisualShape(plane, -1, textureUniqueId=texture_id)


if not os.path.exists("path.pkl"):
    path = [(np.float64(9.75), np.float64(9.75), np.float64(0.3141592653589793)), (np.float64(9.75), np.float64(9.25), np.float64(-0.9424777960769379)), (np.float64(9.75), np.float64(8.75), np.float64(-1.5707963267948966)), (np.float64(9.25), np.float64(7.25), np.float64(-2.199114857512855)), (np.float64(8.75), np.float64(5.75), np.float64(-2.199114857512855)), (np.float64(8.75), np.float64(4.25), np.float64(-2.199114857512855)), (np.float64(8.75), np.float64(2.75), np.float64(-2.827433388230814)), (np.float64(7.25), np.float64(1.75), np.float64(2.827433388230814)), (np.float64(5.75), np.float64(1.75), np.float64(2.827433388230814)), (np.float64(4.25), np.float64(1.25), np.float64(2.827433388230814)), (np.float64(2.75), np.float64(1.25), np.float64(2.199114857512855))]
else:
    with open("path.pkl", "rb") as f:
        path = pickle.load(f)



print(path)
# Load Husky robot
husky = p.loadURDF("husky/husky.urdf", [path[0][0], path[0][1], 0.1])

def create_wall_borders(min_x=0, max_x=10, min_y=0, max_y=10, wall_height=0.5, wall_thickness=0.1):
    """Creates physical walls that robots will collide with"""
    walls = []
    
    # Bottom wall (along x-axis)
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[(max_x-min_x)/2, wall_thickness/2, wall_height/2])
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[(max_x-min_x)/2, wall_thickness/2, wall_height/2], 
                              rgbaColor=[0.3, 0.3, 0.3, 1])
    walls.append(p.createMultiBody(baseCollisionShapeIndex=col, baseVisualShapeIndex=vis, 
                                   basePosition=[(min_x+max_x)/2, min_y, wall_height/2]))
    
    # Top wall
    walls.append(p.createMultiBody(baseCollisionShapeIndex=col, baseVisualShapeIndex=vis, 
                                   basePosition=[(min_x+max_x)/2, max_y, wall_height/2]))
    
    # Left wall (along y-axis)
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[wall_thickness/2, (max_y-min_y)/2, wall_height/2])
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[wall_thickness/2, (max_y-min_y)/2, wall_height/2], 
                              rgbaColor=[0.3, 0.3, 0.3, 1])
    walls.append(p.createMultiBody(baseCollisionShapeIndex=col, baseVisualShapeIndex=vis, 
                                   basePosition=[min_x, (min_y+max_y)/2, wall_height/2]))
    
    # Right wall
    walls.append(p.createMultiBody(baseCollisionShapeIndex=col, baseVisualShapeIndex=vis, 
                                   basePosition=[max_x, (min_y+max_y)/2, wall_height/2]))
    return walls


create_wall_borders(0-0.5, 10+0.5, 0-0.5, 10+0.5, )
if os.path.exists("zones.pkl"):
    with open("zones.pkl", "rb") as f:
        danger_zones = pickle.load(f)
else:
    danger_zones = [
        # center_x, center_y, width/2, height/2, color in rgba [0, 1] -> [0, 255]
        (4.5, 9.0, 0.5, 0.5, (1, 1, 0, 0.5)),
        (9.0, 2.5, 0.5, 0.5, (0, 1, 1, 0.5)),
        (2.5, 1.0, 0.5, 0.5, (141/255, 182/255, 0, 0.5)),
        (5.0, 5.0, 2.0, 2.0, (1, 0, 0, 0.5))
    ]

for center_x, center_y, hx, hy, color in danger_zones:
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[hx, hy, 0.01])
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[hx, hy, 0.01], rgbaColor=color)
    p.createMultiBody(baseCollisionShapeIndex=col, baseVisualShapeIndex=vis, 
                     basePosition=[center_x, center_y, 0.01])

t = 0

path_index = 0
linear_speed = 0.02
angular_speed = 0.05
dt = 1/240

while True:
    path_index = min(path_index, len(path) - 1)
    target_x, target_y, *target_yaw = path[path_index]

    current_pos, current_ori = p.getBasePositionAndOrientation(husky)
    current_x, current_y, current_z = current_pos
    _, _, current_yaw = p.getEulerFromQuaternion(current_ori)

    distance = math.hypot(target_x - current_x, target_y - current_y)
    
    if distance > 0.01:
        dir_x = (target_x - current_x) / distance
        dir_y = (target_y - current_y) / distance
        
        step_distance = min(linear_speed, distance)
        new_x = current_x + dir_x * step_distance
        new_y = current_y + dir_y * step_distance
        
        target_heading = math.atan2(target_y - current_y, target_x - current_x)
        
        delta_yaw = target_heading - current_yaw
        if delta_yaw > math.pi:
            delta_yaw -= 2 * math.pi
        elif delta_yaw < -math.pi:
            delta_yaw += 2 * math.pi
        
        if abs(delta_yaw) > angular_speed:
            new_yaw = current_yaw + math.copysign(angular_speed, delta_yaw)
        else:
            new_yaw = target_heading
    else:
        new_x = current_x
        new_y = current_y
        new_yaw = current_yaw

    p.resetBasePositionAndOrientation(
        husky,
        [new_x, new_y, 0.1],
        p.getQuaternionFromEuler([0, 0, new_yaw])
    )

    if distance < 0.05:
        path_index = min(path_index + 1, len(path) - 1)
        print(path_index, path[path_index])

    time.sleep(dt)