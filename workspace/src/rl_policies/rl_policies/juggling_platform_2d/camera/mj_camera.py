import mujoco as mj
from mujoco.glfw import glfw
import numpy as np
import os
import cv2
from ball_detector import BallDetector

xml_path = '/home/admin/workspace/src/descriptions/juggling_platform_2d_description/urdf/robot_camera.xml'
simend = 20 #simulation time
print_camera_config = 0 #set to 1 to print camera config
                        #this is useful for initializing view of the model)
print_model = 0 #set to 1 to print the model info in the file model.txt in the current location

# For callback functions
button_left = False
button_middle = False
button_right = False
lastx = 0
lasty = 0

def init_controller(model,data):
    #initialize the controller here. This function is called once, in the beginning
    pass

def controller(model, data):
    #put the controller here. This function is called inside the simulation.
    pass

def keyboard(window, key, scancode, act, mods):
    if act == glfw.PRESS and key == glfw.KEY_BACKSPACE:
        mj.mj_resetData(model, data)
        mj.mj_forward(model, data)

def mouse_button(window, button, act, mods):
    # update button state
    global button_left
    global button_middle
    global button_right

    button_left = (glfw.get_mouse_button(
        window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS)
    button_middle = (glfw.get_mouse_button(
        window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS)
    button_right = (glfw.get_mouse_button(
        window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS)

    # update mouse position
    glfw.get_cursor_pos(window)

def mouse_move(window, xpos, ypos):
    # compute mouse displacement, save
    global lastx
    global lasty
    global button_left
    global button_middle
    global button_right

    dx = xpos - lastx
    dy = ypos - lasty
    lastx = xpos
    lasty = ypos

    # no buttons down: nothing to do
    if (not button_left) and (not button_middle) and (not button_right):
        return

    # get current window size
    width, height = glfw.get_window_size(window)

    # get shift key state
    PRESS_LEFT_SHIFT = glfw.get_key(
        window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
    PRESS_RIGHT_SHIFT = glfw.get_key(
        window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
    mod_shift = (PRESS_LEFT_SHIFT or PRESS_RIGHT_SHIFT)

    # determine action based on mouse button
    if button_right:
        if mod_shift:
            action = mj.mjtMouse.mjMOUSE_MOVE_H
        else:
            action = mj.mjtMouse.mjMOUSE_MOVE_V
    elif button_left:
        if mod_shift:
            action = mj.mjtMouse.mjMOUSE_ROTATE_H
        else:
            action = mj.mjtMouse.mjMOUSE_ROTATE_V
    else:
        action = mj.mjtMouse.mjMOUSE_ZOOM

    mj.mjv_moveCamera(model, action, dx/height,
                      dy/height, scene, cam)

def scroll(window, xoffset, yoffset):
    action = mj.mjtMouse.mjMOUSE_ZOOM
    mj.mjv_moveCamera(model, action, 0.0, -0.05 *
                      yoffset, scene, cam)

# #get the full path
dirname = os.path.dirname(__file__)
# abspath = os.path.join(dirname + "/" + xml_path)
# xml_path = abspath

#print the model
if (print_model==1):
    model_name = 'model.txt'
    model_path = os.path.join(dirname + "/" + model_name)
    mj.mj_printModel(model,model_path)


# MuJoCo data structures
model = mj.MjModel.from_xml_path(xml_path)
data = mj.MjData(model)
cam = mj.MjvCamera()
opt = mj.MjvOption()

# Init GLFW, create window, make OpenGL context current, request v-sync
glfw.init()
window = glfw.create_window(1200, 900, "Demo", None, None)
glfw.make_context_current(window)
glfw.swap_interval(1)

# initialize visualization data structures
mj.mjv_defaultCamera(cam)
mj.mjv_defaultOption(opt)
scene = mj.MjvScene(model, maxgeom=10000)
context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)

# install GLFW mouse and keyboard callbacks
glfw.set_key_callback(window, keyboard)
glfw.set_cursor_pos_callback(window, mouse_move)
glfw.set_mouse_button_callback(window, mouse_button)
glfw.set_scroll_callback(window, scroll)

# Example on how to set camera configuration
# cam.azimuth = 90
# cam.elevation = -45
# cam.distance = 2
# cam.lookat = np.array([0.0, 0.0, 0])

init_controller(model,data)
mj.set_mjcb_control(controller)

detector = BallDetector()

while not glfw.window_should_close(window):
    time_prev = data.time

    while (data.time - time_prev < 1.0/60.0):
        mj.mj_step(model, data)
        # qz = 0.25*np.sin(data.time)
        # q0 = np.sqrt(1-qz*qz)
        # quat = np.array([q0,0,0,qz])
        data.time += model.opt.timestep
        # data.qpos[3:] = quat.copy()
        mj.mj_forward(model,data)
    

    if (data.time>=simend):
        break

    # get framebuffer viewport
    viewport_width, viewport_height = glfw.get_framebuffer_size(
        window)
    viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)

    if (print_camera_config==1):
        print('cam.azimuth =',cam.azimuth,';','cam.elevation =',cam.elevation,';','cam.distance = ',cam.distance)
        print('cam.lookat =np.array([',cam.lookat[0],',',cam.lookat[1],',',cam.lookat[2],'])')

    mj.mjv_updateScene(model, data, opt, None, cam,
                       mj.mjtCatBit.mjCAT_ALL.value, scene)
    mj.mjr_render(viewport, scene, context)

    #Code taken from https://github.com/dtorre38/mujoco_opencv
    # ******** inset view *********
    #Settings for inset view
    camera_name = 'robot_camera'
    width, height = 640, 480
    offscreen_viewport = mj.MjrRect(0, 0, width, height)
    camera_id = model.camera(camera_name).id
    offscreen_cam = mj.MjvCamera()
    offscreen_cam.type = mj.mjtCamera.mjCAMERA_FIXED
    offscreen_cam.fixedcamid = camera_id
    
    mj.mjv_updateScene(model, data, opt, None, offscreen_cam,
                       mj.mjtCatBit.mjCAT_ALL.value, scene)
    mj.mjr_render(offscreen_viewport, scene, context)

    # Capture the pixels from the robot_camera render
    rgb_pixels = np.zeros((height, width, 3), dtype=np.uint8)
    depth_pixels = np.zeros((height, width), dtype=np.float32)
    mj.mjr_readPixels(rgb_pixels, depth_pixels, offscreen_viewport, context)

    # Corregir orientación y pasar a BGR (OpenCV)
    rgb_pixels_flipped = cv2.flip(rgb_pixels, 0)
    img_bgr = cv2.cvtColor(rgb_pixels_flipped, cv2.COLOR_RGB2BGR)
    
    # --- Detectar pelota ---
    result, circle, mask_color, mask_circle = detector.detect(img_bgr, gray_background=True)

    cv2.imshow("Ball detection", result)
    cv2.waitKey(1)

    glfw.swap_buffers(window)
    glfw.poll_events()

glfw.terminate()
