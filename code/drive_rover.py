# Do the necessary imports
import argparse
import shutil
import base64
from datetime import datetime
import os
import cv2
import numpy as np
import socketio
import eventlet
import eventlet.wsgi
from PIL import Image
from flask import Flask
from io import BytesIO, StringIO
import json
import pickle
import matplotlib.image as mpimg
import time
import eventlet

# Import functions for perception and decision making
from perception import perception_step
from decision import decision_step
from supporting_functions import update_rover, create_output_images
# Initialize socketio server and Flask application 
# (learn more at: https://python-socketio.readthedocs.io/en/latest/)
sio = socketio.Server()
app = Flask(__name__)

# Read in ground truth map and create 3-channel green version for overplotting
# NOTE: images are read in by default with the origin (0, 0) in the upper left
# and y-axis increasing downward.
ground_truth = mpimg.imread('../calibration_images/map_bw.png')
# This next line creates arrays of zeros in the red and blue channels
# and puts the map into the green channel.  This is why the underlying 
# map output looks green in the display image
ground_truth_3d = np.dstack((ground_truth*0, ground_truth*255, ground_truth*0)).astype(np.float)

# Define RoverState() class to retain rover state parameters
class RoverState():
    def __init__(self):
        self.start_time = None # To record the start time of navigation
        self.total_time = None # To record total duration of naviagation
        self.img = None # Current camera image
        self.pos = None # Current position (x, y)
        self.yaw = None # Current yaw angle
        self.pitch = None # Current pitch angle
        self.roll = None # Current roll angle
        self.vel = None # Current velocity
        self.steer = 0 # Current steering angle
        self.throttle = 0 # Current throttle value
        self.brake = 0 # Current brake value
        self.nav_angles = None # Angles of navigable terrain pixels
        self.nav_dists = None # Distances of navigable terrain pixels

        self.cam_nav_angles = None # For forward decision making. Angles of navigable terrain pixels
        self.cam_nav_dists = None # For forward decision making. Distances of navigable terrain pixels

        self.ground_truth = ground_truth_3d # Ground truth worldmap
        self.mode = 'forward' # Current mode (can be forward or stop)

        # original 0.2, 2, 1.3
        self.throttle_set = 0.2 # Throttle setting when accelerating
        self.max_vel = 2.0 # Maximum velocity (meters/second)
        self.acceptable_roll = 2  # degrees for roll and pitch to be acceptable for mapping

        self.brake_set = 10 # Brake setting when braking
        # The stop_forward and go_forward fields below represent total count
        # of navigable terrain pixels.  This is a very crude form of knowing
        # when you can keep going and when you should stop.  Feel free to
        # get creative in adding new fields or modifying these!
        self.stop_forward = 50 # Threshold to initiate stopping
        self.go_forward = 500 # Threshold to go forward again
        # Image output from perception step
        # Update this image to display your intermediate analysis steps
        # on screen in autonomous mode
        self.vision_image = np.zeros((160, 320, 3), dtype=np.float) 
        # Worldmap
        # Update this image with the positions of navigable terrain
        # obstacles and rock samples
        self.worldmap = np.zeros((200, 200, 3), dtype=np.float) 
        self.samples_pos = None # To store the actual sample positions
        self.samples_found = 0 # To count the number of samples found
        self.near_sample = 0 # Will be set to telemetry value data["near_sample"]
        self.picking_up = 0 # Will be set to telemetry value data["picking_up"]
        self.send_pickup = False # Set to True to trigger rock pickup
        # Threshold for detecting rock
        # 15 is too large, 3 is too small
        self.rock_threshold = 7
        self.rock_pos = None
        self.target_rock_pos = None
        self.initial_rock_distance = 9999
        # rock direction relative to Rover
        # -ve means to the left, +ve means to the right
        self.rock_direction = 0
		# previous good position
        self.prevposition = None

        # our own counter for rocked picked up
        self.rocks_picked = 0

        # the fps we used to do target frames  calculation
        self.used_fps = 30
		# target Yaw for unstuck
        self.unstuck_target_yaw_sec = 20
        self.unstuck_target_yaw = 210
        self.unstuck_count = 0
        # frames exceed to consider unstuck action is stucked. Typical FPS = 33
        # 400 is safe
        self.unstuck_count_target_sec = 20
        self.unstuck_count_target = 300
        # time to run the unstuck action
        # 100 is too short
        self.unstuck_getout_count_target_sec = 20
        self.unstuck_getout_count_target = 210

        # for forward to rock        
        self.unstuck_torock_count_target_sec = 15
        self.unstuck_torock_count_target = 400
        # for drift awhile function
        # 3 is too little
        self.unstuck_awhile_count_target_sec = 4
        self.unstuck_awhile_count_target = 150
		# start position (X, Y)
        self.start_pos = None 
        # number of times we see in the map to consider "old"
        self.new_pix_threshold = 30
        # Navmap is the real position that Rover has travelled
        self.navmap = np.zeros((200, 200), dtype=np.int) 
        # Seenmap is the positions that we guess from view
        self.seenmap = np.zeros((200, 200), dtype=np.float) 
        # obstacle map is the positions that we guess from view
        self.obstaclemap = np.zeros((200, 200), dtype=np.float) 
        # Wall Map is the positions that we are confident is wall. either rover stuck or marked
        self.wallmap = np.zeros((200, 200), dtype=np.int) 

        # target is calculated based on distance and fps
        self.go_home_count_target = 150
        self.go_home_count = 0
        self.home_park_distance = 1
        # steering during search. 2 is slow and good
        self.search_steering = 4
        self.unstuck_steering = 8
        # unstuck turn have 10 degree allowance, so 50 = 40 - 60
        self.unstuck_turn_angle = 50
        # keep screen update time to reduce lag
        self.screen_update_time = 0
        self.unstuckmap = np.zeros((200, 200), dtype=np.int) 
        self.unstucklimit = 5

        # rock in front detection using Camera Warped image
        self.rock_in_front = 0
        self.rock_in_front_left = 0
        self.rock_in_front_right = 0
        # threshold of number of bright pixels to decide not rock. Sample calibration image has 1
        self.rock_in_front_thresh = 15

        # objective: search or go_home or search_only or pick_rock
        self.objective = 'search'
        self.target_rocks = 6

# Initialize our rover 
Rover = RoverState()

# Variables to track frames per second (FPS)
# Intitialize frame counter
frame_counter = 0
# Initalize second counter
second_counter = time.time()
fps = None


# Define telemetry function for what to do with incoming data
@sio.on('telemetry')
def telemetry(sid, data):

    global frame_counter, second_counter, fps
    frame_counter+=1
    # Do a rough calculation of frames per second (FPS)
    if (time.time() - second_counter) > 1:
        fps = frame_counter
        frame_counter = 0
        second_counter = time.time()
    

    # recalculate if FPS differ too much from the used FPS
    
    if (Rover.used_fps is not None) and (fps is not None):
      if (fps > 10) and (abs(Rover.used_fps - fps) > 5):
        Rover.used_fps = fps
        Rover.unstuck_target_yaw = Rover.unstuck_target_yaw_sec * fps
        Rover.unstuck_count_target = Rover.unstuck_count_target_sec * fps
        Rover.unstuck_getout_count_target = Rover.unstuck_getout_count_target_sec * fps
        Rover.unstuck_awhile_count_target = Rover.unstuck_awhile_count_target_sec * fps
        Rover.unstuck_torock_count_target = Rover.unstuck_torock_count_target_sec * fps

    print("Current FPS: {}".format(fps), 'Unstuck count = ', Rover.unstuck_count_target)

    if data:
        global Rover
        # Initialize / update Rover with current telemetry
        Rover, image = update_rover(Rover, data)

        if np.isfinite(Rover.vel):

            # Execute the perception and decision steps to update the Rover's state
            Rover = perception_step(Rover)
            Rover = decision_step(Rover)

            # Create output images to send to server
            out_image_string1, out_image_string2 = create_output_images(Rover)

            # The action step!  Send commands to the rover!
            commands = (Rover.throttle, Rover.brake, Rover.steer)
            send_control(commands, out_image_string1, out_image_string2)
 
            # If in a state where want to pickup a rock send pickup command
            if Rover.send_pickup:
                send_pickup()
                # Reset Rover flags
                Rover.send_pickup = False
        # In case of invalid telemetry, send null commands
        else:

            # Send zeros for throttle, brake and steer and empty images
            send_control((0, 0, 0), '', '')

        # If you want to save camera images from autonomous driving specify a path
        # Example: $ python drive_rover.py image_folder_path
        # Conditional to save image frame if folder was specified
        if args.image_folder != '':
            timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
            image_filename = os.path.join(args.image_folder, timestamp)
            image.save('{}.jpg'.format(image_filename))

    else:
        sio.emit('manual', data={}, skip_sid=True)

@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control((0, 0, 0), '', '')
    sample_data = {}
    sio.emit(
        "get_samples",
        sample_data,
        skip_sid=True)

def send_control(commands, image_string1, image_string2):
    # Define commands to be sent to the rover
    data={
        'throttle': commands[0].__str__(),
        'brake': commands[1].__str__(),
        'steering_angle': commands[2].__str__(),
        'inset_image1': image_string1,
        'inset_image2': image_string2,
        }
    # Send commands via socketIO server
    sio.emit(
        "data",
        data,
        skip_sid=True)
    #eventlet.sleep(0)

# Define a function to send the "pickup" command 
def send_pickup():
    print("Picking up")
    pickup = {}
    sio.emit(
        "pickup",
        pickup,
        skip_sid=True)
    #eventlet.sleep(0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument(
        'image_folder',
        type=str,
        nargs='?',
        default='',
        help='Path to image folder. This is where the images from the run will be saved.'
    )
    args = parser.parse_args()
    
    #os.system('rm -rf IMG_stream/*')
    if args.image_folder != '':
        print("Creating image folder at {}".format(args.image_folder))
        if not os.path.exists(args.image_folder):
            os.makedirs(args.image_folder)
        else:
            shutil.rmtree(args.image_folder)
            os.makedirs(args.image_folder)
        print("Recording this run ...")
    else:
        print("NOT recording this run ...")
    
    # wrap Flask application with socketio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
