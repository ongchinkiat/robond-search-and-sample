import numpy as np

# use the Rover current location to calculate distance from home
def distance_from_home(Rover):
    dist = np.linalg.norm(Rover.pos - Rover.start_pos)
    return dist

def home_angle(Rover):
    rad = np.arctan2(Rover.start_pos[1]-Rover.pos[1], Rover.start_pos[0]-Rover.pos[0])
    degrees = np.degrees(rad)
    if (degrees < 0):
       degrees += 360
    return degrees

# +ve for unstuck anti, -ve for unstuck
def turn_angle(orig, turn_angle):
    result = orig + turn_angle
    result = result % 360
    return result

# difference in degrees between 2 angles, no direction
def diff_angle(angle1, angle2):
    result = angle1 - angle2
    result = abs((result + 180) % 360 - 180)
    return result

#To do:
#(1) detect rock in the middle of view
#     more analysis on the perspect_transform view
#(2) hard_unstuck mode
#     make a 360 degree turn, find the yaw with the most clear view, turn to that yaw


# This is where you can build a decision tree for determining throttle, brake and steer 
# commands based on the output of the perception_step() function
def decision_step(Rover):

    # Implement conditionals to decide what to do given perception data
    # Here you're all set up with some basic functionality but you'll need to
    # improve on this decision tree to do a good job of navigating autonomously!

    # don't go to rock for these objectives
    if (Rover.objective == 'search_only'):
        Rover.rock_pos = None
    if (Rover.objective == 'go_home'):
        Rover.rock_pos = None

    # Example:
    # Check if we have vision data to make decisions with
    if Rover.cam_nav_angles is not None:
		# check if Rover is stuck
		# last position was when mod 20 = 0, check when mod 10 = 5
        if (Rover.mode == 'forward') or (Rover.mode == 'forwardtorock') or (Rover.mode == 'unstuck') or (Rover.mode == 'unstuckanti'):
                    if (Rover.mode == 'unstuck') and (Rover.unstuck_count > Rover.unstuck_count_target):
                        Rover.unstuck_target_yaw = turn_angle(Rover.yaw, Rover.unstuck_turn_angle)
                        Rover.unstuck_count = 0
                        Rover.mode = 'unstuckanti'
                    elif (Rover.mode == 'unstuckanti') and (Rover.unstuck_count > Rover.unstuck_count_target):
                        Rover.unstuck_count = 0
                        Rover.unstuck_target_yaw = turn_angle(Rover.yaw, - Rover.unstuck_turn_angle)
                        Rover.mode = 'unstuck'
                    elif (Rover.unstuck_count > Rover.unstuck_count_target):
                        Rover.unstuck_count = 0
                        # use camera info to decide turning
                        if (np.mean(Rover.cam_nav_angles * 180/np.pi) > 10):
                            Rover.unstuck_target_yaw = turn_angle(Rover.yaw, Rover.unstuck_turn_angle)
                            Rover.unstuck_count = 0
                            Rover.mode = 'unstuckanti'
                        else:
                            Rover.unstuck_target_yaw = turn_angle(Rover.yaw, - Rover.unstuck_turn_angle)
                            Rover.unstuck_count = 0
                            Rover.mode = 'unstuck'

        # Check for Rover.mode status
        if Rover.mode == 'forward':
            if (Rover.near_sample == 1):
                    Rover.throttle = 0
                    # Set brake to stored brake value
                    Rover.brake = Rover.brake_set
                    Rover.steer = 0
                    Rover.mode = 'trypick'
            # Check the extent of navigable terrain
            elif Rover.rock_pos is not None:
                    Rover.throttle = 0
                    # Set brake to stored brake value
                    Rover.brake = Rover.brake_set
                    Rover.steer = 0
                    Rover.unstuck_count = 0
                    Rover.objective = 'pick_rock'
                    Rover.target_rock_pos = Rover.rock_pos
                    Rover.initial_rock_distance = np.linalg.norm(Rover.pos - Rover.target_rock_pos)
                    Rover.mode = 'turntorock'

            # If there's a lack of navigable terrain pixels then go to 'stop' mode
            elif (len(Rover.cam_nav_angles) < Rover.stop_forward):
                    # Set mode to "stop" and hit the brakes!
                    Rover.throttle = 0
                    # Set brake to stored brake value
                    Rover.brake = Rover.brake_set
                    Rover.steer = 0
                    Rover.mode = 'stop'

            # if front got obstacle, unstuck. put this as special case in case we want different action
            elif (Rover.rock_in_front == 1):
                    # Set mode to "stop" and hit the brakes!
                    Rover.throttle = 0
                    # Set brake to stored brake value
                    Rover.brake = Rover.brake_set
                    Rover.steer = 0
                    Rover.unstuck_target_yaw = turn_angle(Rover.yaw, - (Rover.unstuck_turn_angle / 2))
                    #Rover.unstuck_count = 0
                    Rover.mode = 'unstuck'

            # if only left got obstacle, unstuck half
            elif (Rover.rock_in_front_left == 1):
                    # Set mode to "stop" and hit the brakes!
                    Rover.throttle = 0
                    # Set brake to stored brake value
                    Rover.brake = Rover.brake_set
                    Rover.steer = 0
                    Rover.unstuck_target_yaw = turn_angle(Rover.yaw, - (Rover.unstuck_turn_angle / 2))
                    #Rover.unstuck_count = 0
                    Rover.mode = 'unstuck'

            # if only right got obstacle, unstuckanti half
            elif (Rover.rock_in_front_right == 1):
                    # Set mode to "stop" and hit the brakes!
                    Rover.throttle = 0
                    # Set brake to stored brake value
                    Rover.brake = Rover.brake_set
                    Rover.steer = 0
                    Rover.unstuck_target_yaw = turn_angle(Rover.yaw, (Rover.unstuck_turn_angle / 2))
                    #Rover.unstuck_count = 0
                    Rover.mode = 'unstuckanti'


            elif len(Rover.cam_nav_angles) >= Rover.stop_forward:  
                # If mode is forward, navigable terrain looks good 
                # and velocity is below max, then throttle 
                if Rover.vel < Rover.max_vel:
                    # Set throttle value to throttle setting
                    Rover.throttle = Rover.throttle_set
                else: # Else coast
                    Rover.throttle = 0
                Rover.brake = 0
                # Set steering to average angle clipped to the range +/- 15
                # "keep to the left": shift +1 std_dev will face it more anti-clockwise to the mean direction
                #  + 0.2 can reach whole map. + 0.4 look good. - 0.5 not good.
                mean_dir = np.mean(Rover.nav_angles)
                std_dev = np.std(Rover.nav_angles)
                target_angle_rad = mean_dir + 0.2 * std_dev
                Rover.steer = np.clip(target_angle_rad * 180/np.pi, -10, 10)
                if (abs(Rover.throttle) == Rover.throttle_set) and (abs(Rover.vel) < 0.1):
                   Rover.unstuck_count += 3

            if (Rover.objective == 'go_home'):
                home_dist = distance_from_home(Rover)
                Rover.go_home_count += 1
                # stay at home if Rover is home
                if (home_dist < Rover.home_park_distance):
                    Rover.throttle = 0
                    Rover.brake = Rover.brake_set
                    Rover.steer = 0
                    Rover.mode = 'stop'
                else:
                    if (Rover.go_home_count > Rover.go_home_count_target):
                        Rover.throttle = 0
                        Rover.unstuck_count = 0
                        Rover.brake = Rover.brake_set
                        Rover.steer = 0
                        Rover.go_home_count = 0
                        Rover.mode = 'turntohome'
        # If we're already in "stop" mode then make different decisions
        elif Rover.mode == 'stop':
            if (Rover.objective == 'go_home') or (Rover.objective == 'Reached Home!!'):
                home_dist = distance_from_home(Rover)
                # stay at home if Rover is home
                if (home_dist < Rover.home_park_distance):
                    Rover.throttle = 0
                    Rover.brake = Rover.brake_set
                    Rover.steer = 0
                    Rover.objective = 'Reached Home!!'
                else:
                    Rover.mode = 'turntohome'
            elif (Rover.near_sample == 1):
                    Rover.throttle = 0
                    # Set brake to stored brake value
                    Rover.brake = Rover.brake_set
                    Rover.steer = 0
                    Rover.mode = 'trypick'
            elif Rover.rock_pos is not None:
                    Rover.throttle = 0
                    # Set brake to stored brake value
                    Rover.brake = Rover.brake_set
                    Rover.steer = 0
                    Rover.unstuck_count = 0
                    Rover.objective = 'pick_rock'
                    Rover.target_rock_pos = Rover.rock_pos
                    Rover.initial_rock_distance = np.linalg.norm(Rover.pos - Rover.target_rock_pos)
                    Rover.mode = 'turntorock'
            # If we're in stop mode but still moving keep braking
            elif Rover.vel > 0.2:
                Rover.throttle = 0
                Rover.brake = Rover.brake_set
                Rover.steer = 0
            # If we're not moving (vel < 0.2) then do something else
            elif Rover.vel <= 0.2:
                # Now we're stopped and we have vision data to see if there's a path forward
                if (Rover.rock_in_front == 1):
                    Rover.throttle = 0
                    # Release the brake to allow turning
                    Rover.brake = 0
                    # Turn range is +/- 15 degrees, when stopped the next line will induce 4-wheel turning
                    Rover.steer = -15 # Could be more clever here about which way to turn
                elif len(Rover.cam_nav_angles) < Rover.go_forward:
                    Rover.throttle = 0
                    # Release the brake to allow turning
                    Rover.brake = 0
                    # Turn range is +/- 15 degrees, when stopped the next line will induce 4-wheel turning
                    Rover.steer = -15 # Could be more clever here about which way to turn
                # If we're stopped but see sufficient navigable terrain in front then go!
                if len(Rover.cam_nav_angles) >= Rover.go_forward:
                    # Set throttle back to stored value
                    Rover.throttle = Rover.throttle_set
                    # Release the brake
                    Rover.brake = 0
                    # Set steer to mean angle
                    Rover.steer = np.clip(np.mean(Rover.cam_nav_angles * 180/np.pi), -15, 15)
                    Rover.mode = 'forward'

        # in this mode, we don't care about other things and just go forward for 50 frames
        elif Rover.mode == 'forward_awhile':
            if (Rover.near_sample == 1):
                    Rover.throttle = 0
                    # Set brake to stored brake value
                    Rover.unstuck_count = 0
                    Rover.brake = Rover.brake_set
                    Rover.steer = 0
                    Rover.mode = 'trypick'
            else:
                Rover.unstuck_count += 1
                Rover.throttle = Rover.throttle_set
                Rover.brake = 0
                if (Rover.unstuck_count > Rover.unstuck_awhile_count_target):
                    Rover.unstuck_count = 0
                    Rover.mode = 'forward'
        # in this mode, we don't care about other things and just go back for 50 frames
        elif Rover.mode == 'back_awhile':
            if (Rover.near_sample == 1):
                    Rover.throttle = 0
                    # Set brake to stored brake value
                    Rover.unstuck_count = 0
                    Rover.brake = Rover.brake_set
                    Rover.steer = 0
                    Rover.mode = 'trypick'
            else:
                Rover.unstuck_count += 1
                Rover.throttle = -Rover.throttle_set
                Rover.brake = 0
                if (Rover.unstuck_count > Rover.unstuck_awhile_count_target):
                    Rover.unstuck_count = 0
                    Rover.mode = 'forward'
        elif Rover.mode == 'turntorock':
            Rover.unstuck_count += 1
            if (Rover.unstuck_count > Rover.unstuck_getout_count_target):
                Rover.unstuck_count = 0
                Rover.mode = 'forward_awhile'
                Rover.throttle = Rover.throttle_set
                Rover.brake = 0
            elif Rover.vel > 0.2:
                Rover.throttle = 0
                Rover.brake = Rover.brake_set
                Rover.steer = 0
            elif Rover.vel <= 0.2:
                Rover.brake = 0
                if (Rover.rock_direction > 0):
                    Rover.steer = -Rover.search_steering
                elif (Rover.rock_direction < 0):
                    Rover.steer = Rover.search_steering
                else:
                    Rover.steer = 0
                    Rover.unstuck_count = 0
                    Rover.mode = 'forwardtorock'
        elif Rover.mode == 'forwardtorock':
            Rover.unstuck_count += 1
            rock_distance = np.linalg.norm(Rover.pos - Rover.target_rock_pos)
            if (Rover.near_sample == 1):
                Rover.throttle = 0
                Rover.brake = Rover.brake_set
                Rover.steer = 0
                Rover.unstuck_count = 0
                Rover.mode = 'trypick'
            # give up if drift to far 
            elif (rock_distance > (Rover.initial_rock_distance +2)):
                Rover.objective = 'search'
                Rover.mode = 'forward'
            elif (Rover.unstuck_count > Rover.unstuck_torock_count_target):
                Rover.unstuck_count = 0
                if len(Rover.cam_nav_angles) >= Rover.go_forward:
                    Rover.mode = 'forward_awhile'
                else:
                    Rover.mode = 'back_awhile'
                Rover.throttle = Rover.throttle_set
                Rover.brake = 0
            elif Rover.vel > 0.8:
                Rover.throttle = 0
                Rover.brake = Rover.brake_set
                Rover.steer = 0
            elif Rover.vel > 0.6:
                Rover.throttle = 0
                Rover.brake = 0
                Rover.steer = 0
            elif Rover.vel <= 0.4:
                Rover.throttle = 0.2
                Rover.brake = 0
                Rover.steer = 0
        elif Rover.mode == 'trypick':
            if (Rover.near_sample == 1) and (Rover.vel == 0):
                Rover.send_pickup = True
                if (Rover.objective == 'pick_rock'):
                    Rover.objective = 'search'
                    Rover.target_rock_pos = None
                    Rover.initial_rock_distance = 9999
                    Rover.rocks_picked += 1
                Rover.mode = 'pickingup'
            elif (Rover.near_sample == 0):
                Rover.mode = 'forward'
        elif Rover.mode == 'pickingup':
            # if Rover mode exit picking_up, change mode to 'forward'
            # and update picked count
            if (Rover.picking_up == 0):
                Rover.unstuck_count = 0
                Rover.mode = 'forward'
        elif Rover.mode == 'unstuck':
            if Rover.vel > 0.2:
                Rover.throttle = 0
                Rover.brake = Rover.brake_set
                Rover.steer = 0
            elif Rover.vel <= 0.2:
                Rover.throttle = 0
                Rover.brake = 0
                Rover.steer = -Rover.unstuck_steering
                if (diff_angle(Rover.unstuck_target_yaw, Rover.yaw) < 10):
                    Rover.steer = 0
                    Rover.unstuckmap[int(Rover.pos[0]), int(Rover.pos[1])] += 1
                    # 3 seconds of compensation for being stuck
                    if (Rover.objective == 'go_home'):
                        Rover.go_home_count -= Rover.used_fps * 3
                    Rover.mode = 'forward'
        elif Rover.mode == 'unstuckanti':
            if (Rover.unstuckmap[int(Rover.pos[0]), int(Rover.pos[1])] > Rover.unstucklimit):
                Rover.unstuck_target_yaw = turn_angle(Rover.yaw, -40)
                Rover.mode = 'unstuck'
            elif Rover.vel > 0.2:
                Rover.throttle = 0
                Rover.brake = Rover.brake_set
                Rover.steer = 0
            elif Rover.vel <= 0.2:
                Rover.throttle = 0
                Rover.brake = 0
                Rover.steer = Rover.unstuck_steering
                if (diff_angle(Rover.unstuck_target_yaw, Rover.yaw) < 10):
                    Rover.steer = 0
                    Rover.unstuckmap[int(Rover.pos[0]), int(Rover.pos[1])] += 1
                    # 3 seconds of compensation for being stuck
                    if (Rover.objective == 'go_home'):
                        Rover.go_home_count -= Rover.used_fps * 3
                    Rover.mode = 'forward'

        elif Rover.mode == 'turntohome':
          if Rover.vel > 0.2:
            Rover.throttle = 0
            Rover.brake = Rover.brake_set
            Rover.steer = 0
          elif Rover.vel <= 0.2:
            Rover.brake = 0
            homeangle = home_angle(Rover)
            diffangle = (homeangle - Rover.yaw)
            if (diffangle > 180):
                diffangle -= 360
            if (diffangle < -180):
                diffangle += 360
            if (diff_angle(homeangle, Rover.yaw) < 10):
                Rover.throttle = 0
                Rover.brake = Rover.brake_set
                Rover.steer = 0
                Rover.unstuck_count = 0
                # go_home_count_target must be more than 4 seconds because starts from stationary
                target_sec = distance_from_home(Rover) / 1.7
                if (target_sec < 5.0):
                    target_sec = 5
                Rover.go_home_count_target = int(Rover.used_fps * target_sec)
                Rover.go_home_count = 0
                # We may be stuck in loop if a wall is in front of the home angle
                # check navigation availbility before going forward
                if (Rover.rock_in_front == 1):
                    Rover.mode = 'unstuck'
                elif (Rover.rock_in_front_left == 1):
                    Rover.mode = 'unstuck'
                elif (Rover.rock_in_front_right == 1):
                    Rover.mode = 'unstuck'
                elif (len(Rover.cam_nav_angles) < Rover.stop_forward):
                    Rover.mode = 'unstuck'
                else:
                    Rover.mode = 'forward'
            elif (diffangle > 0) :
                Rover.throttle = 0
                Rover.brake = 0
                Rover.steer = Rover.search_steering
            else:
                Rover.throttle = 0
                Rover.brake = 0
                Rover.steer = -Rover.search_steering
    # Just to make the rover do something 
    # even if no modifications have been made to the code
    else:
        Rover.throttle = Rover.throttle_set
        Rover.steer = 0
        Rover.brake = 0

    return Rover

