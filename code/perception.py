import numpy as np
import cv2

import matplotlib.pyplot as plt
from decision import distance_from_home, home_angle, diff_angle

# Identify pixels above the threshold
# Threshold of RGB > 160 does a nice job of identifying ground pixels only
def color_thresh(img, rgb_thresh=(160, 160, 160)):
    # Create an array of zeros same xy size as img, but single channel
    color_select = np.zeros_like(img[:,:,0])
    # Require that each pixel be above all three threshold values in RGB
    # above_thresh will now contain a boolean array with "True"
    # where threshold was met
    above_thresh = (img[:,:,0] > rgb_thresh[0]) \
                & (img[:,:,1] > rgb_thresh[1]) \
                & (img[:,:,2] > rgb_thresh[2])
    # Index the array of zeros with the boolean array and set to 1
    color_select[above_thresh] = 1
    # Return the binary image
    return color_select


# color below threshold
# for detecting walls
def color_below_thresh(img, rgb_thresh=(160, 160, 160)):
    # Create an array of zeros same xy size as img, but single channel
    color_select = np.zeros_like(img[:,:,0])
    # Require that each pixel be above all three threshold values in RGB
    # above_thresh will now contain a boolean array with "True"
    # where threshold was met
    below_thresh = (img[:,:,0] < rgb_thresh[0]) \
                & (img[:,:,1] < rgb_thresh[1]) \
                & (img[:,:,2] < rgb_thresh[2])
    # Index the array of zeros with the boolean array and set to 1
    color_select[below_thresh] = 1
    # Return the binary image
    return color_select

# For detecting rock
# use it on original image, not wraped image
# color for rock c6b027 
# RG above thresh, B below thresh: low noise, small: 160, 160, 110
# Other suggestion: 100, 100, 75
def color_rock_thresh(img, rgb_thresh=(140, 140, 110)):
    # Create an array of zeros same xy size as img, but single channel
    color_select = np.zeros_like(img[:,:,0])
    # Require that each pixel be above all three threshold values in RGB
    # above_thresh will now contain a boolean array with "True"
    # where threshold was met
    rock_thresh = (img[:,:,0] > rgb_thresh[0]) \
                & (img[:,:,1] > rgb_thresh[1]) \
                & (img[:,:,2] < rgb_thresh[2])
    # Index the array of zeros with the boolean array and set to 1
    color_select[rock_thresh] = 1
    # Return the binary image
    return color_select

# Define a function to convert to rover-centric coordinates
def rover_coords(binary_img):
    # Identify nonzero pixels
    ypos, xpos = binary_img.nonzero()
    # Calculate pixel positions with reference to the rover position being at the 
    # center bottom of the image.  
    x_pixel = np.absolute(ypos - binary_img.shape[0]).astype(np.float)
    y_pixel = -(xpos - binary_img.shape[0]).astype(np.float)
    return x_pixel, y_pixel


# Define a function to convert to radial coords in rover space
def to_polar_coords(x_pixel, y_pixel):
    # Convert (x_pixel, y_pixel) to (distance, angle) 
    # in polar coordinates in rover space
    # Calculate distance to each pixel
    dist = np.sqrt(x_pixel**2 + y_pixel**2)
    # Calculate angle away from vertical for each pixel
    angles = np.arctan2(y_pixel, x_pixel)
    return dist, angles

# Define a function to apply a rotation to pixel positions
def rotate_pix(xpix, ypix, yaw):
    # TODO:
    # Convert yaw to radians
    # Apply a rotation
    yaw_rad = np.radians(yaw)
    xpix_rotated = xpix * np.cos(yaw_rad) - ypix * np.sin(yaw_rad)
    ypix_rotated = xpix * np.sin(yaw_rad) + ypix * np.cos(yaw_rad)
    # Return the result  
    return xpix_rotated, ypix_rotated

# Define a function to perform a translation
def translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale): 
    # TODO:
    # Apply a scaling and a translation
    xpix_translated = np.int_(xpos + (xpix_rot / scale))
    ypix_translated = np.int_(ypos + (ypix_rot / scale))
    # Return the result  
    return xpix_translated, ypix_translated

# Define a function to apply rotation and translation (and clipping)
# Once you define the two functions above this function should work
def pix_to_world(xpix, ypix, xpos, ypos, yaw, world_size, scale):
    # Apply rotation
    xpix_rot, ypix_rot = rotate_pix(xpix, ypix, yaw)
    # Apply translation
    xpix_tran, ypix_tran = translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale)
    # Perform rotation, translation and clipping all at once
    x_pix_world = np.clip(np.int_(xpix_tran), 0, world_size - 1)
    y_pix_world = np.clip(np.int_(ypix_tran), 0, world_size - 1)
    # Return the result
    return x_pix_world, y_pix_world

# Define a function to perform a perspective transform
def perspect_transform(img, src, dst):
           
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))# keep same size as input image
    
    return warped

def mark_nav(Rover, y, x):
    if (Rover.seenmap[y, x] > 10):
            Rover.navmap[y, x] = 100
            Rover.wallmap[y, x] = 0
    if (Rover.navmap[y, x] == 0) and (Rover.obstaclemap[y, x] > 20):
        Rover.wallmap[y, x] += 1
    return


# Apply the above functions in succession and update the Rover state accordingly
def perception_step(Rover):
    # Perform perception steps to update Rover()
    # TODO: 
    # NOTE: camera image is coming to you in Rover.img
    # 1) Define source and destination points for perspective transform
    dst_size = 5
    scale = 10
    bottom_offset = 6
    source = np.float32([[14, 140], [301 ,140],[200, 96], [118, 96]])
    destination = np.float32([[Rover.img.shape[1]/2 - dst_size, Rover.img.shape[0] - bottom_offset],
                  [Rover.img.shape[1]/2 + dst_size, Rover.img.shape[0] - bottom_offset],
                  [Rover.img.shape[1]/2 + dst_size, Rover.img.shape[0] - 2*dst_size - bottom_offset], 
                  [Rover.img.shape[1]/2 - dst_size, Rover.img.shape[0] - 2*dst_size - bottom_offset],
                  ])

    # 2) Apply perspective transform
    # 3) Apply color threshold to identify navigable terrain/obstacles/rock samples
    # 4) Update Rover.vision_image (this will be displayed on left side of screen)
        # Example: Rover.vision_image[:,:,0] = obstacle color-thresholded binary image
        #          Rover.vision_image[:,:,1] = rock_sample color-thresholded binary image
        #          Rover.vision_image[:,:,2] = navigable terrain color-thresholded binary image
    if (Rover.pos is None) or (len(Rover.pos) < 2):
       return Rover        

    rover_xpos = Rover.pos[0]
    rover_ypos = Rover.pos[1]
    rover_yaw = Rover.yaw

    # keep target yaw sane
    Rover.unstuck_target_yaw = Rover.unstuck_target_yaw % 360;

    # record the start position once only
    if (Rover.start_pos is None):
        Rover.start_pos = (rover_xpos, rover_ypos)
    if (Rover.prevposition is None):
        Rover.prevposition = (rover_xpos, rover_ypos)
        Rover.unstuck_count = 0

    imgsize = Rover.img.shape
    # print("Image Size " + str(imgsize[0]) + " , " + str(imgsize[1]))
    warped = perspect_transform(Rover.img, source, destination)
    colorsel = color_thresh(warped, rgb_thresh=(160, 160, 160))
    xpix, ypix = rover_coords(colorsel)
    navigable_x_world, navigable_y_world = pix_to_world(xpix, ypix, rover_xpos, 
                                rover_ypos, rover_yaw, 
                                Rover.worldmap.shape[0], scale)

    # get a map for "new" navigable world
    new_xpix_i = []
    new_ypix_i = []

    # rock in front detection
    Rover.rock_in_front = 0
    Rover.rock_in_front_left = 0
    Rover.rock_in_front_right = 0

    # only left
    rif_clip_left = colorsel[90:150,150:160]
    rif_clip_left_ypos, rif_clip_left_xpos = rif_clip_left.nonzero()
    if (len(rif_clip_left_ypos) < Rover.rock_in_front_thresh):
        Rover.rock_in_front_left = 1

    # only right
    rif_clip_right = colorsel[90:150:,160:170]
    rif_clip_right_ypos, rif_clip_right_xpos = rif_clip_right.nonzero()
    if (len(rif_clip_right_ypos) < Rover.rock_in_front_thresh):
        Rover.rock_in_front_right = 1

    if (Rover.rock_in_front_right == 1) and (Rover.rock_in_front_left == 1):
        Rover.rock_in_front = 1

# base on view
#    for idx in range(len(navigable_x_world)):
#        if (Rover.seenmap[navigable_y_world[idx], navigable_x_world[idx]] < Rover.new_pix_threshold):
#            new_xpix_i.append(xpix[idx])
#            new_ypix_i.append(ypix[idx])

# base on real travelled. Only those not travelled x,y are present in new_xpix, new_ypix
    for idx in range(len(navigable_x_world)):
        if (Rover.navmap[navigable_y_world[idx], navigable_x_world[idx]] == 0):
            new_xpix_i.append(xpix[idx])
            new_ypix_i.append(ypix[idx])

    new_xpix = np.array(new_xpix_i)
    new_ypix = np.array(new_ypix_i)

    # for obstacles
    obstacles_warped = perspect_transform(Rover.img, source, destination)
    obstacles_colorsel = color_below_thresh(obstacles_warped, rgb_thresh=(160, 160, 160))
    obstacles_xpix, obstacles_ypix = rover_coords(obstacles_colorsel)
    obstacles_x_world, obstacles_y_world = pix_to_world(obstacles_xpix, obstacles_ypix, rover_xpos, 
                                rover_ypos, rover_yaw, 
                                Rover.worldmap.shape[0], scale)

    # for rock, have to use the original image for detection
    # use the default, so change at the function definition
    rock_colorsel = color_rock_thresh(Rover.img)

    # filter out rock if there are 2 rocks in view
    # remove all to the right of mean
    rock_color_ypos, rock_color_xpos = rock_colorsel.nonzero()
    if (len(rock_color_xpos) >= 1):
        rock_color_xpos_mean = np.mean(rock_color_xpos)
        rock_color_xpos_std = np.std(rock_color_xpos)
        if (rock_color_xpos_std > 3):
            rock_colorsel[int(rock_color_xpos_mean):, :] = 0

    rock_warped = perspect_transform(rock_colorsel, source, destination)
    rock_xpix, rock_ypix = rover_coords(rock_warped)
    rock_x_world, rock_y_world = pix_to_world(rock_xpix, rock_ypix, rover_xpos, 
                                rover_ypos, rover_yaw, 
                                Rover.worldmap.shape[0], scale)

    to_get_rock = 0
    if (len(rock_color_ypos) > Rover.rock_threshold) and (len(rock_ypix) > 1):
       to_get_rock = 1
       Rover.worldmap[rock_y_world, rock_x_world, 1] += 1

    # if we are not in stop or forward mode, even 1 pixel is important
#    if (len(rock_x_world) >= 1):
#       to_get_rock = 1
#       Rover.worldmap[rock_y_world, rock_x_world, 1] += 1

#    if (len(rock_x_world) < Rover.rock_threshold):
#       if Rover.mode == 'stop':
#          to_get_rock = 0
#       if Rover.mode == 'forward':
#          to_get_rock = 0

    # if 2 rocks in the same view, this will screw up
    # use std dev to check if there are 2 rocks
    # simple way: remove those smaller than mean 
    rock_ypix_mean = 0
    rock_ypix_std = 0
    if (to_get_rock > 0):
        rock_ypix_mean = np.mean(rock_ypix)
        print("Rock Ypix Mean = ", rock_ypix_mean)
        if (rock_ypix_mean > 5):
            Rover.rock_direction = -1
        elif (rock_ypix_mean < -5):
            Rover.rock_direction = 1
        else:
            Rover.rock_direction = 0
        Rover.rock_pos = (np.mean(rock_x_world), np.mean(rock_y_world))
    else:
        Rover.rock_pos = None


    # vision_image is (160, 320)
    # world map is (200, 200)
    Rover.vision_image[:,:,:] = 0

    # navpos is used to update worldmap
    navypos, navxpos = Rover.navmap.nonzero()

    # squeeze navmap into vision_image
    squeeze_navypos = navypos / 200 * 160
    squeeze_navxpos = navxpos / 200 * 160
    # flip the y axis for display consitent with world map
    #Rover.vision_image[160 - squeeze_navypos.astype(int),squeeze_navxpos.astype(int),1] = 200

    # squeeze seenmap into vision_image
    seenypos, seenxpos = Rover.seenmap.nonzero()
    squeeze_seenypos = seenypos / 200 * 160
    squeeze_seenxpos = seenxpos / 200 * 160
    # flip the y axis for display consitent with world map
#Rover.vision_image[160 - squeeze_seenypos.astype(int),squeeze_seenxpos.astype(int),0] = 200

    # squeeze wallmap into vision_image
    wallypos, wallxpos = Rover.wallmap.nonzero()
    squeeze_wallypos = wallypos / 200 * 160
    squeeze_wallxpos = wallxpos / 200 * 160
    # flip the y axis for display consitent with world map
    Rover.vision_image[160 - squeeze_wallypos.astype(int),squeeze_wallxpos.astype(int),0] = 200
    
    #squeeze_new_ypix = new_ypix / 200 * 160
    #squeeze_new_xpix = new_xpix / 200 * 160
    #Rover.vision_image[squeeze_new_ypix.astype(int),squeeze_new_xpix.astype(int),0] = 200

    Rover.vision_image[:,:,0] = rock_colorsel * 200
    Rover.vision_image[:,:,1] = colorsel * 200


    #if (int(Rover.total_time) > Rover.screen_update_time):
    modedisplay = ''
    modedisplay2 = ''
    modedisplay3 = ''
    modedisplay4 = ''
    modedisplay5 = ''
    modedisplay6 = ''
    modedisplay7 = ''

    if (Rover.mode == 'turntorock'):
        modedisplay3 = "Rock pixels = {0}".format(len(rock_x_world))
        modedisplay4 = "Rock Ypix Mean = {0:.2f}".format(rock_ypix_mean)
        modedisplay5 = "Rock Ypix Std = {0:.2f}".format(rock_ypix_std)
        if (Rover.rock_pos is not None):
            modedisplay6 = "Rock Position = {0:.2f} , {1:.2f}".format(Rover.rock_pos[0], Rover.rock_pos[1])
            modedisplay7 = "Rock Distance = {0:.2f}".format(np.linalg.norm(Rover.pos - Rover.rock_pos))
    elif (Rover.mode == 'unstuck') or (Rover.mode == 'unstuckanti'):
        modedisplay3 = "Target Yaw = {0:.2f}".format(Rover.unstuck_target_yaw)
        modedisplay4 = "Now Yaw = {0:.2f}".format(Rover.yaw)

    elif (Rover.objective == 'go_home') or (Rover.objective == 'Reached Home!!'):
        modedisplay3 = "Distance From Home = {0:.2f}".format(distance_from_home(Rover))
        time_to_turn = (Rover.go_home_count_target - Rover.go_home_count) / Rover.used_fps
        modedisplay4 = "Turn to home timer = {0:.2f}".format(time_to_turn)
    else:
        modedisplay3 = "Rock picked = {0}".format(Rover.rocks_picked) + ' Target Rocks = ' + str(Rover.target_rocks)

    if 1 == 1:
        Rover.screen_update_time = int(Rover.total_time)
        modedisplay = 'Objective = ' + Rover.objective 
        modedisplay2 = 'FPS = ' + str(Rover.used_fps) + ' Mode = ' + Rover.mode
        cv2.putText(Rover.vision_image,modedisplay, (0, 15), 
                  cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(Rover.vision_image,modedisplay2, (0, 30), 
                  cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(Rover.vision_image,modedisplay3, (0, 45), 
                  cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(Rover.vision_image,modedisplay4, (0, 60), 
                  cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(Rover.vision_image,modedisplay5, (0, 75), 
                  cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(Rover.vision_image,modedisplay6, (0, 90), 
                  cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(Rover.vision_image,modedisplay7, (0, 105), 
                  cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)


    #over.vision_image = warped
   
    # 5) Convert map image pixel values to rover-centric coords
    # 6) Convert rover-centric pixel values to world coordinates
    # 7) Update Rover worldmap (to be displayed on right side of screen)
        # Example: Rover.worldmap[obstacle_y_world, obstacle_x_world, 0] += 1
        #          Rover.worldmap[rock_y_world, rock_x_world, 1] += 1
        #          Rover.worldmap[navigable_y_world, navigable_x_world, 2] += 1

    can_update_map = 1
    if (diff_angle(Rover.roll, 0) > Rover.acceptable_roll):
        can_update_map = 0
    if (diff_angle(Rover.pitch, 0) > Rover.acceptable_roll):
        can_update_map = 0

    if (can_update_map == 1):
        Rover.obstaclemap[obstacles_y_world, obstacles_x_world] += 1        
        Rover.seenmap[navigable_y_world, navigable_x_world] += 1

    # remove nav points from obstaclemap
    Rover.obstaclemap[navypos, navxpos] = 0

    # update worldmap
    # layer 0 is obstacles
    Rover.worldmap[:, :, 0] = Rover.obstaclemap
    # layer 1 is rocks, update above
    # layer 2 is nav
    Rover.worldmap[:,:, 2] = Rover.navmap



    # update position when int(Time) mod 10 = 0
    #if (int(Rover.total_time) % 15) == 0:
    #    Rover.mod10position = Rover.pos
    dist = np.linalg.norm(Rover.pos - Rover.prevposition)
    if (dist > 1.5):
        Rover.prevposition = Rover.pos
        Rover.unstuck_count = 0
    else:
        Rover.unstuck_count += 1

    # current Rover position is always updated
    Rover.seenmap[int(rover_ypos), int(rover_xpos)] += 10
    Rover.navmap[int(rover_ypos), int(rover_xpos)] = 100

    # update the sides in navmap also, if present in navigatable_world
    cur_point_x = int(rover_xpos)
    cur_point_y = int(rover_ypos)
    # we don't care about the edge case, so less things to check
    if ((cur_point_x > 0) and (cur_point_x < 200) and (cur_point_y > 0) and (cur_point_y < 200)):
        mark_nav(Rover, cur_point_y, cur_point_x-1)
        mark_nav(Rover, cur_point_y, cur_point_x+1)
        mark_nav(Rover, cur_point_y-1, cur_point_x-1)
        mark_nav(Rover, cur_point_y-1, cur_point_x+1)
        mark_nav(Rover, cur_point_y-1, cur_point_x)
        mark_nav(Rover, cur_point_y+1, cur_point_x-1)
        mark_nav(Rover, cur_point_y+1, cur_point_x)
        mark_nav(Rover, cur_point_y+1, cur_point_x+1)

    # do the +2 also
    if ((cur_point_x > 1) and (cur_point_x < 199) and (cur_point_y > 1) and (cur_point_y < 199)):
        mark_nav(Rover, cur_point_y, cur_point_x-2)
        mark_nav(Rover, cur_point_y, cur_point_x+2)
        mark_nav(Rover, cur_point_y-1, cur_point_x-2)
        mark_nav(Rover, cur_point_y-1, cur_point_x+2)
        mark_nav(Rover, cur_point_y+1, cur_point_x-2)
        mark_nav(Rover, cur_point_y+1, cur_point_x+2)
        mark_nav(Rover, cur_point_y-2, cur_point_x-2)
        mark_nav(Rover, cur_point_y-2, cur_point_x-1)
        mark_nav(Rover, cur_point_y-2, cur_point_x)
        mark_nav(Rover, cur_point_y-2, cur_point_x+1)
        mark_nav(Rover, cur_point_y-2, cur_point_x+2)
        mark_nav(Rover, cur_point_y+2, cur_point_x-2)
        mark_nav(Rover, cur_point_y+2, cur_point_x-1)
        mark_nav(Rover, cur_point_y+2, cur_point_x)
        mark_nav(Rover, cur_point_y+2, cur_point_x+1)
        mark_nav(Rover, cur_point_y+2, cur_point_x+2)


    # 8) Convert rover-centric pixel positions to polar coordinates
    # Update Rover pixel distances and angles
        # Rover.nav_dists = rover_centric_pixel_distances
        # Rover.nav_angles = rover_centric_angles
    
    # we use this for some decision making in Forward mode
    Rover.cam_nav_dists, Rover.cam_nav_angles = to_polar_coords(xpix, ypix)

    # using (xpix, ypix) means using the current camera view to determine nav path
    # using (new_xpix, new_ypix) means using the unmapped area of current camera view to determine nav path
    if (len(new_xpix) > 10):
        Rover.nav_dists, Rover.nav_angles = to_polar_coords(new_xpix, new_ypix)
    else:
        Rover.nav_dists, Rover.nav_angles = to_polar_coords(xpix, ypix)

    if (Rover.rocks_picked >= Rover.target_rocks):
        Rover.objective = 'go_home'


   
    return Rover


        #fig = plt.figure(figsize=(3.2,1.6))
        #fig.add_subplot(111)
        #plt.axis('off')
        #fig.text(0.05, 0.1, modedisplay)
        #fig.canvas.draw()
        #data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        #data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        #data = 255 - data
        #Rover.vision_image[:,:,:] = data
