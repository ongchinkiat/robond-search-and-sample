## Udacity Robotics Nanodegree Project 1

# Search and Sample Return

## Rover Project Test Notebook

1. Notebook File: /code/Rover_Project_Test_Notebook.ipynb
2. HTML: /Rover_Project_Test_Notebook.html
3. Output Video: /test_mapping.mp4

### Notebook Analysis 1: add obstacle and rock sample identification

For navigable path identification, first do a Prespect Transform to get the warped image. Then apply color threshold for "R,G,B > 160" to get the color selected warped image. In colorsel, navigable path have 1 value and obstacles have 0 value.

Next 2 steps is to transform the warped image into rover-centric and world co-ordinates.


```
    warped = perspect_transform(img, source, destination)

    # 3) Apply color threshold to identify navigable terrain/obstacles/rock samples
    colorsel = color_thresh(warped, rgb_thresh=(160, 160, 160))

    # 4) Convert thresholded image pixel values to rover-centric coords
    xpix, ypix = rover_coords(colorsel)

    # 5) Convert rover-centric pixel values to world coords
    navigable_x_world, navigable_y_world = pix_to_world(xpix, ypix, rover_xpos,
                                rover_ypos, rover_yaw,
                                data.worldmap.shape[0], scale)
```

For obstacle identification, we just do the reverse of navigable path identification in the color select step. Instead of color threshold for "R,G,B > 160", we do color threshold for "R,G,B < 160". So in obstacles_colorsel, navigable path have 0 value and obstacles have 1 value.

Similarly, transform the warped obstacles_colorsel image into rover-centric and world co-ordinates.

```
# for obstacles
    obstacles_colorsel = color_below_thresh(warped, rgb_thresh=(160, 160, 160))
    obstacles_xpix, obstacles_ypix = rover_coords(obstacles_colorsel)
    obstacles_x_world, obstacles_y_world = pix_to_world(obstacles_xpix, obstacles_ypix, rover_xpos,
                                rover_ypos, rover_yaw,
                                data.worldmap.shape[0], scale)
```

For rocks, I find that the results are better if we do color select on the original camera image. So the steps are reversed. We do color select first on the original camera image, with color threshold for "R,G > 160, B < 110".

Then we do the Prespect Transform to get rock_warped. And lastly, transform the rock_warped image into rover-centric and world co-ordinates.

```
# for rock, have to use the original image for detection
    rock_colorsel = color_rock_thresh(img, rgb_thresh=(160, 160, 110))
    rock_warped = perspect_transform(rock_colorsel, source, destination)
    rock_xpix, rock_ypix = rover_coords(rock_warped)
    rock_x_world, rock_y_world = pix_to_world(rock_xpix, rock_ypix, rover_xpos,
                                rover_ypos, rover_yaw,
                                data.worldmap.shape[0], scale)
```

### Notebook Analysis 2: process_image() function

In each step of process_image(), we get the current Rover position and yaw from  data.xpos[], data.ypos[], data.yaw[].

The 3 code blocks above for navigable path, obstacles, rocks identification are done next.

Then, we update the worldmap with obstacles in red (channel [0]), navigable path in blue (channel [2]), rocks in yellow (channel [0:2]), rover position in white (channel [:]).

Since this is a short clip, I set the color intensity to 200 or 255, instead of using +1, so that the points are more visible.

```
data.worldmap[obstacles_y_world, obstacles_x_world, 0] = 200
data.worldmap[navigable_y_world, navigable_x_world, 2] = 200

# rock appears yellow in map
data.worldmap[rock_y_world, rock_x_world, 0:2] = 255



# rover position in white
data.worldmap[int(rover_ypos), int(rover_xpos), :] = 255
```

For the movie, I changed the text line to display the Rover position and yaw.
Top left is the camera view, top right is the prespect view. Bottom left is the overlay with ground truth map. Red is obstacle, Blue is path, both combined to appear magenta. White dots are the rover movement path.

Yellow dots is the detected rock. It looks like the rock distance and direction calculated is not accurate. Thus in my simulator code, I constantly re-detect the rock location as the rover goes nearer to the rock.

### Extra helper functions

I also tested out some helper functions using the Notebook.

1. turn_angle(): Add turn_angle to orig, make sure the result is between 0 to 359 degrees.

2. diff_angle(): Given 2 angles in degrees, calculate the absolute difference between the 2 angles. Make sure the result is between 0 to 179 degrees.

### Identifying obstacle immediately infront: plot_obs()

One problem with just using the whole perspect transformed image to navigate is: when there is a rock right in the center, and the sides are clear, our rover will think that it is still good to go forward.

I used a simple detection strategy: we have to make the rover stop when the middle of the perspect transformed image is all black. On the Warped image, clip rectangle from MiddleX - 10 to MiddleX + 10. If all black, stop Rover.

Note: the obs-in-front.jpg, small-rock-in-front.jpg, looks-ok-but-stuck.jpg are copied from the screen capture from the simulator window. They are smaller in size than the original 320x160 images, and the sides are clipped. That is why the clip range below is 140:160 (middle is 150) . For actual use, the range should be 150:170 (middle is 160).



## Autonomous Navigation and Mapping

In this part of the project, we are given some starter and supporting codes for the autonomous navigator, and a Unity Engine based rover simulator for the  navigator to control.

Note: this code runs with the old version of simulator (downloaded on 25 May 2017). The newer versions supporting code for the new simulator changed the data type of Rover.pos, so a simple marge with the new supporting code version will result in multiple "incompatible operations between list and tuple" errors. If you can't find the correct old version of simulator, I can upload it somewhere.

Files:
1. Main Driver Program: /code/drive_rover.py
2. Perception Engine: /code/perception.py
3. Decision Engine: /code/decision.py
4. Supporting Functions: /code/supporting_functions.py

### Perception Engine: perception_step() function

For the perception_step() function, the same 3 identification code blocks as above, for navigable path, obstacles, rocks are applied, together with the necessary supporting functions. I also added the Extra helper functions and "Identifying obstacle immediately infront" function codes.

I fine tuned the rock identification to color threshold "R,G > 140, B < 110", for better performance.

### Special Case: 2 Rock Nearby

For some cases where 2 rocks are near to each other, the camera view may contain 2 rocks. This will cause the rock detection angle to point at the middle between the 2 rocks.

To avoid runnning the rover through the middle and missing both rocks, we detect this by checking the Standard Deviation of the rock pixels. If the Standard Deviation is greater than a threshold, we remove all pixels to the left of the Mean Y-axis value. This will remove the left rock and the rover can go towards the right rock.


### Starting point

The starting point of the Rover is recorded in Rover.start_pos. This is needed so that we know where to go after we pick up all the rocks.
```
# record the start position once only
if (Rover.start_pos is None):
    Rover.start_pos = (rover_xpos, rover_ypos)
```

### Obstacle In Front

For the obstacle in front detection, there are 3 Rover states to update:
1. Rover.rock_in_front
2. Rover.rock_in_front_left
3. Rover.rock_in_front_right

```
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
```

The decision engine will look at these states to decide whether to stop the rover, and which direction to turn to avoid the obstacle. On hindsight, I shouldn't have name them "rock_in_front", it may cause confusion with the Yellow Rock targets we are supposed to pick up.

### Mapping

For mapping, besides the Rover.worldmap and Rover.vision_image, I created 4 extra maps:
1. Rover.navmap : for real positions that Rover has travelled
2. Rover.seenmap : for navigation positions that we guess from camera feeds
3. Rover.obstaclemap : for obstacle positions that we guess from camera feeds
4. Rover.wallmap : for the positions that we are confident is wall

All 6 maps are updated in each run of the perception_step().

To improve map accuracy, there is a Rover.acceptable_roll limit. If the current Rover roll or pitch is greater than Rover.acceptable_roll, we don't update the Rover.seenmap and Rover.obstaclemap.

### Navigation Decision

To make the rover "more likely" to visit places that it haven't visited before, I use this simple strategy:
1. Given the current navigable points (xpix, ypix) we get from the navigatable path detection
2. Remove those points that are already present in Rover.navmap, to get (new_xpix, new_ypix)
3. Use (new_xpix, new_ypix) to calculate the rover navigation distances and angles
4. If there are not enough (new_xpix, new_ypix), fall back to use (xpix, ypix)

```
if (len(new_xpix) > 10):
    Rover.nav_dists, Rover.nav_angles = to_polar_coords(new_xpix, new_ypix)
else:
    Rover.nav_dists, Rover.nav_angles = to_polar_coords(xpix, ypix)
```

In some situations, e.g. getting unstuck or approching rocks, we still need the camera view navigation distance and angles. So we have 2 new states Rover.cam_nav_dists, Rover.cam_nav_angles to record them.

```
    Rover.cam_nav_dists, Rover.cam_nav_angles = to_polar_coords(xpix, ypix)
```

## Decision Engine: decision_step() function

The decision_step() function reads the inputs set by perception_step(), and use them to control the rover.

### Global Mission Objective

The original Rover.mode is used to track the current action state of the rover. To track the higher level "mission state", we use a new state Rover.objective.

There are 5 possible values for Rover.objective:
1. **search** : This is the default start state to search and pick all rocks, once all rocks picked, it will automatically change to "go_home" state.
2. **go_home** : In this state, the rover will try to navigate home. Once reach home, it will automatically change to "Reached Home!!" state.
3. **Reached Home!!** : We are home! Apply brake, and do nothing forever.
4. **search_only** : This is a "for test" state that we can use so that the rover will not get distracted by rocks when we are testing out mapping strategies
5. **pick_rock** : This is a intermediate state for "search". When we detect a rock, we'll go into this state. Once the rock is picked, or we give up picking, we'll go back to "search" state.

There is also a new state variable Rover.target_rocks to specify how many rocks we want the rover to pick. This should be 6 for a full run. If you want a shorter run, you can set it to 1 or 2.

```
if (Rover.rocks_picked >= Rover.target_rocks):
    Rover.objective = 'go_home'
```

To prevent the decision tree from getting into rock handling, we set Rover.rock_pos to None if objective is 'search_only' or 'go_home'.

```
if (Rover.objective == 'search_only'):
    Rover.rock_pos = None
if (Rover.objective == 'go_home'):
    Rover.rock_pos = None
```

### Global Unstuck Strategy

The first step of the decision tree is to detect whether the rover is stuck. There are many ways that the rover can be stuck. Sometimes, it goes into endless loop changing between 2 or more modes, which is difficult to detect. So we need this "global unstuck detection" to get the rover out of complex situations.

We have Rover.unstuck_count_target_sec which we set to the number of seconds we wait before we execute the unstuck routines. This is calculated based on the FPS of the simulator to get Rover.unstuck_count_target.

```
Rover.unstuck_count_target = Rover.unstuck_count_target_sec * fps
```

The detection counter Rover.unstuck_count is increment by 1 in every step. When we're sure that the rover is not stuck, it is reset to 0. When we suspect that the rover is stuck, we increment by more than 1 to make the trigger come sooner.

When Rover.unstuck_count > Rover.unstuck_count_target, the unstuck routines are triggered.

```
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
```

### Forware Mode

This is the mode when the path is clear and we want the rover to move forward.

There are several situations where will trigger us to change mode:
1. **when near a rock** : brake and change mode to 'trypick'
2. **found a rock** : brake and change mode to 'turntorock'
3. **lack of navigable pixels** : brake and change mode to 'stop'
4. **rock in front** : brake and change mode to 'unstuck'
5. **rock in front left** : brake and change mode to 'unstuck'
6. **rock in front right** : brake and change mode to 'unstuckanti'
7. **normal** : change steering base on navigable angles

There is also a special branch for 'go_home' objective, to decide when to 'stop' and when to trigger 'turntohome' again.

### Stop Mode

In 'stop' mode, decisions were made base on these situations:
1. **reached home** : brake and stay, set objective to "Reached Home!!"
2. **when near a rock** : brake and change mode to 'trypick'
3. **found a rock** : brake and change mode to 'turntorock'
4. **rover still moving** : brake and wait
5. **rover stopped** : turn until enough navigable angles, then change mode to 'forward'

### Forward Awhile Mode

There are some situations where sometimes, just ignore every perception signals and just move forward for awhile can help to get out of a stuck sitation. This is why we have this 'Forward Awhile' Mode.

In 'forward_awhile' mode, decisions were made base on these situations:
1. **when near a rock** : brake and change mode to 'trypick' (don't waste the chance to pick a rock)
2. **normal** : move forward

### Back Awhile Mode

There are some situations where sometimes, just ignore every perception signals and just move backwards for awhile can help to get out of a stuck sitation. This is why we have this 'Back Awhile' Mode.

In 'back_awhile' mode, decisions were made base on these situations:
1. **when near a rock** : brake and change mode to 'trypick' (don't waste the chance to pick a rock)
2. **normal** : move backwards

### Turn To Rock Mode

This is the first step when we detect a rock.

In 'turntorock' mode, decisions were made base on these situations:
1. **rover still moving** : brake and wait
2. **rover stopped** : turn left or right until the rover is facing the rock
3. **rover facing rock** : change mode to 'forwardtorock'
4. **Unstuck Triggered** : change mode to 'forward_awhile'

### Forward To Rock Mode

This is the second step when we detect a rock.

In 'forwardtorock' mode, decisions were made base on these situations:
1. **when near a rock** : brake and change mode to 'trypick'
2. **when we drifted too far off** : give up and change mode to 'forward'
3. **Unstuck Triggered** : change mode to 'forward_awhile' or 'back_awhile'
4. **normal** : maintain velocity between 0.4 to 0.8

### Try Pick Mode

This is the third step when we detect a rock.

In 'trypick' mode, decisions were made base on these situations:
1. **when near a rock but stil moving** : wait
2. **when near a rock and stopped moving** : change mode to 'pickingup'
3. **not near a rock anymore** : give up and change mode to 'forward'

### Picking Up Mode

This is the last step when we detect a rock.

In 'pickingup' mode, decisions were made base on these situations:
1. **when rover state still in picking_up** : wait
2. **when rover state exits picking_up** : change mode to 'forward'

### Unstuck Mode

In unstuck mode, we brake the rover, then steer the rover clockwise for a pre-determined angle.

In 'unstuck' mode, decisions were made base on these situations:
1. **rover still moving** : brake and wait
2. **rover stationary** : turn rover clockwise
3. **pre-determined yaw reached** : change mode to 'forward'

### Unstuck Anticlockwise Mode

In unstuck anticlockwise mode, we brake the rover, then steer the rover anticlockwise for a pre-determined angle.

In 'unstuckanti' mode, decisions were made base on these situations:
1. **rover still moving** : brake and wait
2. **rover stationary** : turn rover anticlockwise
3. **pre-determined yaw reached** : change mode to 'forward'

### Turn To Home Mode

The 'Go Home' strategy is very simple. There is no route planning:
1. Turn rover towards the home direction
2. let the rover roam freely for the next X seconds where X = (distance from home / 1.7) seconds. So the rover has more time to roam freely when far away from home, and less time when near home.
3. Once X seconds reached, stop rover and repeat step 1, until reached home.

In 'turntohome' mode, decisions were made base on these situations:
1. **rover still moving** : brake and wait
2. **rover stationary** : turn rover toward home location, calculate X seconds for rover to roam freely, change mode to 'forward'

## Simulator Runs

The simulator was run using these parameters:
1. Roversim_x86_64.exe (25 May 2017) on Windows 7
2. Screen resolution: 1024x768
3. Graphics quality: Fastest or Good

I have captured a complete mapping, rock collection, and return to home video.
1. Actual run time: 24 min
2. 98.8% mapped
3. 75.9% fidelity
4. 6 Rocks collected

The run could have been much shorter if the sim didn't seriously lag after 4 min (1 min in the video). The rover was missing turning targets due to the lag, and the unstuck strategies went out of control.

Video playing at 4x speed.

Video URL: https://youtu.be/MiPmyRX1FZU

<a href="http://www.youtube.com/watch?feature=player_embedded&v=MiPmyRX1FZU" target="_blank"><img src="http://img.youtube.com/vi/MiPmyRX1FZU/0.jpg"
alt="Rover Simulation" width="240" height="180" border="1" /></a>


There are other simulation runs captured that illustrate interesting situations, e.g.
1. Rover confused by 2 rocks in the same view
2. Picked up 6 rocks, but the simulator lags so much that the rover keep spinning round and round when trying to target home
3. Rock was embedded too deep into the wall, rover moved keeping to the side of wall but still didn't trigger the "near_sample" signal and moved pass the rock

I may upload these to youtube if I can find time.


### Parameters consideration

The current parameters are designed to enhance the chance of completing a full  run of picking all 6 rocks and map close to 100%. The trade off is that the
rover will hit the walls a lot and get stuck a lot. In some runs, the rover will get stuck so deeply that you'll need to end the simulation.

It is possible to increase the obstacle detection thresholds so that the rover runs much more smoothly, but the tight areas in the map will not get visited, and if a rock is there, it will not be detected.

When the simulator lags, the rover may missing turning targets, and it will turn for several rounds. It may also not detecing rocks when passing by. The unstuck strategies also goes out of control when the camera feeds are missing frames during lag. It is possible to counter this by making the rover moves and turn slower. But doing that will make the simulation runs longer, and the lag gets worst as time goes by. My preference was to move faster and hope for the best.

### Supporting Function enhancements

I added some codes in supporting_functions.py to make the rock sample locations marked out as yellow blinking dots, and also mark the current rover location as a white dot. This makes watching the simulation runs more fun. This doesn't change the functionality and statistics.

```
# plot out all rock sample location in yellow blicking
time_blink = int(Rover.total_time) % 3
rock_size = 1
map_add[int(Rover.pos[1])-rock_size:int(Rover.pos[1])+rock_size,
              int(Rover.pos[0])-rock_size:int(Rover.pos[0])+rock_size, :2] = 255
if (time_blink == 0):
    for idx in range(len(Rover.samples_pos[0]) ):
      test_rock_x = Rover.samples_pos[0][idx]
      test_rock_y = Rover.samples_pos[1][idx]
      map_add[test_rock_y-rock_size:test_rock_y+rock_size,
              test_rock_x-rock_size:test_rock_x+rock_size, :2] = 255
```

Add them before this line:
```
# Flip the map for plotting so that the y-axis points upward in the display
```
