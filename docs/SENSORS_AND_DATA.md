# Sensors and Data

<div align="center">
<img src="figures/Prius_sensor_setup_5.png" alt="Prius sensor setup" width="500"/>
</div>

<br>

---

## Camera
The camera provides colored, rectified images of 1936 × 1216 pixels at around 30 Hz.   
The horizontal field of view is ~64° (± 32°), vertical field of view is ~ 44° (± 22°). 

### Data format
Images are stored in jpg files.
<br>
<br>

---
## LiDAR
The LiDAR sensor is a Velodyne 64 sensor mounted on the top of the vehicle, operating at 10 Hz.  
The provided LiDAR point clouds are ego-motion compensated both for ego-motion during the scan (i.e. one full rotation of the LiDAR sensor) and ego-motion between the capture of LiDAR and camera data (i.e. overlaying camera and LiDAR data should give a consistent image).

### Data format
LiDAR point clouds are stored in bin files.  
Each bin file contains a 360° scan in a form of a Nx4 array, where N is the number of points, and 4 is the number of features:
`[x,y,z,reflectance]`
<br>
<br>

---
## Radar
The radar sensor is a ZF FRGen21 3+1D radar (∼13 Hz) mounted behind the front bumper.  
The provided radar point clouds are ego-motion compensated for ego-motion between the capture of radar and camera data (i.e. overlaying camera and radar data should give a consistent image).
We provide radar point clouds in three flavors:
- Single scan
- 3 scans (accumulation of the last 3 radar scans)
- 5 scans (accumulation of the last 5 radar scans)  

Accumulation (i.e. radar_3_scans and radar_5_scans folders) is implemented by transforming point clouds from previous scans to the coordinate system of the last scan.

### Data format
The radar point clouds are stored in bin files.  
Each bin file contains a set of points in the form of a Nx7 array, where N is the number of points, and 7 is the number of features:  

`[x, y, z, RCS, v_r, v_r_compensated, time]`

where `v_r` is the relative radial velocity, and `v_r_compensated` is the absolute (i.e. ego motion compensated) radial velocity of the point.

`time` is the time id of the point, indicating which scan it originates from. E.g., a point from the current scan has a t = 0,
while a point from the third most recent scan has a t = −2. 

<br>
<br>

---
## Odometry
Odometry is a filtered combination of several inputs: RTK GPS, IMU, and wheel odometry with a frame rate around 30 Hz. 

### Data format
We provide odometry information as transformations. For convenience, three transformations is defined for each frame:
- map to camera  (global coordinate system)
- odom to camera (local coordinate system)
- UTM to camera  (Official [UTM](https://en.wikipedia.org/wiki/Universal_Transverse_Mercator_coordinate_system) coordinate system)

<br>
<br>

---
## Calibration files
We provide extrinsic calibration between the point cloud sensors (LiDAR, radar) and the camera in KITTI format.
Further transformations, e.g. LiDAR to radar, or UTM to LiDAR can be derived with our devkit through the transformations described in the calibration files and the odometry data as shown in the examples.
<br>
<br>

---
## Syncronization
Output of the sensors were recorded in an asyncronus way (i.e. no connected triggering) with accurate, synced timestamps.  
For convenience, we provide the dataset in synchronized “frames” similar to the  KITTI dataset, consisting of a: 
- a rectified mono-camera image, 
- a LiDAR point cloud,
- three radar point clouds (single-, 3, and 5 scans), 
- and a pose file describing the position of the egovehicle. 
 
Timestamps of the LiDAR sensor were chosen as lead (~10 Hz), and we chose the closest camera, radar and odometry information available (maximum tolerated time difference is set to **0.04 seconds**).
To get the best possible syncronization, we synced radar and camera data to the moment when the LiDAR sensor **scanned the middle of the camera field of view**.

Corressponding  camera, radar, LiDAR, and pose messages (i.e. content of a frame) are connected via their filenames, see the [GETTING_STARTED](GETTING_STARTED.md) manual and the [EXAMPLES](EXAMPLES.md) manual.

We also share the metadata of the syncronized messages, i.e. the original timestamp of each syncronized message in the frame.


## Scenes

Frames are consequitve withing scenes or clips. The KITTI format does not allow for scene level information, so please find the clips defined below.

Number | Clip | Frames | Index Start Frame | Index End Frame | Split
-- | -- | -- | -- | -- | --
1 | delft_1 | 544 | 0 | 543 | Valid
2 | delft_2 | 768 | 544 | 1311 | Train
3 | delft_3 | 491 | 1312 | 1802 | Train
4 | delft_4 | 397 | 1803 | 2199 | Train
6 | delft_6 | 332 | 2200 | 2531 | Train
7 | delft_7 | 266 | 2532 | 2797 | Test
8 | delft_8 | 479 | 2798 | 3276 | Test
9 | delft_9 | 298 | 3277 | 3574 | Train
10 | delft_10 | 35 | 3575 | 3609 | Valid
11 | delft_11 | 438 | 3610 | 4047 | Train
12 | delft_12 | 338 | 4049 | 4386 | Train
13 | delft_13 | 265 | 4387 | 4651 | Train
14 | delft_14 | 434 | 4652 | 5085 | Valid
16 | delft_16 | 237 | 6334 | 6570 | Test
18 | delft_18 | 188 | 6571 | 6758 | Test
19 | delft_19 | 784 | 6759 | 7542 | Train
20 | delft_20 | 357 | 7543 | 7899 | Test
21 | delft_21 | 298 | 7900 | 8197 | Test
22 | delft_22 | 283 | 8198 | 8480 | Valid
23 | delft_23 | 268 | 8481 | 8748 | Train
24 | delft_24 | 347 | 8749 | 9095 | Train
25 | delft_25 | 422 | 9096 | 9517 | Test
26 | delft_26 | 258 | 9518 | 9775 | Train
27 | delft_27 | 155 | 9776 | 9930 | Train

