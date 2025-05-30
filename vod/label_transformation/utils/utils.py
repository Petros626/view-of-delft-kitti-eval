import numpy as np
import os
import math
import copy

def normalize(num, lower=0, upper=360, b=False):
    # source: # https://gist.github.com/phn/1111712/35e8883de01916f64f7f97da9434622000ac0390
    """Normalize number to range [lower, upper) or [lower, upper].

    Parameters
    ----------
    num : float
        The number to be normalized.
    lower : int
        Lower limit of range. Default is 0.
    upper : int
        Upper limit of range. Default is 360.
    b : bool
        Type of normalization. Default is False. See notes.

    Returns
    -------
    n : float
        A number in the range [lower, upper) or [lower, upper].

    Raises
    ------
    ValueError
    If lower >= upper.

    Notes
    -----
    If the keyword `b == False`, then the normalization is done in the
    following way. Consider the numbers to be arranged in a circle,
    with the lower and upper ends sitting on top of each other. Moving
    past one limit, takes the number into the beginning of the other
    end. For example, if range is [0 - 360), then 361 becomes 1 and 360
    becomes 0. Negative numbers move from higher to lower numbers. So,
    -1 normalized to [0 - 360) becomes 359.

    If the keyword `b == True`, then the given number is considered to
    "bounce" between the two limits. So, -91 normalized to [-90, 90],
    becomes -89, instead of 89. In this case the range is [lower,
    upper]. This code is based on the function `fmt_delta` of `TPM`.

    Range must be symmetric about 0 or lower == 0.

    Examples
    --------
    >>> normalize(-270,-180,180)
    90.0
    >>> import math
    >>> math.degrees(normalize(-2*math.pi,-math.pi,math.pi))
    0.0
    >>> normalize(-180, -180, 180)
    -180.0
    >>> normalize(180, -180, 180)
    -180.0
    >>> normalize(180, -180, 180, b=True)
    180.0
    >>> normalize(181,-180,180)
    -179.0
    >>> normalize(-180,0,360)
    180.0
    >>> normalize(36,0,24)
    12.0
    >>> normalize(368.5,-180,180)
    8.5
    >>> normalize(-100, -90, 90, b=True)
    -80.0
    >>> normalize(100, -90, 90, b=True)
    80.0
    >>> normalize(181, -90, 90, b=True)
    -1.0
    >>> normalize(270, -90, 90, b=True)
    -90.0
    >>> normalize(271, -90, 90, b=True)
    -89.0
    """
    from math import floor, ceil
    # abs(num + upper) and abs(num - lower) are needed, instead of
    # abs(num), since the lower and upper limits need not be 0. We need
    # to add half size of the range, so that the final result is lower +
    # <value> or upper - <value>, respectively.
    res = num
    if not b:
        if lower >= upper:
            raise ValueError("Invalid lower and upper limits: (%s, %s)" %
                            (lower, upper))

        res = num
        if num > upper or num == lower:
            num = lower + abs(num + upper) % (abs(lower) + abs(upper))
        if num < lower or num == upper:
            num = upper - abs(num - lower) % (abs(lower) + abs(upper))

        res = lower if res == upper else num
    else:
        total_length = abs(lower) + abs(upper)
        if num < -total_length:
            num += ceil(num / (-2 * total_length)) * 2 * total_length
        if num > total_length:
            num -= floor(num / (2 * total_length)) * 2 * total_length
        if num > upper:
            num = total_length - num
        if num < lower:
            num = -total_length - num

        res = num

    res = num * 1.0  # Make all numbers float, to be consistent

    return res


def normalize_angle(angle, gt_angle=None):
    """Normalize angle to [-pi, pi] range."""
    angle = (angle + math.pi) % (2 * math.pi) - math.pi
    if gt_angle is not None:
        diff = abs(angle - gt_angle)
        if diff > math.pi:
            angle += -2 * math.pi if angle > 0 else 2 * math.pi
    return angle


def normalize_angle_pred(angle):
    """Normalize angle to [0, pi/2] range."""
    while angle > math.pi/2:
        angle -= math.pi/2
    while angle < 0:
        angle += math.pi/2
    return angle


def bev_to_pixel_coords(norm_coords, image_width, image_height):
    """Convert normalized BEV coordinates to pixel coordinates."""
    points = []
    for x_norm, y_norm in norm_coords:
        px = x_norm * image_width
        py = y_norm * image_height
        points.append([px, py])
    return np.array(points)


def pixel_to_world_coords(pixel_coords, image_width, image_height, cell_size):
    """Convert pixel coordinates to world coordinates (source was LiDAR frame)."""
    world_points = []
    for px, py in pixel_coords:
        y = -(px - image_width / 2) * cell_size # image x -> lidar y
        x = (image_height - py) * cell_size # image y -> lidar x
        world_points.append([x, y])
    return np.array(world_points)


def pixel_to_world_coords_pred(center_px, dim_px, image_width, image_height, cell_size):
    """Convert YOLO prediction format from pixel to world coordinates."""
    y = -(center_px[0] - image_width / 2) * cell_size # image x -> lidar y
    x = (image_height - center_px[1]) * cell_size # image y -> lidar x
    
    length_meters = dim_px[0] * cell_size  # width in YOLO = length in KITTI
    width_meters = dim_px[1] * cell_size   # height in YOLO = width in KITTI
    
    return (x, y), (length_meters, width_meters)


def cart_to_hom(pts):
        """Convert Cartesian to homogeneious coordinates
        Args:
            pts: (N, 3 or 2)
        Returns:
            pts_hom: (N, 4 or 3)
        """
        pts_hom = np.hstack((pts, np.ones((pts.shape[0], 1), dtype=np.float32)))
        return pts_hom


def extract_gt_for_lidar_idx(gt_data, lidar_idx):
    """Extract GT boxes with original indices and names etc."""
    for entry in gt_data:
        if entry['point_cloud']['lidar_idx'] == lidar_idx:
            return [
                {
                    'name': entry['annos']['name'][i],
                    '3Dbox': entry['annos']['gt_boxes_lidar'][i][:7],
                    'alpha': entry['annos']['alpha'][i],
                    'score': entry['annos']['score'][i],
                    'original_index': i
                }
                for i in range(len(entry['annos']['name']))
            ]
    return None


def save_transf_lidar_labels(output_dir, lidar_idx, lidar_labels):
    """
    Save transformed LiDAR labels to a .txt file.

    Args:
        output_dir (str): Directory to save the labels
        lidar_ifx (str): LiDAR frame index
        bev_labels (list): List of transformed LiDAR labels
    """
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{lidar_idx}.txt")

    with open(output_file, "w") as f:
        for label in lidar_labels:
            line = f"{label['type']} {label['truncated']} {label['occluded']} {label['alpha']} " \
                   f"{label['bbox'][0]} {label['bbox'][1]} {label['bbox'][2]} {label['bbox'][3]} " \
                   f"{label['dimensions'][0]} {label['dimensions'][1]} {label['dimensions'][2]} " \
                   f"{label['location'][0]} {label['location'][1]} {label['location'][2]} {label['rotation_z']} {label['score']}\n"
            f.write(line)


def save_transf_camera_labels(output_dir, lidar_idx, camera_labels):
    """
    Save transformed camera labels to a .txt file in KITTI format.

    Args:
        output_dir (str): Directory to save the labels
        lidar_idx (str): LiDAR frame index
        camera_labels (list): List of transformed camera labels
    """
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{lidar_idx}.txt")

    with open(output_file, "w") as f:
        for label in camera_labels:
            line = f"{label['type']} {label['truncated']} {label['occluded']} {label['alpha']} " \
                   f"{label['bbox'][0]} {label['bbox'][1]} {label['bbox'][2]} {label['bbox'][3]} " \
                   f"{label['dimensions'][0]:.2f} {label['dimensions'][1]:.2f} {label['dimensions'][2]:.2f} " \
                   f"{label['location'][0]:.2f} {label['location'][1]:.2f} {label['location'][2]:.2f} {label['rotation_y']:.2f} {label['score']}\n"
            f.write(line)


def boxes3d_to_corners3d_kitti_camera(boxes3d, bottom_center=True):
    """
    :param boxes3d: (N, 7) [x, y, z, h, w, l, ry] in camera coords, see the definition of ry in KITTI dataset
    :param bottom_center: whether y is on the bottom center of object
    :return: corners3d: (N, 8, 3)
        7 -------- 4
       /|         /|
      6 -------- 5 .
      | |        | |
      . 3 -------- 0
      |/         |/
      2 -------- 1
    """
    boxes_num = boxes3d.shape[0]
    h, w, l = boxes3d[:, 3], boxes3d[:, 4], boxes3d[:, 5]
    x_corners = np.array([l / 2., l / 2., -l / 2., -l / 2., l / 2., l / 2., -l / 2., -l / 2], dtype=np.float32).T
    z_corners = np.array([w / 2., -w / 2., -w / 2., w / 2., w / 2., -w / 2., -w / 2., w / 2.], dtype=np.float32).T
    if bottom_center:
        y_corners = np.zeros((boxes_num, 8), dtype=np.float32)
        y_corners[:, 4:8] = -h.reshape(boxes_num, 1).repeat(4, axis=1)  # (N, 8)
    else:
        y_corners = np.array([h / 2., h / 2., h / 2., h / 2., -h / 2., -h / 2., -h / 2., -h / 2.], dtype=np.float32).T

    ry = boxes3d[:, 6]
    zeros, ones = np.zeros(ry.size, dtype=np.float32), np.ones(ry.size, dtype=np.float32)
    rot_list = np.array([[np.cos(ry), zeros, -np.sin(ry)],
                         [zeros, ones, zeros],
                         [np.sin(ry), zeros, np.cos(ry)]])  # (3, 3, N)
    R_list = np.transpose(rot_list, (2, 0, 1))  # (N, 3, 3)

    temp_corners = np.concatenate((x_corners.reshape(-1, 8, 1), y_corners.reshape(-1, 8, 1),
                                   z_corners.reshape(-1, 8, 1)), axis=2)  # (N, 8, 3)
    rotated_corners = np.matmul(temp_corners, R_list)  # (N, 8, 3)
    x_corners, y_corners, z_corners = rotated_corners[:, :, 0], rotated_corners[:, :, 1], rotated_corners[:, :, 2]

    x_loc, y_loc, z_loc = boxes3d[:, 0], boxes3d[:, 1], boxes3d[:, 2]

    x = x_loc.reshape(-1, 1) + x_corners.reshape(-1, 8)
    y = y_loc.reshape(-1, 1) + y_corners.reshape(-1, 8)
    z = z_loc.reshape(-1, 1) + z_corners.reshape(-1, 8)

    corners = np.concatenate((x.reshape(-1, 8, 1), y.reshape(-1, 8, 1), z.reshape(-1, 8, 1)), axis=2)

    return corners.astype(np.float32)


def obtain_z_from_bev_pixel_value(pixel_value, OFFSET_LIDAR=2.73, Z_MIN_HEIGHT=-2.73, Z_MAX_HEIGHT=1.27, OUT_MIN=0, OUT_MAX=255):
    """Converts a pixel value back to z-coordinate in the LiDAR coordinate system.
    Args:
        pixel_value (float): Pixel value to convert
        OFFSET_LIDAR (float): LiDAR offset in meters 
        Z_MIN_HEIGHT (float): Minimum height value 
        Z_MAX_HEIGHT (float): Maximum height value 
        OUT_MIN (float): Minimum output range 
        OUT_MAX (float): Maximum output range 
    Returns:
        float: z-coordinate in meters (LiDAR coordinate system)
    """
    # Rückwärts-Skalierung vom Ausgabebereich [OUT_MIN, OUT_MAX] zu [0, 1]
    normalized_value = (pixel_value - OUT_MIN) / (OUT_MAX - OUT_MIN)
    
    # Rückwärts-Normalisierung von [0, 1] zum Höhenbereich mit Offset
    height_with_offset = normalized_value * (Z_MAX_HEIGHT - Z_MIN_HEIGHT) + Z_MIN_HEIGHT
    
    # Entfernung des LiDAR-Offsets -> z-Koordinate im LiDAR-System
    z_coordinate = height_with_offset - OFFSET_LIDAR

    object_height = z_coordinate + abs(Z_MIN_HEIGHT + OFFSET_LIDAR)
    
    return object_height
