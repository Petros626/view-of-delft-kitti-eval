import numpy as np
from math import pi
import os, copy



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
    angle = (angle + pi) % (2 * pi) - pi
    if gt_angle is not None:
        diff = abs(angle - gt_angle)
        if diff > pi:
            angle += -2*pi if angle > 0 else 2*pi
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
        y = -(px - image_width / 2) * cell_size
        x = (image_height - py) * cell_size
        world_points.append([x, y])
    return np.array(world_points)


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
                    'box': entry['annos']['gt_boxes_lidar'][i][:7],
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
                   f"{label['bbox_pre_height'][0]} {label['dimensions'][0]} {label['dimensions'][1]} {label['dimensions'][2]} " \
                   f"{label['location'][0]} {label['location'][1]} {label['location'][2]} {label['rotation_z']} {label['score']}\n"
            f.write(line)
    #print(f"Transformed LiDAR labels saved to {output_file}")


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
                   f"{label['bbox_pre_height']} {label['dimensions'][0]:.2f} {label['dimensions'][1]:.2f} {label['dimensions'][2]:.2f} " \
                   f"{label['location'][0]:.2f} {label['location'][1]:.2f} {label['location'][2]:.2f} {label['rotation_y']:.2f} {label['score']}\n"
            f.write(line)
    #print(f"Transformed Camera labels saved to {output_file}")
