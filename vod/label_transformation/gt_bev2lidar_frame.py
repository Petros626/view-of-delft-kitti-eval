import numpy as np 
import pickle
from math import atan2, pi, sqrt
from vod.label_transformation.utils.utils import normalize_angle, bev_to_pixel_coords, pixel_to_world_coords
from vod.label_transformation.utils.utils import extract_gt_for_lidar_idx, save_transf_lidar_labels

class BEVtoLiDARConverter:
    def __init__(self, image_size=(640, 640), cell_size=0.1):
        self.image_width, self.image_height = image_size
        self.cell_size = cell_size

    def bev_to_lidar_label(self, bev_label, gt_match=None, gt_rotation=None):
        """
        Convert a BEV label to a LiDAR label.
        
        Args:
            bev_label: List containing [class_id, x1, y1, x2, y2, x3, y3, x4, y4, bbox, truncation, occlusion]
            gt_match: ground truth match dictionary (optional)
        Returns:
        lidar_label: Dictionary containing the LiDAR label fields    
        """
        class_id = int(bev_label[0])
        x1, y1, x2, y2, x3, y3, x4, y4 = bev_label[1:9]
        #bbox = bev_label[9:13] # use original values
        truncation = bev_label[13]
        occlusion = bev_label[14]

        # Reorder points from YOLO to standard clockwise
        """
        YOLO-Format                     Default
       x4,y4 --- x1,y1                x2,y2 --- x3,y3
        |           |                  |           |
        |           |       →          |           |
        |           |                  |           |
       x3,y3 --- x2,y2                x1,y1 --- x4,y4
        """
        x1, x2, x3, x4 = x3, x4, x1, x2
        y1, y2, y3, y4 = y3, y4, y1, y2

        # Convert from normalized pixel coords to pixel coords
        pixel_coords = bev_to_pixel_coords(
            [(x1, y1), (x2, y2), (x3, y3), (x4, y4)],
            self.image_width,
            self.image_height
        )

        # Convert from pixel coords to world coords
        world_coords = pixel_to_world_coords(
            pixel_coords,
            self.image_width,
            self.image_height,
            self.cell_size
        )

        # Calculate center, dimensions, and rotation from BEV data
        center =  np.mean(world_coords, axis=0)
        edge = world_coords[1] - world_coords[0]
        length = np.linalg.norm(edge)
        width = np.linalg.norm(world_coords[2] - world_coords[1])

        # Calculate the rotation directly from the world coordinates of the bounding box
        heading_lidar = atan2(edge[1], edge[0]) # y, x, directly provides the correct angle in the LiDAR CW system
        heading_lidar = normalize_angle(heading_lidar, gt_rotation) # [-pi...pi]
        #print(f"(Calc.) x,y: {center[0], center[1]}")
        #print(f"(Calc.) rot_z: {heading_lidar}")
        
        # Don't subtract the class-specific offsets applied before training, it would distort the metrics
        #if class_id == 1:  # Car
        #    length = max(0, length - 0.4)
        #    width = max(0, width - 0.4)
        #elif class_id in [2, 3]:  # Pedestrian/Cyclist
        #    length = max(0, length - 0.3)
        #    width = max(0, width - 0.3)
        #print(f"(Calc.) w,l: {width, length}")

        # Extract alpha, score, height, z from gt_match in .pkl file
        alpha = gt_match.get('alpha', -10) if gt_match else -10
        score = gt_match.get('score', 0.0) if gt_match else 0.0
        height = gt_match['3Dbox'][5] if gt_match else -1
        z = gt_match['3Dbox'][2] if gt_match else -1000
  
        # Map class_id to type
        class_map = {1: "Car", 2: "Pedestrian", 3: "Cyclist"}
        obj_type = class_map.get(class_id, "DontCare")

        #print(f"(.pkl) h, z: {height, z}")

        # Create LiDAR label
        lidar_label = {
            "type": str(obj_type),
            "truncated": float(truncation),
            "occluded": int(occlusion),
            "alpha": alpha, # from gt data
            #"bbox": bbox,  # from gt data
            "bbox": [-1, -1, -1, -1], # dummy values, calc. later
            "dimensions": [height, width, length], #  h, w, l
            "location": [center[0], center[1], z], # x, y, z
            "rotation_z": heading_lidar,
            "score": float(score) # from gt data
        }

        return lidar_label
    
    def match_bev_to_gt(self, bev_label, gt_info):
        """
        Match a BEV label to the closest ground truth box without creating a full LiDAR label.
        
        Args:
            bev_label: List containing BEV label information
            gt_info: List of ground truth boxes
            
        Returns:
            best_match: Closest ground truth box dictionary or None
        """
        class_id = int(bev_label[0])
        class_map = {1: 'Car', 2: 'Pedestrian', 3: 'Cyclist'}
        
        # Extract BEV coordinates without creating a full LiDAR label
        x1, y1, x2, y2, x3, y3, x4, y4 = bev_label[1:9]
        
        # Reorder points from YOLO to standard clockwise
        x1, x2, x3, x4 = x3, x4, x1, x2
        y1, y2, y3, y4 = y3, y4, y1, y2

        # Convert to pixel coordinates
        pixel_coords = bev_to_pixel_coords(
            [(x1, y1), (x2, y2), (x3, y3), (x4, y4)],
            self.image_width,
            self.image_height
        )

        # Convert to world coordinates
        world_coords = pixel_to_world_coords(
            pixel_coords,
            self.image_width,
            self.image_height,
            self.cell_size
        )

        # Calculate center and dimensions without storing them as a label
        center = np.mean(world_coords, axis=0)
        edge = world_coords[1] - world_coords[0]
        length = np.linalg.norm(edge)
        width = np.linalg.norm(world_coords[2] - world_coords[1])
        
        # Calculate rotation angle
        heading = atan2(edge[1], edge[0])
        
        best_match = None
        min_score = float('inf')

        for gt in gt_info:
            # Class filter
            if gt['name'] != class_map.get(class_id, 'DontCare'):
                continue

            gt_box3d = gt['3Dbox']

            # Position difference (Euclidean distance)
            pos_diff = sqrt((center[0] - gt_box3d[0])**2 + 
                            (center[1] - gt_box3d[1])**2)

            # Dimension difference
            dim_diff = min(
                abs(length - gt_box3d[3]) + abs(width - gt_box3d[4]),
                abs(length - gt_box3d[4]) + abs(width - gt_box3d[3])
            )

            # Rotation difference
            rot_diff = abs(heading - gt_box3d[6])
            rot_diff = min(rot_diff, 2 * pi - rot_diff)

            # Combined score (weighted)
            score = 0.6 * pos_diff + 0.2 * dim_diff + 0.2 * rot_diff

            if score < min_score:
                min_score = score
                best_match = gt
        
        return best_match if min_score < 2.0 else None
    
if __name__ == "__main__":
    import os

    single_file_mode = False

    if single_file_mode:
        # Path to file with bev labels
        bev_label_file = "kitti_gt_annos_2/all_bev_gt_annos_2/bev_val_000076.txt"

        # Check if the file exists
        if not os.path.exists(bev_label_file):
            print(f"BEV label file '{bev_label_file}' not found.")
            exit(1)

        # Load BEV labels from the file
        bev_labels = []
        with open(bev_label_file, "r") as f:
            for line in f:
                bev_labels.append([float(x) for x in line.strip().split()])

        # Load ground truth data
        with open("validation_pickle/kitti_val_dataset.pkl", "rb") as f:
            gt_data = pickle.load(f)

        # Initialise converter
        converter = BEVtoLiDARConverter()

        # Convert BEV labels into LiDAR labels and output them
        for i, bev_label in enumerate(bev_labels):
            lidar_idx = bev_label_file.split('_')[-1].split('.')[0]
            gt_boxes = extract_gt_for_lidar_idx(gt_data, lidar_idx)

            gt_match = converter.match_bev_to_gt(bev_label, gt_boxes)
            lidar_label = converter.bev_to_lidar_label(bev_label, gt_match, gt_match['3Dbox'][6])

            print(f"LiDAR label {i + 1}:")
            for key, value in lidar_label.items():
                print(f"  {key}: {value}")
            print("\n")

            #save_transf_lidar_labels("kitti_gt_annos/gt_bev_to_lidar_labels", lidar_idx, [lidar_label])
    else:
        import glob
        from progress.bar import IncrementalBar
    
        bev_label_dir = "kitti_gt_annos_2/all_bev_gt_annos_2"
        output_dir = "kitti_gt_annos_2/gt_bev_to_lidar_labels_2"

        if not os.path.exists(bev_label_dir):
            print(f"Directory '{bev_label_dir}' not found.")
            exit(1)
        
        with open("validation_pickle/kitti_val_dataset.pkl", "rb") as f:
            gt_data_pkl = pickle.load(f)

        converter = BEVtoLiDARConverter()

        bev_label_files = glob.glob(os.path.join(bev_label_dir, "*.txt"))

        bar = IncrementalBar('Processing', max=len(bev_label_files), 
                             suffix='%(percent).1f%% - Estimated time: %(eta)ds')
    
        for bev_label_file in bev_label_files:
            bev_labels = []
            with open(bev_label_file, "r") as f:
                for line in f:
                    bev_labels.append([float(x) for x in line.strip().split()])

            lidar_idx = os.path.basename(bev_label_file).split('_')[-1].split('.')[0]
            gt_info = extract_gt_for_lidar_idx(gt_data_pkl, lidar_idx)

            lidar_labels = []
            for bev_label in bev_labels:
                gt_match = converter.match_bev_to_gt(bev_label, gt_info)
                lidar_label = converter.bev_to_lidar_label(bev_label, gt_match, gt_match['3Dbox'][6])
                lidar_labels.append(lidar_label)

            #save_transf_lidar_labels(output_dir, lidar_idx, lidar_labels)

            bar.next()

bar.finish()
                                                      