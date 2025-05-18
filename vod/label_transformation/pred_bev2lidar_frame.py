from math import pi
from pathlib import Path
from vod.label_transformation.utils.utils import normalize_angle_pred, pixel_to_world_coords_pred, save_transf_lidar_labels

class BEVPredtoLiDARConverter:
    def __init__(self, image_size=(640, 640), cell_size=0.1):
        self.image_width, self.image_height = image_size
        self.cell_size = cell_size

    def parse_bev_prediction(self, pred_line):
        """Parse a line from YOLO BEV prediction file"""
        parts = pred_line.strip().split()
        return {
            'class_id': int(float(parts[0])),
            'x_center': float(parts[1]),
            'y_center': float(parts[2]),
            'width': float(parts[3]), # this is length in KITTI
            'height': float(parts[4]), # this is width in KITTI
            'rotation': float(parts[5]), # in radians
            'confidence': float(parts[6]),
        }

    def convert_to_lidar_label(self, pred, regularized=False):
        """Convert YOLO BEV prediction to LiDAR label"""
        # Convert center and dimensions to world coordinates
        center, dimensions = pixel_to_world_coords_pred(
            (pred['x_center'], pred['y_center']),
            (pred['width'], pred['height']),  # width=length, height=width in KITTI
            self.image_width,
            self.image_height,
            self.cell_size
        )
 
        # Ensure that rotation is in range. Since the zero angle in the BEV is 
        # along the image x-axis (right) and in the LiDAR along the x-axis (front), 
        # you must shift the angle by +90° (i.e. +π/2) to align the reference axes.
        #r = normalize_angle_pred(pred['rotation']) # depends on how the angle comes, regularized or raw
        #r = r + pi/2 # do i really need this ???
        
        # Case 1: rotation is regularized CW, prediction angle is [0°...90°]
        if regularized:
            heading = pred['rotation'] - pi/2  # BEV-X → LiDAR-X
            #heading = pred['rotation'] # unsure about it
        # Case 2: rotation is raw CW, the biggest values were [-28°...122°], so it's from [-pi/4...3pi/4] range
        else:
            heading = normalize_angle_pred(pred['rotation']) # [-π/4, 3π/4] → [0, π/2]
            heading = heading - pi/2 # Same axis adjustment like above
            #heading = pred['rotation'] # unsure about it
        
        # Default height values based on class
        default_heights = {1: 1.53, 2: 1.76, 3: 1.74}  # Car: 1.53m, Ped: 1.76m, Cyc: 1.74m
        height = default_heights.get(pred['class_id'])

        # Map class_id to type
        class_map = {1: "Car", 2: "Pedestrian", 3: "Cyclist"}
        obj_type = class_map.get(pred['class_id'], "DontCare")

        # Create LiDAR label in default KITTI format
        lidar_label = {
            "type": str(obj_type),
            "truncated": float(0.0), # float
            "occluded": int(0), # int
            "alpha": float(0.0), # float
            "bbox_pre_height": [0.0], # float
            "dimensions": [height, dimensions[1], dimensions[0]],  # h, w, l
            "location": [center[0], center[1], 0.0], # x, y, z
            "rotation_z": (heading), # rad
            "score": float(pred['confidence'])
        }

        return lidar_label
    
if __name__ == "__main__":
    import os 
    import glob
    from progress.bar import IncrementalBar

    single_file_mode = False
    
    if single_file_mode:
        pred_file = "predictions/all_bev_preds/val_trt_fp32/labels/bev_val_000001.txt"
        output_dir = "predictions/pred_bev_to_lidar_fp32"

        if not os.path.exists(pred_file):
            print(f"Prediction file '{pred_file}' not found.")
            exit(1)

        converter = BEVPredtoLiDARConverter()

        # Process single file
        lidar_labels = []
        with open(pred_file, 'r') as f:
            for line in f:
                pred = converter.parse_bev_prediction(line)
                lidar_label = converter.convert_to_lidar_label(pred)
                lidar_labels.append(lidar_label)

        lidar_idx = Path(pred_file).stem.split('_')[-1]
        save_transf_lidar_labels(output_dir, lidar_idx, lidar_labels)

    else:
        pred_dir = "predictions/all_bev_preds_minAreaRect()/val_trt_fp32/labels"
        output_dir = "predictions/pred_bev_to_lidar_fp32"

        if not os.path.exists(pred_dir):
            print(f"Directory '{pred_dir}' not found.")
            exit(1)

        converter = BEVPredtoLiDARConverter()
        pred_files = glob.glob(os.path.join(pred_dir, "*.txt"))

        bar = IncrementalBar('Processing', max=len(pred_files), 
                            suffix='%(percent).1f%% - Estimated time: %(eta)ds')
        
        for pred_file in pred_files:
            lidar_labels = []
            with open(pred_file, 'r') as f:
                for line in f:
                    pred = converter.parse_bev_prediction(line)
                    lidar_label = converter.convert_to_lidar_label(pred, regularized=False)
                    lidar_labels.append(lidar_label)

            lidar_idx = Path(pred_file).stem.split('_')[-1]
            #save_transf_lidar_labels(output_dir, lidar_idx, lidar_labels)
            bar.next()

        bar.finish()


# Reverse engineering the height information
# height = (pixel_value * (Z_MAX_HEIGHT - Z_MIN_HEIGHT) / 255.0) + Z_MIN_HEIGHT - OFFSET_LIDAR
# height = (pixel_value * (1.27 - (-2.73)) / 255.0) + (-2.73) + 2.73