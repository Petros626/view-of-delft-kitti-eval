import math
from pathlib import Path
from vod.label_transformation.utils.utils import normalize_angle_minuspi4_3pi4, pixel_to_world_coords_pred, save_transf_lidar_labels

class BEVPredtoLiDARConverter:
    def __init__(self, image_size=(640, 640), cell_size=0.1):
        self.image_width, self.image_height = image_size
        self.cell_size = cell_size

    def parse_bev_prediction(self, pred_line):
        """
        Parse a line from YOLO BEV prediction file
        
        Args:
            pred_line (str): Single line from prediction file containing space-separated values
                            Format: class_id cx cy width height rotation confidence

        Returns:
            dict: Parsed prediction data with keys
        """
        parts = pred_line.strip().split()

        # Prediction format (cx, cy), (w, h), angle, conf
        return {
            'class_id': int(float(parts[0])),
            'x_center': float(parts[1]),
            'y_center': float(parts[2]),
            'width': float(parts[3]), # OpenCV format width = KITTI format length
            'height': float(parts[4]), # OpenCV format height = KITTI format width
            'rotation': float(parts[5]), # in radians
            'confidence': float(parts[6]),
        }

    def convert_to_lidar_label(self, pred, regularized=False):
        """
        Convert YOLO BEV prediction to LiDAR label
        
        Args:
            pred (dict): Parsed BEV prediction from parse_bev_prediction()
            regularized (bool): Whether rotation angles are regularized to [0, π/2] range.
                           If False, expects raw rotation range [-π/4, 3π/4]. Default is False
                           
        Returns:
            dict: LiDAR label in KITTI format with keys
        """
        
        # Convert center and dimensions to world coordinates
        center, dimensions = pixel_to_world_coords_pred(
            (pred['x_center'], pred['y_center']),
            (pred['width'], pred['height']),
            self.image_width,
            self.image_height,
            self.cell_size
        )
        
        # Case 1: rotation is regularized CW, prediction angle is [0°...90°]
        if regularized:
            heading = -pred['rotation'] - math.pi/2
            
        # Case 2: rotation is raw CW, the biggest values were [-28°...122°], so it's from [-pi/4...3pi/4] range
        else:             
            # BEV-space to LiDAR-space
            # -: Reflects the x-axis (due to y-axis difference)
            # -pi/2: Rotates coordinate system by -90°
            # NOTE: Both spaces are CW, but the axis orientation is different. 
            # The transformation heading = -pred['rotation'] - math.pi/2 corrects 
            # both the axis mirroring and the reference rotation.
            heading = -pred['rotation'] - math.pi/2
            
        
        # Default height values based on class
        # NOTE: source: "BirdNet+: End-to-End 3D Object Detection in LiDAR Bird's Eye View"
        default_heights = {1: 1.53, 2: 1.76, 3: 1.74}  # Car: 1.53m, Ped: 1.76m, Cyc: 1.74m
        height = default_heights.get(pred['class_id'])
        # 3D Bounding Box Regression not supported by YOLOv8 OBB Head 
        # Approaches like BirdNet, BirdNet+ etc. can't be applied yet
        z = -0.5 # approx. value fitting for 3d boxes vs 3d gt

        # Map class_id to type
        class_map = {1: "Car", 2: "Pedestrian", 3: "Cyclist"}
        obj_type = class_map.get(pred['class_id'], "DontCare")

        # Create LiDAR label in default KITTI format
        lidar_label = {
            "type": str(obj_type),
            "truncated": float(-1), # dummy value like BirdNet2 or OpenPCDet
            "occluded": int(-1), # dummy value
            "alpha": float(-10), # dummy value
            "bbox": [-1, -1, -1, -1], # dummy value, calc. later
            "dimensions": [height, dimensions[1], dimensions[0]],  # h, w, l
            "location": [center[0], center[1], z], # x, y, z
            "rotation_z": heading, # rad
            "score": float(pred['confidence'])
        }

        return lidar_label
    
if __name__ == "__main__":
    import os 
    import glob
    from progress.bar import IncrementalBar

    single_file_mode = False
    
    if single_file_mode:
       
        # [pi/4...3pi/4]
        pred_file = "predictions/all_bev_preds_minAreaRect()/val_trt_fp32/labels/bev_val_000001.txt"
        # [0...pi/2]
        #pred_file = "predictions/all_bev_preds_regularized/val_trt_fp32_rgd/labels/bev_val_000004.txt"

        output_dir = "predictions/all_bev_preds_minAreaRect()/pred_bev_to_lidar_fp32"

        if not os.path.exists(pred_file):
            print(f"Prediction file '{pred_file}' not found.")
            exit(1)

        converter = BEVPredtoLiDARConverter()

        # Process single file
        lidar_labels = []
        with open(pred_file, 'r') as f:
            for line in f:
                pred = converter.parse_bev_prediction(line)
                lidar_label = converter.convert_to_lidar_label(pred, regularized=False)
                lidar_labels.append(lidar_label)

        lidar_idx = Path(pred_file).stem.split('_')[-1]
        
        save_transf_lidar_labels(output_dir, lidar_idx, lidar_labels)

    else:
        # Model: FP32, Yaw range: [-pi/4...3pi/4]
        pred_dir = "predictions/all_bev_preds_minAreaRect()/val_trt_fp32/labels"
        # Model: FP16, Yaw range: [-pi/4...3pi/4] 
        #pred_dir = "predictions/all_bev_preds_minAreaRect()/val_trt_fp16/labels"
        # Model: INT8, Yaw range: [-pi/4...3pi/4]
        #pred_dir = "predictions/all_bev_preds_minAreaRect()/val_trt_int8/labels"

        output_dir = "predictions/all_bev_preds_minAreaRect()/pred_bev_to_lidar_fp32"

        # Model: FP32, Yaw range: [0, pi/2]
        #pred_dir = "predictions/all_bev_preds_regularized/val_trt_fp32_rgd/labels"
        # Model: FP16, Yaw range: [0, pi/2]
        #pred_dir = "predictions/all_bev_preds_regularized/val_trt_fp16_rgd/labels"
        # Model: INT8, Yaw range: [0, pi/2]
        #pred_dir = "predictions/all_bev_preds_regularized/val_trt_int8_rgd/labels"

        #output_dir = "predictions/all_bev_preds_regularized/pred_bev_to_lidar_fp32_rgd"

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

            save_transf_lidar_labels(output_dir, lidar_idx, lidar_labels)
            
            bar.next()

        bar.finish()

# Future work:
# Reverse engineering the z information from lidar_bev script
# height = (pixel_value * (Z_MAX_HEIGHT - Z_MIN_HEIGHT) / 255.0) + Z_MIN_HEIGHT - OFFSET_LIDAR
# height = (pixel_value * (1.27 - (-2.73)) / 255.0) + (-2.73) + 2.73
# 
# or with this function
# https://github.com/AlejandroBarrera/birdnet2/blob/5ceed811b289796d7d7420a064ecb079c80801ab/tools/birdview_detection_refiner.py#L293