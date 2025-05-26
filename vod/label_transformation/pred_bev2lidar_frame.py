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
            'width': float(parts[3]), # YOLO format width = KITTI format length
            'height': float(parts[4]), # YOLO format height = KITTI format width
            'rotation': float(parts[5]), # in radians
            'confidence': float(parts[6]),
        }

    def convert_to_lidar_label(self, pred, regularized=False):
        """Convert YOLO BEV prediction to LiDAR label"""
        
        # Convert center and dimensions to world coordinates
        center, dimensions = pixel_to_world_coords_pred(
            (pred['x_center'], pred['y_center']),
            (pred['width'], pred['height']),  # assign width=length, height=width for KITTI
            self.image_width,
            self.image_height,
            self.cell_size
        )

        #print(f"x,y: {center[0], center[1]}")
        #print(f"w,l: {dimensions[1], dimensions[0]}")

        """
        YOLO Prediction Format                    
        x4,y4 --- x1,y1                
         |           |                 
         |           |      
         |           |                
        x3,y3 --- x2,y2               
        """

        # Ensure that rotation is in range. Since the zero angle in the BEV is 
        # along the image x-axis (right) and in the LiDAR along the x-axis (front), 
        # you must shift the angle by +90° (i.e. +π/2) to align the reference axes.
        #r = normalize_angle_pred(pred['rotation']) # depends on how the angle comes, regularized or raw
        #r = r + pi/2 # do i really need this ???
        
        # Case 1: rotation is regularized CW, prediction angle is [0°...90°]
        if regularized:
            heading = pred['rotation']
            normalize_angle_pred(heading)
            #print(f"A rot_z: {heading}")
            
        # Case 2: rotation is raw CW, the biggest values were [-28°...122°], so it's from [-pi/4...3pi/4] range
        else:
            if (pred['confidence'] * 100) > 0.1:                
                heading = pred['rotation']
                # Approaches test:
                # pred (raw)
                # -pred (inverted)
                # pred - pi (aligned I)
                # pi - pred (aligned II)
                #print(f"B rot_z: {heading}")
        
        # Default height values based on class
        # source: "BirdNet+: End-to-End 3D Object Detection in LiDAR Bird's Eye View"
        default_heights = {1: 1.53, 2: 1.76, 3: 1.74}  # Car: 1.53m, Ped: 1.76m, Cyc: 1.74m
        height = default_heights.get(pred['class_id'])

        # Map class_id to type
        class_map = {1: "Car", 2: "Pedestrian", 3: "Cyclist"}
        obj_type = class_map.get(pred['class_id'], "DontCare")
      
        #print(f"class: {obj_type}")
        #print(f"conf: {pred['confidence'] * 100}")

        # Create LiDAR label in default KITTI format
        lidar_label = {
            "type": str(obj_type),
            "truncated": float(0.0), # float
            "occluded": int(0), # int
            "alpha": float(0.0), # float
            "bbox": [0.0, 0.0, 0.0, 0.0], # xmin, ymin, xmax, ymax
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
        # Test some files
        # [pi/4...3pi/4]
        pred_file = "predictions/all_bev_preds_minAreaRect()/val_trt_fp32/labels/bev_val_000156.txt"
        # [0...pi/2]
        #pred_file = "predictions/all_bev_preds_regularized/val_trt_fp32_rgd/labels/bev_val_000004.txt"

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
            save_transf_lidar_labels(output_dir, lidar_idx, lidar_labels)
            
            bar.next()

        bar.finish()

# Future work:
# Reverse engineering the height information from lidar_bev script
# height = (pixel_value * (Z_MAX_HEIGHT - Z_MIN_HEIGHT) / 255.0) + Z_MIN_HEIGHT - OFFSET_LIDAR
# height = (pixel_value * (1.27 - (-2.73)) / 255.0) + (-2.73) + 2.73
# 
# or with this function
# https://github.com/AlejandroBarrera/birdnet2/blob/5ceed811b289796d7d7420a064ecb079c80801ab/tools/birdview_detection_refiner.py#L293