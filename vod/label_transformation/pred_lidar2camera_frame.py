import numpy as np
import pickle
from pathlib import Path
import copy
from vod.label_transformation.utils.utils import save_transf_camera_labels, cart_to_hom, normalize_angle_pred

class PredLiDARtoCameraConverter:
    def __init__(self):
        """Initialize converter with calibration data from dataset"""
        self.dataset = None
        self.calib_data = {}
        self.P2 = None
        self.R0 = None
        self.V2C = None

   
    def load_dataset(self, dataset_path):
        """Load dataset to get calibration data"""
        with open(dataset_path, 'rb') as f:
            self.dataset = pickle.load(f)
            
        # Extract calibration data for each frame
        for frame in self.dataset:
            if 'point_cloud' in frame and 'calib' in frame:
                lidar_idx = frame['point_cloud']['lidar_idx']
                calib = frame['calib']
                self.calib_data[lidar_idx] = {
                    'P2': calib['P2'][:3],  # 3 x 4
                    'R0': calib['R0_rect'][:3, :3],  # 3 x 3
                    'Tr_velo2cam': calib['Tr_velo_to_cam'][:3],  # 3 x 4
                }
    

    def get_calib_for_frame(self, lidar_idx):
        """Get calibration data for specific frame"""
        if lidar_idx not in self.calib_data:
            raise ValueError(f"No calibration data found for frame {lidar_idx}")
        
        calib = self.calib_data[lidar_idx]
        self.P2 = calib['P2']
        self.R0 = calib['R0']
        self.V2C = calib['Tr_velo2cam']

        return calib
    

    def lidar_to_rect(self, pts_lidar):
        """Convert points from LiDAR to camera rect coordinates
        Args:
            pts_lidar: (N, 3)
        Returns:
            pts_rect: (N, 3)    
        """
        pts_lidar_hom = cart_to_hom(pts_lidar)
        pts_rect = np.dot(pts_lidar_hom, np.dot(self.V2C.T, self.R0.T))
        return pts_rect
    

    def boxes3d_lidar_to_kitti_camera_pred(self, boxes3d_lidar):
        """Convert 3D boxes from LiDAR to KITTI camera frame
        Args:
            boxes3d_lidar: (N, 7) [x, y, z, h, w, l, heading]
        Returns:
            boxes3d_camera: (N, 7) [x, y, z, h, w, l, ry] in rect camera coords
        """
        boxes3d_lidar_copy = copy.deepcopy(boxes3d_lidar)
        xyz_lidar = boxes3d_lidar_copy[:, 0:3] 
        h, w, l = boxes3d_lidar_copy[:, 3:4], boxes3d_lidar_copy[:, 4:5], boxes3d_lidar_copy[:, 5:6]
        r = boxes3d_lidar_copy[:, 6:7]

        xyz_lidar[:, 2] -= h.reshape(-1) / 2
        xyz_cam = self.lidar_to_rect(xyz_lidar)
        r = -r - np.pi / 2

        return np.concatenate([xyz_cam, h, w, l, r], axis=-1)
    
    def parse_lidar_label(self, label_line):
        """Parse a line from LiDAR label file"""
        parts = label_line.strip().split()
        return {
            'type': str(parts[0]),
            'truncated': float(parts[1]),
            'occluded': int(parts[2]),
            'alpha': float(parts[3]),  # Convert to float
            'bbox_pre_height': float(parts[4]),  # Convert to float
            'dimensions': [float(parts[5]), float(parts[6]), float(parts[7])],  # h, w, l
            'location': [float(parts[8]), float(parts[9]), float(parts[10])],   # x, y, z
            'rotation_z': float(parts[11]),  # Convert to float
            'score': float(parts[12])
        }



    def convert_label(self, lidar_label):
        """Convert LiDAR label to camera frame using OpenPCDet method"""
        # Extract box parameters
        x, y, z = lidar_label['location']
        h, w, l = lidar_label['dimensions']
        r = lidar_label['rotation_z']
        obj_type = lidar_label['type']

    
        #if obj_type in ['Car', 'Cyclist'] and w > l:
        #    w, l = l, w
 
        # Create box array in OpenPCDet format [x,y,z,h,w,l,r]
        box_lidar = np.array([[x, y, z, h, w, l, r]])

        # Convert using OpenPCDet method
        box_camera = self.boxes3d_lidar_to_kitti_camera_pred(box_lidar)
        x_rect, y_rect, z_rect, h, w, l, rotation_y = box_camera[0]


        return {
            'type': lidar_label['type'],
            'truncated': float(lidar_label['truncated']),
            'occluded': int(lidar_label['occluded']),
            'alpha': lidar_label['alpha'],
            'bbox_pre_height': lidar_label['bbox_pre_height'],
            'dimensions': [h, w, l],  # Keep h, w, l order
            'location': [x_rect, y_rect, z_rect],
            'rotation_y': (rotation_y),
            'score': float(lidar_label['score'])
        }
    
if __name__ == "__main__":

    single_file_mode = True

    # Initialize converter
    converter = PredLiDARtoCameraConverter()

    # Load dataset with calibration info
    dataset_path = Path("validation_pickle/kitti_val_dataset.pkl")
    converter.load_dataset(dataset_path)

    if single_file_mode:
        pred_file = "predictions/pred_bev_to_lidar_fp32/000040.txt"
        lidar_idx = Path(pred_file).stem
        output_dir = "predictions/pred_lidar_to_camera_fp32"

        try:
            # Get calibration data for this frame
            converter.get_calib_for_frame(lidar_idx)

            # Read and convert predictions
            camera_labels = []
            with open(pred_file, 'r') as f:
                for line in f:
                    lidar_label = converter.parse_lidar_label(line)
                    camera_label = converter.convert_label(lidar_label)
                    camera_labels.append(camera_label)

                    output = f"{camera_label['type']} {camera_label['truncated']} {camera_label['occluded']} " \
                            f"{camera_label['alpha']} {camera_label['bbox_pre_height']} " \
                            f"{camera_label['dimensions'][0]:.2f} {camera_label['dimensions'][1]:.2f} {camera_label['dimensions'][2]:.2f} " \
                            f"{camera_label['location'][0]:.2f} {camera_label['location'][1]:.2f} {camera_label['location'][2]:.2f} " \
                            f"{camera_label['rotation_y']:.2f} {camera_label['score']}"
                    print("Output (Camera):", output)

                save_transf_camera_labels(output_dir, lidar_idx, camera_labels)
        
        except FileNotFoundError:
            print(f"Label file not found: {pred_file}")
        except ValueError as e:
            print(f"Error: {e}")
    
    else:
        import os
        import glob
        from progress.bar import IncrementalBar

        pass

 # TODO: test if change of h, l is required