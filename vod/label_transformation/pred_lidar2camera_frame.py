import numpy as np
import pickle
from pathlib import Path
import copy
from vod.label_transformation.utils.utils import save_transf_camera_labels, cart_to_hom, boxes3d_to_corners3d_kitti_camera

class PredLiDARtoCameraConverter:
    def __init__(self):
        """
        Initialize converter with calibration data from dataset.
        """
        self.dataset = None
        self.calib_data = {}
        self.P2 = None
        self.R0 = None
        self.V2C = None
        self.image_shape = None

   
    def load_calib_from_pkl(self, dataset_path):
        """
        Load dataset to get calibration data from .pkl file.

        Args:
            dataset_path (str or Path): Path to the pickle file 

        Returns:
            None
        """
        with open(dataset_path, 'rb') as f:
            self.dataset = pickle.load(f)
            
        # Extract calibration data for each frame
        for frame in self.dataset:
            if 'point_cloud' in frame and 'calib' in frame:
                lidar_idx = frame['point_cloud']['lidar_idx']
                calib = frame['calib']

                if 'image' in frame:
                    image_shape = frame['image']['image_shape']

                self.calib_data[lidar_idx] = {
                    'P2': calib['P2'][:3],  # 3 x 4
                    'R0': calib['R0_rect'][:3, :3],  # 3 x 3
                    'Tr_velo2cam': calib['Tr_velo_to_cam'][:3],  # 3 x 4
                    'image_shape': image_shape # [height, width]    
                }
    

    def get_calib_for_frame(self, lidar_idx):
        """
        Get calibration data for specific frame.
        
        Args:
            lidar_idx (str): LiDAR frame identifier/index

        Returns:
            dict: Calibration data dictionary containing P2, R0, and Tr_velo2cam matrices
        """
        if lidar_idx not in self.calib_data:
            raise ValueError(f"No calibration data found for frame {lidar_idx}")
        
        calib = self.calib_data[lidar_idx]
        self.P2 = calib['P2']
        self.R0 = calib['R0']
        self.V2C = calib['Tr_velo2cam']
        self.image_shape = calib['image_shape']

        return calib
    

    def lidar_to_rect(self, pts_lidar):
        """
        Convert points from LiDAR to camera rect coordinates.

        Args:
            pts_lidar: Points in LiDAR coordinates, shape (N, 3)
            
        Returns:
            np.ndarray: Points in camera rectified coordinates, shape (N, 3)   
        """
        pts_lidar_hom = cart_to_hom(pts_lidar)
        pts_rect = np.dot(pts_lidar_hom, np.dot(self.V2C.T, self.R0.T))

        return pts_rect
    

    def rect_to_img(self, pts_rect):
        """
        Project rectified camera coordinates to image plane.
        
        Args:
            pts_rect: Points in rectified camera coordinates, shape (N, 3)
            
        Returns:
            tuple: (pts_img, pts_rect_depth)
                - pts_img (np.ndarray): Image coordinates, shape (N, 2)
                - pts_rect_depth (np.ndarray): Depth values in rectified camera coords, shape (N,)
        """
        pts_rect_hom = cart_to_hom(pts_rect)
        pts_2d_hom = np.dot(pts_rect_hom, self.P2.T)
        pts_img = (pts_2d_hom[:, 0:2].T / pts_rect_hom[:, 2]).T  # (N, 2)
        pts_rect_depth = pts_2d_hom[:, 2] - self.P2.T[3, 2]  # depth in rect camera coord

        return pts_img, pts_rect_depth


    def boxes3d_lidar_to_kitti_camera_pred(self, boxes3d_lidar):
        """
        Convert 3D boxes from LiDAR to KITTI camera frame.

        Args:
            boxes3d_lidar (np.ndarray): 3D boxes in LiDAR coordinates, shape (N, 7)
                                       Format: [x, y, z, h, w, l, heading]
                                       
        Returns:
            np.ndarray: 3D boxes in camera coordinates, shape (N, 7)
                       Format: [x, y, z, h, w, l, rotation_y] in rectified camera coords
        """
        boxes3d_lidar_copy = copy.deepcopy(boxes3d_lidar)
        xyz_lidar = boxes3d_lidar_copy[:, 0:3] 
        h, w, l = boxes3d_lidar_copy[:, 3:4], boxes3d_lidar_copy[:, 4:5], boxes3d_lidar_copy[:, 5:6]
        heading = boxes3d_lidar_copy[:, 6:7]

        xyz_lidar[:, 2] -= h.reshape(-1) / 2
        xyz_cam = self.lidar_to_rect(xyz_lidar)
        # Turn the rotation direction from CW (LiDAR) to CCW (Camera). 
        # Shift the reference angle, as the y-axis is vertical in the 
        # camera frame and the z-axis in the LiDAR frame.
        ry = -heading - np.pi / 2 

        return np.concatenate([xyz_cam, h, w, l, ry], axis=-1)
    

    def boxes3d_kitti_camera_to_imageboxes(self, boxes3d_camera, image_shape=None):
        """
        Convert 3D camera boxes to 2D image bounding boxes.
        
        Args:
            boxes3d_camera (np.ndarray): 3D boxes in camera coordinates, shape (N, 7)
                                        Format: [x, y, z, h, w, l, rotation_y]
            image_shape (list, optional): Image dimensions [height, width] for clipping.
                                         If None, no clipping is applied
                                         
        Returns:
            np.ndarray: 2D bounding boxes, shape (N, 4)
                       Format: [xmin, ymin, xmax, ymax] in image coordinates
        """
        corners3d = boxes3d_to_corners3d_kitti_camera(boxes3d_camera)
        pts_img, _ = self.rect_to_img(corners3d.reshape(-1, 3))
        corners_in_image = pts_img.reshape(-1, 8, 2)

        min_uv = np.min(corners_in_image, axis=1)  # (N, 2)
        max_uv = np.max(corners_in_image, axis=1)  # (N, 2)
        boxes2d_image = np.concatenate([min_uv, max_uv], axis=1)
        if image_shape is not None:
            boxes2d_image[:, 0] = np.clip(boxes2d_image[:, 0], a_min=0, a_max=image_shape[1] - 1)
            boxes2d_image[:, 1] = np.clip(boxes2d_image[:, 1], a_min=0, a_max=image_shape[0] - 1)
            boxes2d_image[:, 2] = np.clip(boxes2d_image[:, 2], a_min=0, a_max=image_shape[1] - 1)
            boxes2d_image[:, 3] = np.clip(boxes2d_image[:, 3], a_min=0, a_max=image_shape[0] - 1)

        return boxes2d_image

    
    def parse_lidar_label(self, label_line):
        """
        Parse a single line from LiDAR label file into structured format.
        
        Args:
            label_line (str): Single line from label file containing space-separated values
                             Format: type truncated occluded alpha bbox dimensions location rotation_z score
                             
        Returns:
            dict: Parsed label data with keys
        """
        parts = label_line.strip().split()
        return {
            'type': str(parts[0]),
            'truncated': float(parts[1]),
            'occluded': int(parts[2]),
            'alpha': float(parts[3]), 
            'bbox': [float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])],  # xmin, ymin, xmax, ymax
            'dimensions': [float(parts[8]), float(parts[9]), float(parts[10])],  # h, w, l
            'location': [float(parts[11]), float(parts[12]), float(parts[13])],   # x, y, z
            'rotation_z': float(parts[14]), 
            'score': float(parts[15])
        }

    def convert_label(self, lidar_label):
        """
        Convert LiDAR label to camera frame using OpenPCDet transformation method.
        
        Args:
            lidar_label (dict): Label data in LiDAR frame format from parse_lidar_label()
            
        Returns:
            dict: Label data converted to camera frame format with keys
        """
        # Extract box parameters
        x, y, z = lidar_label['location']
        h, w, l = lidar_label['dimensions']
        heading = lidar_label['rotation_z']
 
        # Create box array in OpenPCDet format [x,y,z,h,w,l,r]
        box3d_lidar = np.array([[x, y, z, h, w, l, heading]])

        # Convert LiDAR 3D Boxes to Camera 3D Boxes using OpenPCDet method
        box3d_camera = self.boxes3d_lidar_to_kitti_camera_pred(box3d_lidar)
        x_rect, y_rect, z_rect, h, w, l, rotation_y = box3d_camera[0]
        
        # Convert Camera 3D Boxes to Camera 2D Boxes using OpenPCDet method
        box2d_camera = self.boxes3d_kitti_camera_to_imageboxes(box3d_camera, self.image_shape)
        xmin, ymin, xmax, ymax = box2d_camera[0]

        # Create Camera label in default KITTI format
        return {
            'type': lidar_label['type'],
            'truncated': float(lidar_label['truncated']), # adopted dummy value
            'occluded': int(lidar_label['occluded']), # adopted dummy value
            'alpha': lidar_label['alpha'], # adopted dummy value
            'bbox': [xmin, ymin, xmax, ymax], # Use calculated 2D bbox values
            'dimensions': [h, w, l],  # h, w, l 
            'location': [x_rect, y_rect, z_rect], # x, y, z
            'rotation_y': rotation_y,
            'score': float(lidar_label['score'])
        }
    
if __name__ == "__main__":

    single_file_mode = False

    # Initialize converter
    converter = PredLiDARtoCameraConverter()

    # Load dataset with calibration info
    dataset_path = Path("validation_pickle/kitti_val_dataset.pkl")
    converter.load_calib_from_pkl(dataset_path)


    if single_file_mode:
        pred_file = "predictions/all_bev_preds_minAreaRect()/pred_bev_to_lidar_fp32/000006.txt"
        lidar_idx = Path(pred_file).stem
        output_dir = "predictions/all_bev_preds_minAreaRect()/pred_lidar_to_camera_fp32"

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
                            f"{camera_label['alpha']} {camera_label['bbox']} " \
                            f"{camera_label['dimensions'][0]:.2f} {camera_label['dimensions'][1]:.2f} {camera_label['dimensions'][2]:.2f} " \
                            f"{camera_label['location'][0]:.2f} {camera_label['location'][1]:.2f} {camera_label['location'][2]:.2f} " \
                            f"{camera_label['rotation_y']:.2f} {camera_label['score']}"
                    #print("Output (Camera):", output)

                #save_transf_camera_labels(output_dir, lidar_idx, camera_labels)
        
        except FileNotFoundError:
            print(f"Label file not found: {pred_file}")
        except ValueError as e:
            print(f"Error: {e}")
    
    else:
        import os
        import glob
        from progress.bar import IncrementalBar

        # Model: TRT FP32, Yaw range: [-pi/4...3pi/4]
        pred_dir = "predictions/all_bev_preds_minAreaRect()/pred_bev_to_lidar_fp32"
        # Model: TRT FP16, Yaw range: [-pi/4...3pi/4]
        #pred_dir = "predictions/all_bev_preds_minAreaRect()/pred_bev_to_lidar_fp16"
        # Model: TRT INT8, Yaw range: [-pi/4...3pi/4]
        #pred_dir = "predictions/all_bev_preds_minAreaRect()/pred_bev_to_lidar_int8"
        # Model: PT FP32, Yaw range: [-pi/4...3pi/4]
        #pred_dir = "predictions/all_bev_preds_minAreaRect()/pred_bev_to_lidar_pt_fp32"

        output_dir = "predictions/all_bev_preds_minAreaRect()/pred_lidar_to_camera_fp32"

        # Model: TRT FP32, Yaw range: [0, pi/2]
        #pred_dir = "predictions/all_bev_preds_regularized/pred_bev_to_lidar_fp32_rgd"
        #output_dir = "predictions/all_bev_preds_regularized/pred_lidar_to_camera_fp32_rgd"

        if not os.path.exists(pred_dir):
            print(f"Directoy '{pred_dir}' not found.")
            exit(1)
        
        lidar_label_files = glob.glob(os.path.join(pred_dir, "*.txt"))
        bar = IncrementalBar('Processing', max=len(lidar_label_files), 
                             suffix='%(percent).1f%% - Estimated time: %(eta)ds')

        for lidar_label_file in lidar_label_files:
            lidar_idx = Path(lidar_label_file).stem
            converter.get_calib_for_frame(lidar_idx)

            camera_labels = []
            with open(lidar_label_file, 'r') as f:
                for line in f:
                    lidar_label = converter.parse_lidar_label(line)
                    camera_label = converter.convert_label(lidar_label)
                    camera_labels.append(camera_label)

                save_transf_camera_labels(output_dir, lidar_idx, camera_labels)

            bar.next()
            
        bar.finish()