import numpy as np
import pickle
from pathlib import Path
import copy
from vod.label_transformation.utils.utils import save_transf_camera_labels, cart_to_hom, boxes3d_to_corners3d_kitti_camera

class LiDARtoCameraConverter:
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
        Load dataset and extract calibration data from pickle file.
    
        Args:
            dataset_path (str or Path): Path to the pickle file containing dataset with calibration data
            
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
        Get calibration data for specific frame and set internal matrices.
    
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
            pts_lidar: (N, 3) [x, y, z]
        Returns:
            pts_rect: (N, 3) [x, y, z]    
        """
        pts_lidar_hom = cart_to_hom(pts_lidar)
        pts_rect = np.dot(pts_lidar_hom, np.dot(self.V2C.T, self.R0.T))
        return pts_rect
    

    def rect_to_img(self, pts_rect):
        """
        :param pts_rect: (N, 3)
        :return pts_img: (N, 2)
        """
        pts_rect_hom = cart_to_hom(pts_rect)
        pts_2d_hom = np.dot(pts_rect_hom, self.P2.T)
        pts_img = (pts_2d_hom[:, 0:2].T / pts_rect_hom[:, 2]).T  # (N, 2)
        pts_rect_depth = pts_2d_hom[:, 2] - self.P2.T[3, 2]  # depth in rect camera coord

        return pts_img, pts_rect_depth
    

    def boxes3d_lidar_to_kitti_camera(self, boxes3d_lidar):
        """
        Convert 3D boxes from LiDAR to KITTI camera frame.

        Args:
            boxes3d_lidar: (N, 7) [x, y, z, h, w, l, heading]
        Returns:
            boxes3d_camera: (N, 7) [x, y, z, h, w, l, ry] in rect camera coords
        """
        boxes3d_lidar_copy = copy.deepcopy(boxes3d_lidar)
        xyz_lidar = boxes3d_lidar_copy[:, 0:3] 
        h, w, l = boxes3d_lidar_copy[:, 3:4], boxes3d_lidar_copy[:, 4:5], boxes3d_lidar_copy[:, 5:6]
        heading = boxes3d_lidar_copy[:, 6:7]

        xyz_lidar[:, 2] -= h.reshape(-1) / 2
        xyz_cam = self.lidar_to_rect(xyz_lidar)
        r_y = -heading - np.pi / 2 # Adjust rotation (LiDAR-CW → Camera-CCW + axis correction)

        return np.concatenate([xyz_cam, h, w, l, r_y], axis=-1)
    

    def boxes3d_kitti_camera_to_imageboxes(self, boxes3d_camera, image_shape=None):
        """
        Args:
            boxes3d_camera: (N, 7) [x, y, z, h, w, l, ry] in rect camera coords
        Returns: 
            boxes_2d_preds: (N, 4) [x1, y1, x2, y2] = [xmin, ymin, xmax, ymax]
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
            'alpha': float(parts[3]),  # Convert to float
            'bbox': [float(parts[4]), (parts[5]), (parts[6]), (parts[7])],  # Convert to float
            'dimensions': [float(parts[8]), float(parts[9]), float(parts[10])],  # h, w, l
            'location': [float(parts[11]), float(parts[12]), float(parts[13])],   # x, y, z
            'rotation_z': float(parts[14]),  # Convert to float
            'score': float(parts[15])
        }


    def lidar_to_camera_label(self, lidar_label):
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
        heading_lidar = lidar_label['rotation_z']
 
        # Create box array in OpenPCDet format [x,y,z,h,w,l,heding]
        box3d_lidar = np.array([[x, y, z, h, w, l, heading_lidar]])

        # Convert using OpenPCDet method
        box3d_camera = self.boxes3d_lidar_to_kitti_camera(box3d_lidar)
        x_rect, y_rect, z_rect, h, w, l, rotation_y = box3d_camera[0]

        box2d_camera = self.boxes3d_kitti_camera_to_imageboxes(box3d_camera, self.image_shape)
        xmin, ymin, xmax, ymax = box2d_camera[0]
        #print(f"{lidar_label['type']}")
        #print(f"Esti.: {xmin, ymin, xmax, ymax}\n")

        return {
            'type': lidar_label['type'],
            'truncated': float(lidar_label['truncated']),
            'occluded': int(lidar_label['occluded']),
            'alpha': lidar_label['alpha'],
            #'bbox': lidar_label['bbox'], # from gt data
            'bbox': [xmin, ymin, xmax, ymax], # calculated 2D boxes
            'dimensions': [h, w, l],  # h, w, l
            'location': [x_rect, y_rect, z_rect], # x, y, z
            'rotation_y': rotation_y,
            'score': float(lidar_label['score'])
        }
   
    
if __name__ == "__main__":

    single_file_mode = False

    # Initialize converter
    converter = LiDARtoCameraConverter()
    
    # Load dataset with calibration info
    dataset_path = Path("validation_pickle/kitti_val_dataset.pkl")
    converter.load_calib_from_pkl(dataset_path)

    if single_file_mode:
        test_label_path = "predictions/pred_bev_to_lidar_fp32/000002.txt"
        lidar_idx = Path(test_label_path).stem
        output_dir = "kitti_gt_annos/gt_lidar_to_camera_labels"

        try:
            # Get calibration data for this frame
            converter.get_calib_for_frame(lidar_idx)

            # Read and convert labels
            camera_labels = []
            with open(test_label_path, 'r') as f:
                for line in f:
                    #print("Input (LiDAR):", line.strip())
                    lidar_label = converter.parse_lidar_label(line)
                    camera_label = converter.lidar_to_camera_label(lidar_label)
                    camera_labels.append(camera_label)

                    output = f"{camera_label['type']} {camera_label['truncated']} {camera_label['occluded']} " \
                            f"{camera_label['alpha']} {camera_label['bbox']} " \
                            f"{camera_label['dimensions'][0]:.2f} {camera_label['dimensions'][1]:.2f} {camera_label['dimensions'][2]:.2f} " \
                            f"{camera_label['location'][0]:.2f} {camera_label['location'][1]:.2f} {camera_label['location'][2]:.2f} " \
                            f"{camera_label['rotation_y']:.2f} {camera_label['score']}"
                    #print("Output (Camera):", output)

                #save_transf_camera_labels(output_dir, lidar_idx, camera_labels)
                
        except FileNotFoundError:
            print(f"Label file not found: {test_label_path}")
        except ValueError as e:
            print(f"Error: {e}")

    else:
        import glob
        from progress.bar import IncrementalBar
        import os

        lidar_label_dir = "kitti_gt_annos_2/gt_bev_to_lidar_labels_2"
        output_dir = "kitti_gt_annos_2/gt_lidar_to_camera_labels_2"

        if not os.path.exists(lidar_label_dir):
            print(f"Directoy '{lidar_label_dir}' not found.")
            exit(1)
        
        lidar_label_files = glob.glob(os.path.join(lidar_label_dir, "*.txt"))
        bar = IncrementalBar('Processing', max=len(lidar_label_files), 
                             suffix='%(percent).1f%% - Estimated time: %(eta)ds')

        for lidar_label_file in lidar_label_files:
            lidar_idx = Path(lidar_label_file).stem
            converter.get_calib_for_frame(lidar_idx)

            camera_labels = []
            with open(lidar_label_file, 'r') as f:
                for line in f:
                    lidar_label = converter.parse_lidar_label(line)
                    camera_label = converter.lidar_to_camera_label(lidar_label)
                    camera_labels.append(camera_label)

                save_transf_camera_labels(output_dir, lidar_idx, camera_labels)

            bar.next()
            
        bar.finish()

#current topic: should I recalculate the 2D Boxes or take them from gt, bc the 3d boxes can't be properly regressed
# from the yolo model output parameters.