import numpy as np
from pathlib import Path

def load_labels(label_file):
    """Load labels from file into structured format"""
    labels = []
    with open(label_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            labels.append({
                'type': parts[0],
                'location': [float(parts[8]), float(parts[9]), float(parts[10])],
                'dimensions': [float(parts[5]), float(parts[6]), float(parts[7])],
                'rotation_y': float(parts[11])
            })
    return labels

def find_nearest_prediction(gt_label, pred_labels):
    """Find prediction that best matches the ground truth label"""
    min_dist = float('inf')
    best_match = None
    
    gt_x, gt_y = gt_label['location'][0:2]
    
    for pred in pred_labels:
        if pred['type'] != gt_label['type']:
            continue
            
        pred_x, pred_y = pred['location'][0:2]
        dist = np.sqrt((gt_x - pred_x)**2 + (gt_y - pred_y)**2)
        
        if dist < min_dist:
            min_dist = dist
            best_match = pred
            
    return best_match, min_dist

def format_label(label):
    """Format label for pretty printing"""
    return (f"  type: {label['type']}\n"
            f"  location (x,y,z): ({label['location'][0]:.2f}, {label['location'][1]:.2f}, {label['location'][2]:.2f})\n"
            f"  dimensions (h,w,l): ({label['dimensions'][0]:.2f}, {label['dimensions'][1]:.2f}, {label['dimensions'][2]:.2f})\n"
            f"  rotation_y: {label['rotation_y']:.2f}rad ({label['rotation_y']*180/np.pi:.1f}°)")

def compare_labels(gt_file, pred_file, distance_threshold=2.0):
    """Compare GT and prediction labels focusing on x,y,w,l,r"""
    gt_labels = load_labels(gt_file)
    pred_labels = load_labels(pred_file)
    
    print(f"Total GT labels: {len(gt_labels)}, Total Pred labels: {len(pred_labels)}")
    
    # Gruppiere nach Objekttyp
    types = set(l['type'] for l in gt_labels)
    
    for obj_type in types:
        gt_obj = [l for l in gt_labels if l['type'] == obj_type]
        pred_obj = [l for l in pred_labels if l['type'] == obj_type]
        
        if not gt_obj or not pred_obj:
            continue
            
        print(f"\n{obj_type} (GT: {len(gt_obj)}, Pred: {len(pred_obj)}):")
        
        # Finde und zeige beste Matches
        for i, gt in enumerate(gt_obj, 1):
            match, dist = find_nearest_prediction(gt, pred_obj)
            if match and dist < distance_threshold:
                print(f"\nMatch {i} (distance: {dist:.2f}m):")
                print("Ground Truth:")
                print(format_label(gt))
                print("\nNearest Prediction:")
                print(format_label(match))
                print("-" * 50)
            else:
                print(f"\nNo match found for GT {i} within {distance_threshold}m threshold")
                print("Ground Truth:")
                print(format_label(gt))
                print("-" * 50)

if __name__ == "__main__":
    frame_id = "000040"
    gt_file = f"kitti_gt_annos/gt_lidar_to_camera_labels/{frame_id}.txt"
    pred_file = f"predictions/pred_lidar_to_camera_fp32/{frame_id}.txt"
    
    compare_labels(gt_file, pred_file)