"""
Open3d visualization tool box
Written by Jihan YANG
Modified by Petros626
All rights preserved from 2021 - present.
"""
import open3d
import numpy as np


box_colormap = [
    [1, 1, 1], # not assigned
    [0, 1, 0], # Car Green
    [1, 0, 1], # Pedestrian Violet
    [1, 1, 0], # Cyclist Yellow
]

def draw_scenes(points, gt_boxes=None, ref_boxes=None, ref_labels=None, 
                ref_scores=None, point_colors=None, draw_origin=True, 
                save_image=False, output_path=None, draw_obj_heading=False):
    
    if not isinstance(points, np.ndarray):
        points = np.asarray(points)
    if gt_boxes is not None and not isinstance(gt_boxes, np.ndarray):
        gt_boxes = np.asarray(gt_boxes)
    if ref_boxes is not None and not isinstance(ref_boxes, np.ndarray):
        ref_boxes = np.asarray(ref_boxes)

    vis = open3d.visualization.Visualizer()
    vis.create_window(width=1200, height=900)

    vis.get_render_option().point_size = 1.0
    vis.get_render_option().background_color = np.zeros(3)

    # draw origin
    if draw_origin:
        axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        vis.add_geometry(axis_pcd)

    pts = open3d.geometry.PointCloud()
    pts.points = open3d.utility.Vector3dVector(points[:, :3])

    vis.add_geometry(pts)
    if point_colors is None:
        pts.colors = open3d.utility.Vector3dVector(np.ones((points.shape[0], 3)))
    else:
        pts.colors = open3d.utility.Vector3dVector(point_colors)

    if gt_boxes is not None:
        vis = draw_box(vis, gt_boxes, (1, 0, 0), use_class_colors=False, is_pred=False, draw_diagonals=draw_obj_heading)

    if ref_boxes is not None:
        vis = draw_box(vis, ref_boxes, (1, 0, 0), 
                       ref_labels, ref_scores, use_class_colors=True, is_pred=True, draw_diagonals=False)
    
    view_control = vis.get_view_control()
    params = open3d.io.read_pinhole_camera_parameters('ScreenCamera_val.json')
    view_control.convert_from_pinhole_camera_parameters(params,allow_arbitrary=True)

    vis.run()

    if save_image and output_path:
        vis.poll_events()
        vis.update_renderer()
        print("Saving screenshot...")
        vis.capture_screen_image(output_path, do_render=True)

    vis.destroy_window()

def translate_boxes_to_open3d_instance(gt_boxes, is_pred=False):
    """ Standard Open3D Box
             4-------- 6
           /|         /|
          5 -------- 3 .
          | |        | |
          . 7 -------- 1
          |/         |/
          2 -------- 0
    """
    center = gt_boxes[0:3]
    lwh = gt_boxes[3:6]
    axis_angles = np.array([0, 0, gt_boxes[6] + 1e-10])
    rot = open3d.geometry.get_rotation_matrix_from_axis_angle(axis_angles)
    box3d = open3d.geometry.OrientedBoundingBox(center, rot, lwh)

    line_set = open3d.geometry.LineSet.create_from_oriented_bounding_box(box3d)
    
    lines = np.asarray(line_set.lines)

    if not is_pred: # diagonal only for gt boxes
        diagonal_lines = np.array([[1, 4], [7, 6]]) # creates diagonal cross in direction of the object
        lines = np.concatenate([lines, diagonal_lines], axis=0)

    line_set.lines = open3d.utility.Vector2iVector(lines) # 3D-BBox lines

    return line_set, box3d

def create_box_label(vis, box3d, score, ref_label=None, is_pred=False):
    class_names = {1: "Car", 2: "Ped", 3: "Cyc"}

    if is_pred and ref_label is not None:
        class_name = class_names.get(ref_label, f'C{ref_label}')
        S = round(score, 2)
        txt = f"{class_name} {S:.2f}"
        label_color = box_colormap[ref_label]
    else: # fallback
        S = round(score, 2)
        txt = f"{S:.2f}"
        label_color = (1, 1, 0)

    text_mesh = open3d.t.geometry.TriangleMesh.create_text(txt, depth=5).to_legacy()
    text_mesh.paint_uniform_color(label_color)

    corners = box3d.get_box_points()
    location = corners[6] + np.array([0, -1, 0])

    rotation_matrix = open3d.geometry.get_rotation_matrix_from_xyz([0, -np.pi/2, -np.pi/2])
    text_mesh.scale(0.05, center=[0, 0, 0])
    text_mesh.rotate(rotation_matrix, center=[0, 0, 0])
    text_mesh.translate(location)
    
    vis.add_geometry(text_mesh)

    return vis

def draw_box(vis, gt_boxes, color=(0, 1, 0), ref_labels=None, 
             score=None, use_class_colors=False, is_pred=False, 
             draw_diagonals=False):
    
    for i in range(gt_boxes.shape[0]):
        line_set, box3d = translate_boxes_to_open3d_instance(
            gt_boxes[i], is_pred=is_pred)

        if not use_class_colors:
            if ref_labels is None:
                line_set.paint_uniform_color(color) # GT: red
            else:
                line_set.paint_uniform_color(box_colormap[ref_labels[i]]) # Pred: cls colors
        else:
            if ref_labels is not None:
                line_set.paint_uniform_color(box_colormap[ref_labels[i]])

        if draw_diagonals and not is_pred:
            colors = np.asarray(line_set.colors)
            colors[-2:] = [0, 1, 1] # cyan for diagonals
            line_set.colors = open3d.utility.Vector3dVector(colors)

        vis.add_geometry(line_set)

        if score is not None and len(score) > i:
            label = ref_labels[i] if ref_labels is not None and len(ref_labels) > i else None
            vis = create_box_label(vis, box3d, score[i], label, is_pred)

        # Old code from open3d 0.9.0
        #if score is not None:
        #     corners = box3d.get_box_points()
        #     vis.add_3d_label(corners[5], '%.2f' % score[i])
    return vis
