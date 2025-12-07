# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>, Hao Xiang <haxiang@g.ucla.edu>,
# License: TDG-Attribution-NonCommercial-NoDistrib

import time

import cv2
import numpy as np
import open3d as o3d
import matplotlib
import matplotlib.pyplot as plt

from matplotlib import cm

from opencood.utils import box_utils
from opencood.utils import common_utils
from itertools import zip_longest


VIRIDIS = np.array(cm.get_cmap('plasma').colors)
VID_RANGE = np.linspace(0.0, 1.0, VIRIDIS.shape[0])


def bbx2linset(bbx_corner, order='hwl', color=(0, 1, 0)):
    """
    Convert predicted boxes to Open3D LineSets.
    Accepts None, torch.Tensor, or np.ndarray.
    Returns an empty list if there is nothing to draw.
    """
    import numpy as np
    from opencood.utils import common_utils, box_utils

    # --- handle None early ---
    if bbx_corner is None:
        return []

    # Convert tensors/dicts/lists safely (your torch_tensor_to_numpy already patched)
    if not isinstance(bbx_corner, np.ndarray):
        bbx_corner = common_utils.torch_tensor_to_numpy(bbx_corner)
        if bbx_corner is None:
            return []

    bbx_corner = np.asarray(bbx_corner)

    # Nothing to draw
    if bbx_corner.size == 0:
        return []

    # If input is (N, 7) param boxes, convert to (N, 8, 3) corners
    if bbx_corner.ndim == 2:
        bbx_corner = box_utils.boxes_to_corners_3d(bbx_corner, order)

    # Still nothing?
    if bbx_corner.shape[0] == 0:
        return []

    lines = [[0, 1], [1, 2], [2, 3], [0, 3],
             [4, 5], [5, 6], [6, 7], [4, 7],
             [0, 4], [1, 5], [2, 6], [3, 7]]
    colors = [list(color) for _ in range(len(lines))]
    bbx_linset = []

    for i in range(bbx_corner.shape[0]):
        bbx = bbx_corner[i].copy()
        # o3d uses right-handed; flip X
        bbx[:, :1] = -bbx[:, :1]                                #bbx[:, 1:2] = -bbx[:, 1:2] # flip y axis
                                                                #bbx[:, 2:3] = -bbx[:, 2:3] # flip z axis

        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(bbx)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(colors)
        bbx_linset.append(line_set)

    return bbx_linset


def bbx2oabb(bbx_corner, order='hwl', color=(0, 0, 1)):
    """
    Convert bbox corners or (N,7) boxes to Open3D OABBs for visualization.

    bbx_corner:  (N, 8, 3) corners  OR  (N, 7) boxes (order given by `order`)
                 Can also be a torch.Tensor or numpy.ndarray. May be None/empty.
    """
    # --- early exits for None/empty inputs ---
    if bbx_corner is None:
        return []

    # Accept torch or numpy; handle empties before conversion
    try:
        import torch
    except Exception:
        torch = None

    if torch is not None and isinstance(bbx_corner, torch.Tensor):
        if bbx_corner.numel() == 0 or (bbx_corner.ndim >= 1 and bbx_corner.shape[0] == 0):
            return []
        # move to CPU numpy safely
        bbx_corner = bbx_corner.detach().cpu().numpy()
    else:
        # numpy path
        import numpy as np
        if not isinstance(bbx_corner, np.ndarray):
            # fallback to original util, but guard for None
            if bbx_corner is None:
                return []
            bbx_corner = common_utils.torch_tensor_to_numpy(bbx_corner)

        if bbx_corner.size == 0 or (bbx_corner.ndim >= 1 and bbx_corner.shape[0] == 0):
            return []

    # If shape is (N,7) convert to corners
    if bbx_corner.ndim == 2:  # assume (N,7)
        bbx_corner = box_utils.boxes_to_corners_3d(bbx_corner, order)

    # Validate final shape (N,8,3)
    if bbx_corner.ndim != 3 or bbx_corner.shape[-2:] != (8, 3):
        # malformed input; nothing to draw
        return []

    import numpy as np
    import open3d as o3d

    oabbs = []
    for i in range(bbx_corner.shape[0]):
        bbx = bbx_corner[i]
        if not np.isfinite(bbx).all():
            continue  # skip bad boxes

        # Flip x to match right-hand coord used by Open3D
        bbx[:, :1] = -bbx[:, :1]                                #bbx[:, 1:2] = -bbx[:, 1:2] # flip y axis
                                                                #bbx[:, 2:3] = -bbx[:, 2:3] # flip z axis

        tmp_pcd = o3d.geometry.PointCloud()
        tmp_pcd.points = o3d.utility.Vector3dVector(bbx)

        oabb = tmp_pcd.get_oriented_bounding_box()
        oabb.color = color
        oabbs.append(oabb)

    return oabbs


def bbx2aabb(bbx_center, order):
    """
    Convert the torch tensor bounding box to o3d aabb for visualization.

    Parameters
    ----------
    bbx_center : torch.Tensor
        shape: (n, 7).

    order: str
        hwl or lwh.

    Returns
    -------
    aabbs : list
        The list containing all o3d.aabb
    """
    if not isinstance(bbx_center, np.ndarray):
        bbx_center = common_utils.torch_tensor_to_numpy(bbx_center)
    bbx_corner = box_utils.boxes_to_corners_3d(bbx_center, order)

    aabbs = []

    for i in range(bbx_corner.shape[0]):
        bbx = bbx_corner[i]
        # o3d use right-hand coordinate
        bbx[:, :1] = - bbx[:, :1]                  #bbx[:, 1:2] = -bbx[:, 1:2] # flip y axis
                                                   #bbx[:, 2:3] = -bbx[:, 2:3] # flip z axis

        tmp_pcd = o3d.geometry.PointCloud()
        tmp_pcd.points = o3d.utility.Vector3dVector(bbx)

        aabb = tmp_pcd.get_axis_aligned_bounding_box()
        aabb.color = (0, 0, 1)
        aabbs.append(aabb)

    return aabbs


def linset_assign_list(vis,
                       lineset_list1,
                       lineset_list2,
                       update_mode='update'):
    """
    Safely combine/update two LineSet lists (pred vs gt).
    Works when either list is shorter or empty.
    """
    ls1 = lineset_list1 or []
    ls2 = lineset_list2 or []

    out = []
    for a, b in zip_longest(ls1, ls2, fillvalue=None):
        # lineset_assign(a, b) should handle None (see patch below).
        new_ls = lineset_assign(a, b)
        if new_ls is None:
            continue
        if update_mode == 'add':
            vis.add_geometry(new_ls)
        else:
            vis.update_geometry(new_ls)
        out.append(new_ls)

    return out


def lineset_assign(ls_old, ls_new):
    """
    Update an existing LineSet with a new one. Either may be None.
    If ls_new is None, keep ls_old. If ls_old is None, use ls_new.
    """
    if ls_new is None:
        return ls_old
    if ls_old is None:
        return ls_new

    # copy points/lines/colors from ls_new into ls_old
    ls_old.points = ls_new.points
    ls_old.lines = ls_new.lines
    ls_old.colors = ls_new.colors
    return ls_old


def color_encoding(intensity, mode='intensity'):
    """
    Encode the single-channel intensity to 3 channels rgb color.

    Parameters
    ----------
    intensity : np.ndarray
        Lidar intensity, shape (n,)

    mode : str
        The color rendering mode. intensity, z-value and constant are
        supported.

    Returns
    -------
    color : np.ndarray
        Encoded Lidar color, shape (n, 3)
    """
    assert mode in ['intensity', 'z-value', 'constant']

    if mode == 'intensity':
        intensity = np.asarray(intensity, dtype=float)
        intensity = np.clip(intensity, 1e-6, None)
        intensity_col = 1.0 - np.log(intensity) / np.log(np.exp(-0.004 * 100))
        int_color = np.c_[
            np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 0]),
            np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 1]),
            np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 2])]

    elif mode == 'z-value':
        min_value = -1.5
        max_value = 0.5
        norm = matplotlib.colors.Normalize(vmin=min_value, vmax=max_value)
        cmap = cm.jet
        m = cm.ScalarMappable(norm=norm, cmap=cmap)

        colors = m.to_rgba(intensity)
        colors[:, [2, 1, 0, 3]] = colors[:, [0, 1, 2, 3]]
        colors[:, 3] = 0.5
        int_color = colors[:, :3]

    elif mode == 'constant':
        # regard all point cloud the same color
        int_color = np.ones((intensity.shape[0], 3))
        int_color[:, 0] *= 247 / 255
        int_color[:, 1] *= 244 / 255
        int_color[:, 2] *= 237 / 255

    return int_color


def visualize_single_sample_output_gt(pred_tensor,
                                      gt_tensor,
                                      pcd,
                                      show_vis=True,
                                      pc_offset=[20.0, 0.0, 0.0],
                                      save_path='',
                                      mode='constant'):
    """
    Visualize the prediction, groundtruth with point cloud together.

    Parameters
    ----------
    pred_tensor : torch.Tensor or np.ndarray
        (N, 8, 3) prediction.

    gt_tensor : torch.Tensor or np.ndarray
        (N, 8, 3) groundtruth bbx

    pcd : torch.Tensor or np.ndarray
        PointCloud, (M, 4) or (B, M, 4).

    pc_offset : list/tuple/np.ndarray of length 3
        Offset [dx, dy, dz] in meters applied to the point cloud ONLY.

    show_vis : bool
        Whether to show visualization.

    save_path : str
        Save the visualization results to given path.

    mode : str
        Color rendering mode.
    """

    def custom_draw_geometry(pcd, pred, gt):
        vis = o3d.visualization.Visualizer()
        vis.create_window()

        opt = vis.get_render_option()
        opt.background_color = np.asarray([0, 0, 0])
        opt.point_size = 1.0

        vis.add_geometry(pcd)
        for ele in pred:
            vis.add_geometry(ele)
        for ele in gt:
            vis.add_geometry(ele)

        vis.run()
        vis.destroy_window()

    # ---------- point cloud shape & type ----------
    if len(pcd.shape) == 3:
        # take first frame if batched: (B, M, 4) -> (M, 4)
        pcd = pcd[0]

    if isinstance(pcd, np.ndarray):
        origin_lidar = pcd.copy()
    else:
        origin_lidar = common_utils.torch_tensor_to_numpy(pcd)

    # ---------- apply point cloud offset ----------
    # hard-coded example if pc_offset is None
    if pc_offset is None:
        # you can change this to something huge to see it clearly
        pc_offset = np.array([10.0, 5.0, 0.0], dtype=np.float32)
    else:
        pc_offset = np.asarray(pc_offset, dtype=np.float32)

    # debug: print before / after one point
    print("PC first point BEFORE offset:", origin_lidar[0, :3])
    origin_lidar[:, 0] += pc_offset[0]
    origin_lidar[:, 1] += pc_offset[1]
    origin_lidar[:, 2] += pc_offset[2]
    print("PC first point AFTER  offset:", origin_lidar[0, :3])

    # ---------- color + handedness flip ----------
    origin_lidar_intcolor = \
        color_encoding(origin_lidar[:, -1] if mode == 'intensity'
                       else origin_lidar[:, 2], mode=mode)

    # left -> right hand (flip X)
    origin_lidar[:, :1] = -origin_lidar[:, :1]

    o3d_pcd = o3d.geometry.PointCloud()
    o3d_pcd.points = o3d.utility.Vector3dVector(origin_lidar[:, :3])
    o3d_pcd.colors = o3d.utility.Vector3dVector(origin_lidar_intcolor)

    # ---------- boxes (no offset, no noise) ----------
    oabbs_pred = bbx2oabb(pred_tensor, color=(1, 0, 0))
    oabbs_gt = []
    if gt_tensor is not None:
        oabbs_gt = bbx2oabb(gt_tensor, color=(0, 1, 0))

    # ---------- visualize / save ----------
    visualize_elements = [o3d_pcd] + oabbs_pred + oabbs_gt

    if show_vis:
        custom_draw_geometry(o3d_pcd, oabbs_pred, oabbs_gt)
    if save_path:
        save_o3d_visualization(visualize_elements, save_path)



def visualize_single_sample_output_bev(pred_box, gt_box, pcd, dataset,
                                       show_vis=True,
                                       save_path=''):
    """
    Visualize the prediction, groundtruth with point cloud together in
    a bev format.

    Parameters
    ----------
    pred_box : torch.Tensor
        (N, 4, 2) prediction.

    gt_box : torch.Tensor
        (N, 4, 2) groundtruth bbx

    pcd : torch.Tensor
        PointCloud, (N, 4).

    show_vis : bool
        Whether to show visualization.

    save_path : str
        Save the visualization results to given path.
    """

    if not isinstance(pcd, np.ndarray):
        pcd = common_utils.torch_tensor_to_numpy(pcd)  #conversion of pcd to tensor and then from a tensor to numpy array.
    if pred_box is not None and not isinstance(pred_box, np.ndarray):
        pred_box = common_utils.torch_tensor_to_numpy(pred_box)
    if gt_box is not None and not isinstance(gt_box, np.ndarray):
        gt_box = common_utils.torch_tensor_to_numpy(gt_box)

    ratio = dataset.params["preprocess"]["args"]["res"]
    L1, W1, H1, L2, W2, H2 = dataset.params["preprocess"]["cav_lidar_range"]
    bev_origin = np.array([L1, W1]).reshape(1, -1)
    # (img_row, img_col)
    bev_map = dataset.project_points_to_bev_map(pcd, ratio)
    # (img_row, img_col, 3)
    bev_map = \
        np.repeat(bev_map[:, :, np.newaxis], 3, axis=-1).astype(np.float32)
    bev_map = bev_map * 255

    if pred_box is not None:
        num_bbx = pred_box.shape[0]
        for i in range(num_bbx):
            bbx = pred_box[i]

            bbx = ((bbx - bev_origin) / ratio).astype(int)
            bbx = bbx[:, ::-1]
            cv2.polylines(bev_map, [bbx], True, (0, 0, 255), 1)

    if gt_box is not None and len(gt_box):
        for i in range(gt_box.shape[0]):
            bbx = gt_box[i][:4, :2]
            bbx = (((bbx - bev_origin)) / ratio).astype(int)
            bbx = bbx[:, ::-1]
            cv2.polylines(bev_map, [bbx], True, (255, 0, 0), 1)

    if show_vis:
        plt.axis("off")
        plt.imshow(bev_map)
        plt.show()
    if save_path:
        plt.axis("off")
        plt.imshow(bev_map)
        plt.savefig(save_path)


def visualize_single_sample_dataloader(batch_data,
                                       o3d_pcd,
                                       order,
                                       key='origin_lidar',
                                       visualize=False,
                                       save_path='',
                                       oabb=False,
                                       mode='constant'):
    """
    Visualize a single frame of a single CAV for validation of data pipeline.
    """

    # ---------- 1. Get point cloud from batch ----------
    origin_lidar = batch_data[key]
    if not isinstance(origin_lidar, np.ndarray):
        origin_lidar = common_utils.torch_tensor_to_numpy(origin_lidar)

    # we only visualize the first cav for single sample
    if len(origin_lidar.shape) > 2:
        origin_lidar = origin_lidar[0]

    # ---------- 2. Apply point cloud offset ----------
    # offset in meters: (dx, dy, dz)
    offset = np.array([20.0, 0.0, 0.0], dtype=np.float32)  # 20 m in x
    print("PC first point BEFORE offset:", origin_lidar[0, :3])
    origin_lidar[:, 0] += offset[0]
    origin_lidar[:, 1] += offset[1]
    origin_lidar[:, 2] += offset[2]
    print("PC first point AFTER  offset:", origin_lidar[0, :3])

    # ---------- 3. Color encoding ----------
    origin_lidar_intcolor = \
        color_encoding(origin_lidar[:, -1] if mode == 'intensity'
                       else origin_lidar[:, 2], mode=mode)

    # ---------- 4. Coordinate system flip (left -> right hand) ----------
    origin_lidar[:, :1] = -origin_lidar[:, :1]

    # ---------- 5. Fill Open3D point cloud ----------
    o3d_pcd.points = o3d.utility.Vector3dVector(origin_lidar[:, :3])
    o3d_pcd.colors = o3d.utility.Vector3dVector(origin_lidar_intcolor)

    # ---------- 6. Bounding boxes ----------
    object_bbx_center = batch_data['object_bbx_center']
    object_bbx_mask = batch_data['object_bbx_mask']
    object_bbx_center = object_bbx_center[object_bbx_mask == 1]

    aabbs = bbx2linset(object_bbx_center, order) if not oabb else \
        bbx2oabb(object_bbx_center, order)

    visualize_elements = [o3d_pcd] + aabbs

    # ---------- 7. Optional visualization ----------
    if visualize:
        o3d.visualization.draw_geometries(visualize_elements)

    if save_path:
        save_o3d_visualization(visualize_elements, save_path)

    return o3d_pcd, aabbs



def visualize_inference_sample_dataloader(pred_box_tensor,
                                          gt_box_tensor,
                                          origin_lidar,
                                          o3d_pcd,
                                          mode='constant'):
    """
    Visualize a frame during inference for video stream.

    Parameters
    ----------
    pred_box_tensor : torch.Tensor
        (N, 8, 3) prediction.

    gt_box_tensor : torch.Tensor
        (N, 8, 3) groundtruth bbx

    origin_lidar : torch.Tensor
        PointCloud, (N, 4).

    o3d_pcd : open3d.PointCloud
        Used to visualize the pcd.

    mode : str
        lidar point rendering mode.
    """

    if not isinstance(origin_lidar, np.ndarray):
        origin_lidar = common_utils.torch_tensor_to_numpy(origin_lidar)
    # we only visualize the first cav for single sample
    if len(origin_lidar.shape) > 2:
        origin_lidar = origin_lidar[0]
    # this is for 2-stage origin lidar, it has different format
    if origin_lidar.shape[1] > 4:
        origin_lidar = origin_lidar[:, 1:]

    origin_lidar_intcolor = \
        color_encoding(origin_lidar[:, -1] if mode == 'intensity'
                       else origin_lidar[:, 2], mode=mode)

    if not isinstance(pred_box_tensor, np.ndarray):
        pred_box_tensor = common_utils.torch_tensor_to_numpy(pred_box_tensor)
    if not isinstance(gt_box_tensor, np.ndarray):
        gt_box_tensor = common_utils.torch_tensor_to_numpy(gt_box_tensor)

    # left -> right hand
    origin_lidar[:, :1] = -origin_lidar[:, :1]                     #bbx[:, 1:2] = -bbx[:, 1:2] # flip y axis
                                                                   #bbx[:, 2:3] = -bbx[:, 2:3] # flip z axis

    o3d_pcd.points = o3d.utility.Vector3dVector(origin_lidar[:, :3])
    o3d_pcd.colors = o3d.utility.Vector3dVector(origin_lidar_intcolor)

    gt_o3d_box = bbx2linset(gt_box_tensor, order='hwl', color=(0, 1, 0))
    pred_o3d_box = bbx2linset(pred_box_tensor, color=(1, 0, 0))

    return o3d_pcd, pred_o3d_box, gt_o3d_box


def visualize_sequence_dataloader(dataloader, order, color_mode='z-value'):
    """
    Visualize the batch data in animation.
    """
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    opt = vis.get_render_option()
    opt.background_color = [0.05, 0.05, 0.05]
    opt.point_size = 1.0
    opt.show_coordinate_frame = True

    # dynamic containers (no fixed 50)
    vis_pcd = o3d.geometry.PointCloud()
    vis_aabbs = []

    while True:
        for i_batch, sample_batched in enumerate(dataloader):
            print(i_batch)

            # build current frame's pcd + AABBs
            pcd, aabbs = visualize_single_sample_dataloader(
                sample_batched['ego'],
                vis_pcd,
                order,
                mode=color_mode
            )

            # --- resize vis_aabbs to match current count ---
            cur_n = len(aabbs)
            if len(vis_aabbs) < cur_n:
                vis_aabbs += [o3d.geometry.LineSet() for _ in range(cur_n - len(vis_aabbs))]
            elif len(vis_aabbs) > cur_n:
                # remove extras from the scene first
                for j in range(len(vis_aabbs) - 1, cur_n - 1, -1):
                    try:
                        vis.remove_geometry(vis_aabbs[j], reset_bounding_box=False)
                    except Exception:
                        pass
                vis_aabbs = vis_aabbs[:cur_n]

            # --- first batch: add geometries once ---
            if i_batch == 0:
                vis.add_geometry(pcd)
                for ls in vis_aabbs:
                    vis.add_geometry(ls)

            # --- update per-box (use i, not 'index') ---
            for i in range(cur_n):
                vis_aabbs[i] = lineset_assign(vis_aabbs[i], aabbs[i])
                vis.update_geometry(vis_aabbs[i])

            # --- update point cloud and render ---
            vis.update_geometry(pcd)
            vis.poll_events()
            vis.update_renderer()
            time.sleep(0.001)

    vis.destroy_window()


def save_o3d_visualization(element, save_path):
    """
    Save the open3d drawing to folder.

    Parameters
    ----------
    element : list
        List of o3d.geometry objects.

    save_path : str
        The save path.
    """
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    for i in range(len(element)):
        vis.add_geometry(element[i])
        vis.update_geometry(element[i])

    vis.poll_events()
    vis.update_renderer()

    vis.capture_screen_image(save_path)
    vis.destroy_window()


def visualize_bev(batch_data):
    bev_input = batch_data["processed_lidar"]["bev_input"]
    label_map = batch_data["label_dict"]["label_map"]
    if not isinstance(bev_input, np.ndarray):
        bev_input = common_utils.torch_tensor_to_numpy(bev_input)

    if not isinstance(label_map, np.ndarray):
        label_map = label_map[0].numpy() if not label_map[0].is_cuda else \
            label_map[0].cpu().detach().numpy()

    if len(bev_input.shape) > 3:
        bev_input = bev_input[0, ...]

    plt.matshow(np.sum(bev_input, axis=0))
    plt.axis("off")
    plt.matshow(label_map[0, :, :])
    plt.axis("off")
    plt.show()


def draw_box_plt(boxes_dec, ax, color=None, linewidth_scale=1.0):
    """
    draw boxes in a given plt ax
    :param boxes_dec: (N, 5) or (N, 7) in metric
    :param ax:
    :return: ax with drawn boxes
    """
    if not len(boxes_dec)>0:
        return ax
    boxes_np= boxes_dec
    if not isinstance(boxes_np, np.ndarray):
        boxes_np = boxes_np.cpu().detach().numpy()
    if boxes_np.shape[-1]>5:
        boxes_np = boxes_np[:, [0, 1, 3, 4, 6]]
    x = boxes_np[:, 0]
    y = boxes_np[:, 1]
    dx = boxes_np[:, 2]
    dy = boxes_np[:, 3]

    x1 = x - dx / 2
    y1 = y - dy / 2
    x2 = x + dx / 2
    y2 = y + dy / 2
    theta = boxes_np[:, 4:5]
    # bl, fl, fr, br
    corners = np.array([[x1, y1],[x1,y2], [x2,y2], [x2, y1]]).transpose(2, 0, 1)
    new_x = (corners[:, :, 0] - x[:, None]) * np.cos(theta) + (corners[:, :, 1]
              - y[:, None]) * (-np.sin(theta)) + x[:, None]
    new_y = (corners[:, :, 0] - x[:, None]) * np.sin(theta) + (corners[:, :, 1]
              - y[:, None]) * (np.cos(theta)) + y[:, None]
    corners = np.stack([new_x, new_y], axis=2)
    for corner in corners:
        ax.plot(corner[[0,1,2,3,0], 0], corner[[0,1,2,3,0], 1], color=color, linewidth=0.5*linewidth_scale)
        # draw front line (
        ax.plot(corner[[2, 3], 0], corner[[2, 3], 1], color=color, linewidth=2*linewidth_scale)
    return ax


def draw_points_boxes_plt(pc_range, points=None, boxes_pred=None, boxes_gt=None, save_path=None,
                          points_c='y.', bbox_gt_c='green', bbox_pred_c='red', return_ax=False, ax=None):
    if ax is None:
        ax = plt.figure(figsize=(15, 6)).add_subplot(1, 1, 1)
        ax.set_aspect('equal', 'box')
        ax.set(xlim=(pc_range[0], pc_range[3]),
               ylim=(pc_range[1], pc_range[4]))
    if points is not None:
        ax.plot(points[:, 0], points[:, 1], points_c, markersize=0.1)
    if (boxes_gt is not None) and len(boxes_gt)>0:
        ax = draw_box_plt(boxes_gt, ax, color=bbox_gt_c)
    if (boxes_pred is not None) and len(boxes_pred)>0:
        ax = draw_box_plt(boxes_pred, ax, color=bbox_pred_c)
    plt.xlabel('x')
    plt.ylabel('y')

    plt.savefig(save_path)
    if return_ax:
        return ax
    plt.close()
