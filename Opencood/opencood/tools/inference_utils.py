# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib
"""
opencood/tools/inference_utils.py

This is a COMPLETE inference_utils.py that:

1) Preserves OpenCOOD's standard inference APIs:
   - inference_late_fusion(batch_data, model, dataset, return_per_cav=False)
   - inference_early_fusion(batch_data, model, dataset)
   - inference_intermediate_fusion(batch_data, model, dataset, return_per_cav=False)
   - save_prediction_gt(pred, gt, pcd, timestamp, save_path)

2) Adds robust helpers to save per-CAV raw pointclouds and per-CAV GT:
   - save_pcd_per_cav(batch_data, timestamp, save_root)
   - save_gt_per_cav(gt_box_tensor, batch_data, timestamp, save_root)

3) Provides return_per_cav for late fusion:
   - If dataset.post_process supports return_per_cav=True -> uses it
   - Otherwise -> falls back to single-agent post_process per cav

4) For intermediate fusion:
   - If dataset.post_process supports return_per_cav=True -> returns per_cav
   - Otherwise returns per_cav=None (correct; does not hallucinate)

5) Works with two batch_data styles:
   A) explicit cav keys: batch_data = {'ego':..., 'cav_1':..., ...}
   B) packed style: batch_data = {'ego': {..., 'origin_lidar': [ego, cav1, cav2, ...]}}

NOTE:
- save_gt_per_cav transforms ego-frame GT to each cav frame if it can find T_cav_to_ego.
- It supports GT as:
    (N,7)   [x,y,z,l,w,h,yaw]
    (N,K,3) points/corners (includes (N,8,3))
"""

import os
import re
import math
from collections import OrderedDict
from typing import Any, Dict, Optional, Tuple

import torch.nn.functional as F
import numpy as np
import torch
import open3d as o3d

from opencood.utils.common_utils import torch_tensor_to_numpy


# ==========================================================
# Inference (OpenCOOD standard)
# ==========================================================

def inference_late_fusion(batch_data, model, dataset, return_per_cav: bool = False):
    """
    Model inference for late fusion.

    Default:
      output_dict[cav_key] = model(cav_content)
      dataset.post_process(batch_data, output_dict)

    If return_per_cav=True:
      - Try dataset.post_process(..., return_per_cav=True)
      - Else fallback: run single-agent post_process per cav.
    """
    output_dict = OrderedDict()
    for cav_key, cav_content in batch_data.items():
        if not isinstance(cav_content, dict):
            continue
        output_dict[cav_key] = model(cav_content)

    if not return_per_cav:
        pred_box_tensor, pred_score, gt_box_tensor = dataset.post_process(batch_data, output_dict)
        return pred_box_tensor, pred_score, gt_box_tensor

    # ---- Try native per-cav support in dataset.post_process ----
    try:
        ret = dataset.post_process(batch_data, output_dict, return_per_cav=True)
        if isinstance(ret, (list, tuple)) and len(ret) == 4:
            pred_box_tensor, pred_score, gt_box_tensor, per_cav = ret
            return pred_box_tensor, pred_score, gt_box_tensor, per_cav
        if isinstance(ret, dict):
            pred_box_tensor = ret.get("pred_box_tensor", None)
            pred_score = ret.get("pred_score", None)
            gt_box_tensor = ret.get("gt_box_tensor", None)
            per_cav = ret.get("per_cav", None)
            return pred_box_tensor, pred_score, gt_box_tensor, per_cav
    except TypeError:
        # dataset.post_process doesn't accept return_per_cav
        pass

    # ---- Fallback: compute per-cav by single-agent post_process per cav ----
    pred_box_tensor, pred_score, gt_box_tensor = dataset.post_process(batch_data, output_dict)

    per_cav: Dict[str, Dict[str, torch.Tensor]] = {}

    # determine device
    try:
        device = next(model.parameters()).device
    except Exception:
        device = torch.device("cpu")

    for cav_key, cav_content in batch_data.items():
        if not isinstance(cav_content, dict):
            continue

        # Try post_process with cav_key as-is
        b, s = None, None
        try:
            agent_batch = {str(cav_key): cav_content}
            agent_out = OrderedDict()
            agent_out[str(cav_key)] = model(cav_content)
            b, s, _ = dataset.post_process(agent_batch, agent_out)
        except Exception:
            # Some datasets insist on 'ego' key
            try:
                agent_batch2 = {"ego": cav_content}
                agent_out2 = OrderedDict()
                agent_out2["ego"] = model(cav_content)
                b, s, _ = dataset.post_process(agent_batch2, agent_out2)
            except Exception:
                b, s = None, None

        if b is None:
            b = torch.zeros((0, 8, 3), device=device)
        if s is None:
            s = torch.zeros((0,), device=device)

        per_cav[str(cav_key)] = {"boxes": b, "scores": s}

    return pred_box_tensor, pred_score, gt_box_tensor, per_cav


def inference_early_fusion(batch_data, model, dataset):
    """
    Model inference for early fusion.
    """
    output_dict = OrderedDict()
    cav_content = batch_data['ego']
    output_dict['ego'] = model(cav_content)

    pred_box_tensor, pred_score, gt_box_tensor = dataset.post_process(batch_data, output_dict)
    return pred_box_tensor, pred_score, gt_box_tensor


def inference_intermediate_fusion(batch_data, model, dataset, return_per_cav: bool = False):
    """
    Model inference for intermediate fusion.
    In many OpenCOOD versions, this mirrors early fusion at inference time.

    If return_per_cav=True:
      - returns per_cav only if dataset.post_process supports return_per_cav=True
      - otherwise returns per_cav=None
    """
    output_dict = OrderedDict()
    cav_content = batch_data['ego']
    output_dict['ego'] = model(cav_content)

    if not return_per_cav:
        pred_box_tensor, pred_score, gt_box_tensor = dataset.post_process(batch_data, output_dict)
        return pred_box_tensor, pred_score, gt_box_tensor

    try:
        ret = dataset.post_process(batch_data, output_dict, return_per_cav=True)
        if isinstance(ret, (list, tuple)) and len(ret) == 4:
            return ret
        if isinstance(ret, dict):
            return (ret.get("pred_box_tensor", None),
                    ret.get("pred_score", None),
                    ret.get("gt_box_tensor", None),
                    ret.get("per_cav", None))
    except TypeError:
        pass

    pred_box_tensor, pred_score, gt_box_tensor = dataset.post_process(batch_data, output_dict)
    return pred_box_tensor, pred_score, gt_box_tensor, None


def save_prediction_gt(pred_tensor, gt_tensor, pcd, timestamp, save_path):
    """
    Save prediction and gt tensor to npy files.

    Produces:
      %04d_pcd.npy
      %04d_pred.npy
      %04d_gt.npy
    """
    os.makedirs(save_path, exist_ok=True)

    pred_np = torch_tensor_to_numpy(pred_tensor) if pred_tensor is not None else np.zeros((0, 8, 3), np.float32)
    gt_np = torch_tensor_to_numpy(gt_tensor) if gt_tensor is not None else np.zeros((0, 8, 3), np.float32)
    pcd_np = torch_tensor_to_numpy(pcd)

    np.save(os.path.join(save_path, '%04d_pcd.npy' % timestamp), pcd_np)
    np.save(os.path.join(save_path, '%04d_pred.npy' % timestamp), pred_np)
    np.save(os.path.join(save_path, '%04d_gt.npy' % timestamp), gt_np)


# ==========================================================
# Helpers
# ==========================================================

def _sanitize_cav_filename(cav_id: str) -> str:
    """Make stable filenames."""
    s = str(cav_id)
    if s == "ego":
        return "ego"
    if re.fullmatch(r"\d+", s):
        return f"cav_{s}"
    m = re.fullmatch(r"cav(\d+)", s)
    if m:
        return f"cav_{m.group(1)}"
    return s


def _to_numpy(x):
    if x is None:
        return None
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _pick_aligned_id_list(ego_dict: dict, n: int):
    """
    Try to find a list of cav ids aligned with origin_lidar list length n.
    If not found, fallback: ['ego','cav_1','cav_2',...]
    """
    for k in ["cav_id_list", "cav_ids", "vehicle_ids", "agent_ids", "cav_id"]:
        if k in ego_dict and isinstance(ego_dict[k], (list, tuple)) and len(ego_dict[k]) == n:
            return list(ego_dict[k])

    for k in ["cav_id_list", "cav_ids", "vehicle_ids", "agent_ids"]:
        if k in ego_dict and torch.is_tensor(ego_dict[k]):
            arr = ego_dict[k].detach().cpu().numpy().tolist()
            if isinstance(arr, list) and len(arr) == n:
                return arr

    return ["ego"] + [f"cav_{i}" for i in range(1, n)]


def _get_T_list_packed(ego_dict: dict, n: int):
    """
    For packed format, try to find a list of 4x4 transforms aligned with cav list.
    Interpreted as T_cav_to_ego (cav -> ego).
    """
    for k in ["transformation_matrix", "cav2ego", "trans_mat", "T_cav_to_ego"]:
        if k in ego_dict and isinstance(ego_dict[k], (list, tuple)) and len(ego_dict[k]) == n:
            Ts = []
            ok = True
            for item in ego_dict[k]:
                T = _to_numpy(item)
                T = np.asarray(T)
                if T.shape != (4, 4):
                    ok = False
                    break
                Ts.append(T)
            if ok:
                return Ts
    return None


def _get_T_cav_to_ego_from_cavdict(cav_content: dict):
    """
    For explicit cav-key format, try to read a 4x4 matrix mapping cav -> ego.
    """
    for k in ["transformation_matrix", "cav2ego", "trans_mat", "T", "T_cav_to_ego"]:
        if k in cav_content:
            T = _to_numpy(cav_content[k])
            T = np.asarray(T)
            if T.shape == (4, 4):
                return T

    for k in ["transformation_matrix", "cav2ego", "trans_mat", "T_cav_to_ego"]:
        if k in cav_content and isinstance(cav_content[k], (list, tuple)) and len(cav_content[k]) > 0:
            T = _to_numpy(cav_content[k][0])
            T = np.asarray(T)
            if T.shape == (4, 4):
                return T

    return None


def _wrap_to_pi(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def transform_boxes_7d_ego_to_cav(boxes_ego: np.ndarray, T_cav_to_ego: np.ndarray) -> np.ndarray:
    """
    Transform (N,7) [x,y,z,l,w,h,yaw] from ego frame -> cav frame.
    """
    boxes_ego = np.asarray(boxes_ego, dtype=np.float64)
    if boxes_ego.size == 0:
        return boxes_ego.reshape(-1, 7)
    if boxes_ego.ndim == 1:
        boxes_ego = boxes_ego.reshape(1, -1)

    boxes_ego = boxes_ego[:, :7]
    T_ego_to_cav = np.linalg.inv(T_cav_to_ego)

    xyz1 = np.ones((boxes_ego.shape[0], 4), dtype=np.float64)
    xyz1[:, 0:3] = boxes_ego[:, 0:3]
    xyz_cav = (T_ego_to_cav @ xyz1.T).T[:, 0:3]

    R = T_ego_to_cav[0:3, 0:3]
    delta_yaw = math.atan2(R[1, 0], R[0, 0])

    out = boxes_ego.copy()
    out[:, 0:3] = xyz_cav
    out[:, 6] = np.vectorize(_wrap_to_pi)(out[:, 6] + delta_yaw)
    return out


def transform_points_ego_to_cav(points_ego: np.ndarray, T_cav_to_ego: np.ndarray) -> np.ndarray:
    """
    Transform points from ego -> cav.

    Supports:
      (N,3)
      (B,K,3)  (e.g., corners/points per object)
    """
    pts = np.asarray(points_ego, dtype=np.float64)
    T_ego_to_cav = np.linalg.inv(T_cav_to_ego)

    if pts.ndim == 2 and pts.shape[1] == 3:
        ones = np.ones((pts.shape[0], 1), dtype=np.float64)
        pts1 = np.concatenate([pts, ones], axis=1)  # (N,4)
        out = (T_ego_to_cav @ pts1.T).T[:, :3]
        return out

    if pts.ndim == 3 and pts.shape[-1] == 3:
        B, K, _ = pts.shape
        flat = pts.reshape(-1, 3)
        ones = np.ones((flat.shape[0], 1), dtype=np.float64)
        flat1 = np.concatenate([flat, ones], axis=1)  # (B*K,4)
        out_flat = (T_ego_to_cav @ flat1.T).T[:, :3]
        return out_flat.reshape(B, K, 3)

    return pts


# ==========================================================
# ✅ Save per-CAV PCD (packed OR explicit format)
# ==========================================================

def save_pcd_per_cav(batch_data, timestamp: int, save_root: str):
    """
    Creates:
      save_root/frame_00000/<cav>.pcd
    """
    frame_dir = os.path.join(save_root, f"frame_{timestamp:05d}")
    os.makedirs(frame_dir, exist_ok=True)

    # ---- packed style (only ego key) ----
    if isinstance(batch_data, dict) and ("ego" in batch_data) and (len(batch_data.keys()) == 1):
        ego = batch_data["ego"]
        lidar_list = ego.get("origin_lidar", None)

        if isinstance(lidar_list, (list, tuple)) and len(lidar_list) > 0:
            cav_ids = _pick_aligned_id_list(ego, len(lidar_list))

            for cav_id, lidar in zip(cav_ids, lidar_list):
                lidar_np = _to_numpy(lidar)
                if lidar_np is None or lidar_np.ndim != 2 or lidar_np.shape[1] < 3:
                    continue

                xyz = lidar_np[:, :3].astype(np.float64)
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(xyz)

                cav_name = _sanitize_cav_filename(cav_id)
                out_path = os.path.join(frame_dir, f"{cav_name}.pcd")
                o3d.io.write_point_cloud(out_path, pcd, write_ascii=False)

            return

    # ---- explicit cav keys ----
    for cav_id, cav_content in batch_data.items():
        if not isinstance(cav_content, dict):
            continue
        if "origin_lidar" not in cav_content:
            continue

        origin = cav_content["origin_lidar"]
        lidar = origin[0] if isinstance(origin, (list, tuple)) else origin
        lidar_np = _to_numpy(lidar)
        if lidar_np is None or lidar_np.ndim != 2 or lidar_np.shape[1] < 3:
            continue

        xyz = lidar_np[:, :3].astype(np.float64)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)

        cav_name = _sanitize_cav_filename(cav_id)
        out_path = os.path.join(frame_dir, f"{cav_name}.pcd")
        o3d.io.write_point_cloud(out_path, pcd, write_ascii=False)


# ==========================================================
# ✅ Save per-CAV GT (packed OR explicit format)
# ==========================================================

def save_gt_per_cav(gt_box_tensor, batch_data, timestamp: int, save_root: str):
    """
    Creates:
      save_root/frame_00000/gt_<cav>.npy
    Also saves:
      save_root/frame_00000/gt_ego.npy
    """
    frame_dir = os.path.join(save_root, f"frame_{timestamp:05d}")
    os.makedirs(frame_dir, exist_ok=True)

    if gt_box_tensor is None:
        return

    gt_ego = _to_numpy(gt_box_tensor)
    np.save(os.path.join(frame_dir, "gt_ego.npy"), gt_ego)

    is_7d = (gt_ego.ndim == 2 and gt_ego.shape[1] >= 7)
    is_points = (gt_ego.ndim == 3 and gt_ego.shape[-1] == 3)

    # ---- packed style (only ego key) ----
    if isinstance(batch_data, dict) and ("ego" in batch_data) and (len(batch_data.keys()) == 1):
        ego = batch_data["ego"]

        lidar_list = ego.get("origin_lidar", None)
        n = len(lidar_list) if isinstance(lidar_list, (list, tuple)) else 1
        cav_ids = _pick_aligned_id_list(ego, n)

        T_list = _get_T_list_packed(ego, n)

        for idx, cav_id in enumerate(cav_ids):
            cav_name = _sanitize_cav_filename(cav_id)
            out_path = os.path.join(frame_dir, f"gt_{cav_name}.npy")

            T_cav_to_ego = None
            if T_list is not None and idx < len(T_list):
                T_cav_to_ego = T_list[idx]

            if T_cav_to_ego is None:
                np.save(out_path, gt_ego)
                continue

            if is_7d:
                gt_cav = transform_boxes_7d_ego_to_cav(gt_ego[:, :7], T_cav_to_ego)
                np.save(out_path, gt_cav)
            elif is_points:
                gt_cav_pts = transform_points_ego_to_cav(gt_ego, T_cav_to_ego)
                np.save(out_path, gt_cav_pts)
            else:
                np.save(out_path, gt_ego)

        return

    # ---- explicit cav keys ----
    for cav_id, cav_content in batch_data.items():
        if not isinstance(cav_content, dict):
            continue

        cav_name = _sanitize_cav_filename(cav_id)
        out_path = os.path.join(frame_dir, f"gt_{cav_name}.npy")

        T_cav_to_ego = _get_T_cav_to_ego_from_cavdict(cav_content)
        if T_cav_to_ego is None:
            np.save(out_path, gt_ego)
            continue

        if is_7d:
            gt_cav = transform_boxes_7d_ego_to_cav(gt_ego[:, :7], T_cav_to_ego)
            np.save(out_path, gt_cav)
        elif is_points:
            gt_cav_pts = transform_points_ego_to_cav(gt_ego, T_cav_to_ego)
            np.save(out_path, gt_cav_pts)
        else:
            np.save(out_path, gt_ego)

def _infer_bev_params_from_batch(batch_data, feat_h, feat_w):
    """
    Infer BEV meters-per-cell in x/y using lidar_range + voxel_size if available.
    Falls back to 1.0m if missing (still produces a consistent relative metric).

    Returns:
      x_min, y_min, x_max, y_max, meters_per_cell_x, meters_per_cell_y
    """
    ego = batch_data.get("ego", {})
    lidar_range = ego.get("lidar_range", None)  # sometimes present
    voxel_size = ego.get("voxel_size", None)    # sometimes present

    # Most OpenCOOD configs store lidar_range in hypes, not in batch_data.
    # If not present here, we still can compute misalignment in feature space (unitless).
    if lidar_range is None or voxel_size is None:
        # fallback: treat feature grid as abstract; scale=1
        return 0.0, 0.0, float(feat_w), float(feat_h), 1.0, 1.0

    # lidar_range is usually [x_min, y_min, z_min, x_max, y_max, z_max]
    lr = lidar_range
    x_min, y_min, x_max, y_max = float(lr[0]), float(lr[1]), float(lr[3]), float(lr[4])

    # Feature map is downsampled vs voxel grid.
    # Compute meters per cell directly from the covered physical range.
    meters_per_cell_x = (x_max - x_min) / float(feat_w)
    meters_per_cell_y = (y_max - y_min) / float(feat_h)

    return x_min, y_min, x_max, y_max, meters_per_cell_x, meters_per_cell_y


def _warp_bev_feature_cav_to_ego(feat_cav, T_cav_to_ego, x_min, y_min, mpc_x, mpc_y):
    """
    Warp a BEV feature map feat_cav [C,H,W] into ego frame using 4x4 transform.

    We treat BEV as an (x,y) plane:
      - W axis corresponds to x (forward)
      - H axis corresponds to y (left/right)  (typical in OpenCOOD)
    """
    assert feat_cav.dim() == 3, f"feat_cav must be [C,H,W], got {feat_cav.shape}"
    C, H, W = feat_cav.shape
    device = feat_cav.device
    dtype = feat_cav.dtype

    if not isinstance(T_cav_to_ego, torch.Tensor):
        T = torch.as_tensor(T_cav_to_ego, device=device, dtype=dtype)
    else:
        T = T_cav_to_ego.to(device=device, dtype=dtype)

    # Build a sampling grid in ego BEV, then map to cav BEV using inverse transform.
    # grid_sample expects normalized coords in [-1,1] with last dim (x,y) order.
    ys = torch.arange(H, device=device, dtype=dtype)
    xs = torch.arange(W, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")  # [H,W]

    # Convert ego pixel -> ego meters
    x_ego = x_min + (xx + 0.5) * mpc_x
    y_ego = y_min + (yy + 0.5) * mpc_y

    ones = torch.ones_like(x_ego)
    zeros = torch.zeros_like(x_ego)

    # ego points in homogeneous (x,y,0,1)
    pts_ego = torch.stack([x_ego, y_ego, zeros, ones], dim=-1)  # [H,W,4]
    pts_ego_flat = pts_ego.view(-1, 4).t()  # [4, H*W]

    # Map ego -> cav to sample from cav feature map
    T_ego_to_cav = torch.linalg.inv(T)
    pts_cav = (T_ego_to_cav @ pts_ego_flat).t()[:, :2]  # [H*W,2] x,y in cav meters

    x_cav = pts_cav[:, 0].view(H, W)
    y_cav = pts_cav[:, 1].view(H, W)

    # Convert cav meters -> cav pixel coords
    # Assume cav BEV grid covers same x_min,y_min extents (common in OpenCOOD).
    # If your setup differs, use per-agent ranges.
    u = (x_cav - x_min) / mpc_x - 0.5  # pixel x
    v = (y_cav - y_min) / mpc_y - 0.5  # pixel y

    # Normalize to [-1, 1]
    u_norm = (u / (W - 1)) * 2 - 1
    v_norm = (v / (H - 1)) * 2 - 1
    grid = torch.stack([u_norm, v_norm], dim=-1).unsqueeze(0)  # [1,H,W,2]

    feat_in = feat_cav.unsqueeze(0)  # [1,C,H,W]
    feat_warp = F.grid_sample(
        feat_in, grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True
    )
    return feat_warp.squeeze(0)  # [C,H,W]

def compute_feature_misalignment_intermediate(batch_data, spatial_features_2d):
    """
    Robust to multiple layouts of spatial_features_2d.

    Expected goal:
      get agent_feats as [L, C, H, W] where L = record_len[0]

    If the model only outputs ego feature ([1,C,H,W]) while L>1:
      return (None, {}) so caller skips logging.
    """
    ego = batch_data.get("ego", {})
    pairwise = ego.get("pairwise_t_matrix", None)
    record_len = ego.get("record_len", None)

    if pairwise is None or record_len is None:
        return None, {}

    # record_len is usually tensor([L]) for batch_size=1
    if isinstance(record_len, torch.Tensor):
        L = int(record_len[0].item())
    elif isinstance(record_len, (list, tuple)):
        L = int(record_len[0])
    else:
        L = int(record_len)

    feats = spatial_features_2d
    if not isinstance(feats, torch.Tensor):
        return None, {}

    # -----------------------------
    # Normalize feature layout
    # -----------------------------
    # Case 1) feats is [B, L, C, H, W]
    if feats.ndim == 5:
        # assume batch_size=1 for your inference
        if feats.shape[0] != 1:
            return None, {}
        if feats.shape[1] < L:
            return None, {}
        feats = feats[0, :L]   # -> [L,C,H,W]

    # Case 2) feats is [L, C, H, W] or [sumL, C, H, W]
    elif feats.ndim == 4:
        if feats.shape[0] >= L:
            feats = feats[:L]  # -> [L,C,H,W]
        else:
            # This is the error you’re seeing:
            # feats.shape[0] == 1 but L > 1
            return None, {}

    else:
        return None, {}

    # Now feats must be [L,C,H,W]
    if feats.shape[0] != L:
        return None, {}

    F_ego = feats[0]  # [C,H,W]
    C, H, W = F_ego.shape

    x_min, y_min, x_max, y_max, mpc_x, mpc_y = _infer_bev_params_from_batch(batch_data, H, W)

    per_cav = {}
    vals = []

    for cav_idx in range(1, L):
        F_cav = feats[cav_idx]  # [C,H,W]

        # cav -> ego transform:
        # you stated: pairwise[0,0,cav] is ego->cav
        # therefore cav->ego is pairwise[0,cav,0]
        T_cav_to_ego = pairwise[0, cav_idx, 0]

        F_warp = _warp_bev_feature_cav_to_ego(
            F_cav, T_cav_to_ego,
            x_min, y_min, mpc_x, mpc_y
        )

        diff = F_warp - F_ego  # [C,H,W]
        l2_map = torch.norm(diff, dim=0)  # [H,W]
        mis = float(l2_map.mean().item())

        per_cav[cav_idx] = mis
        vals.append(mis)

    frame_mean = float(sum(vals) / len(vals)) if len(vals) > 0 else 0.0
    return frame_mean, per_cav
