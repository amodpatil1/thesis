# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>, Hao Xiang <haxiang@g.ucla.edu>, Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib

"""
opencood/tools/inference.py

This version adds (LATE only):
- Pose proxy from GT-vs-Pred boxes (NO pose estimator needed):
    pose_proxy_from_boxes.csv
    pose_proxy_jitter.csv

It estimates a best-fit SE(2) transform that aligns predicted box centers to GT box centers
(using matched boxes by BEV IoU). This gives a "pose-misalignment proxy" (theta, tx, ty)
and its temporal jitter over frames.

Keeps your existing features:
- Save raw per-CAV point clouds as .pcd (late only) if --save_raw_pcd
- Save per-CAV IoU per frame (late only) if --save_per_cav_iou
- Pose inconsistency (temporal jitter + cycle) based on transforms if --save_pose_inconsistency
- Save npy pred/gt corners if --save_npy
"""

print("executing", flush=True)

import argparse
import csv
import os
from collections import defaultdict
import math
import resource

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.data_utils.datasets import build_dataset
from opencood.tools import inference_utils, train_utils
from opencood.utils import common_utils, eval_utils

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: F401

# Increase max open files (must run before DataLoader workers start)
soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, hard))
mp.set_sharing_strategy("file_system")


# ============================================================
# RAW PCD SAVING (LATE fusion per-agent dict)
# ============================================================
def _as_numpy_xyz(lidar_any):
    x = lidar_any
    if x is None:
        return None

    if isinstance(x, (list, tuple)):
        if len(x) == 0:
            return None
        if len(x) == 1:
            x = x[0]

    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    else:
        x = np.asarray(x)

    if x.ndim == 1 and x.dtype == object and len(x) == 1:
        x = x[0]
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        else:
            x = np.asarray(x)

    if x.ndim == 3 and x.shape[0] == 1:
        x = x[0]

    if x.ndim != 2 or x.shape[0] == 0 or x.shape[1] < 3:
        return None

    return x[:, :3].astype(np.float64, copy=False)


def _find_lidar_key(cav_content: dict):
    for k in ["origin_lidar", "raw_lidar", "lidar_np", "lidar", "processed_lidar"]:
        if k in cav_content:
            return k
    return None


def save_raw_pcd_late_like_intermediate(batch_data, scene_id, sample_idx, out_root):
    import open3d as o3d

    for cav_key, cav_content in batch_data.items():
        if not isinstance(cav_content, dict):
            continue

        lk = _find_lidar_key(cav_content)
        if lk is None:
            continue

        xyz = _as_numpy_xyz(cav_content.get(lk))
        if xyz is None or xyz.shape[0] == 0:
            continue

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)

        cav_dir = os.path.join(out_root, str(scene_id), str(cav_key))
        os.makedirs(cav_dir, exist_ok=True)

        save_path = os.path.join(cav_dir, f"{sample_idx:06d}.pcd")
        o3d.io.write_point_cloud(save_path, pcd)


# ============================================================
# Scene id
# ============================================================
def extract_scene_id(batch_data, batch_idx, dataset=None):
    ego = batch_data.get('ego', {})
    if isinstance(ego, dict):
        for k in ['lidar_path', 'pcd_path', 'yaml_path', 'file_path', 'filename', 'path']:
            if k in ego:
                path_val = ego[k]
                if isinstance(path_val, (list, tuple)) and len(path_val) > 0:
                    path_val = path_val[0]
                if isinstance(path_val, str):
                    scene_dir = os.path.dirname(path_val)
                    scene_root = os.path.dirname(scene_dir)
                    return os.path.basename(scene_root)

    chunk = batch_idx // 100
    return f"chunk_{chunk:04d}"


# ============================================================
# IoU helpers (your mean IoU diagnostic)
# ============================================================
def mean_det_to_gt_iou(det_boxes_tensor, gt_boxes_tensor):
    if det_boxes_tensor is None or gt_boxes_tensor is None:
        return None, 0, 0

    num_pred = int(det_boxes_tensor.shape[0]) if hasattr(det_boxes_tensor, "shape") else 0
    num_gt = int(gt_boxes_tensor.shape[0]) if hasattr(gt_boxes_tensor, "shape") else 0
    if num_pred == 0 or num_gt == 0:
        return None, num_pred, num_gt

    det_np = common_utils.torch_tensor_to_numpy(det_boxes_tensor)
    gt_np = common_utils.torch_tensor_to_numpy(gt_boxes_tensor)

    det_polys = list(common_utils.convert_format(det_np))
    gt_polys = list(common_utils.convert_format(gt_np))
    if len(det_polys) == 0 or len(gt_polys) == 0:
        return None, len(det_polys), len(gt_polys)

    ious = []
    for det_poly in det_polys:
        iou_arr = common_utils.compute_iou(det_poly, gt_polys)
        if len(iou_arr) > 0:
            ious.append(float(np.max(iou_arr)))

    if len(ious) == 0:
        return 0.0, len(det_polys), len(gt_polys)

    return float(np.mean(ious)), len(det_polys), len(gt_polys)


def compute_per_cav_iou_rows(per_cav_pred_dict, gt, scene_id, frame_id):
    rows = []
    for cav_key, cav_out in per_cav_pred_dict.items():
        if cav_out is None:
            continue

        if isinstance(cav_out, (list, tuple)) and len(cav_out) >= 1:
            pred = cav_out[0]
        elif isinstance(cav_out, dict):
            pred = cav_out.get("pred_box_tensor", None) or cav_out.get("pred", None)
        else:
            pred = cav_out

        mean_iou, num_pred, num_gt = mean_det_to_gt_iou(pred, gt)
        if mean_iou is None:
            mean_iou = float("nan")

        rows.append([str(scene_id), int(frame_id), str(cav_key),
                     float(mean_iou), int(num_pred), int(num_gt)])
    return rows


# ============================================================
# POSE INCONSISTENCY (LATE): temporal jitter + cycle consistency (transform-based)
# ============================================================
def _as_torch_4x4(x, device=None, dtype=torch.float32):
    if x is None:
        return None
    if isinstance(x, (list, tuple)) and len(x) == 1:
        x = x[0]
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x)
    x = x.to(device=device, dtype=dtype)
    if x.ndim == 3 and x.shape[0] == 1 and x.shape[1:] == (4, 4):
        x = x[0]
    if x.ndim != 2 or x.shape != (4, 4):
        return None
    return x


def _yaw_deg_from_R(R: torch.Tensor) -> float:
    yaw = torch.atan2(R[1, 0], R[0, 0])
    return float(yaw.item() * 180.0 / math.pi)


def _so3_angle_deg(R: torch.Tensor) -> float:
    tr = (R[0, 0] + R[1, 1] + R[2, 2])
    c = ((tr - 1.0) * 0.5).clamp(-1.0, 1.0)
    ang = torch.acos(c)
    return float(ang.item() * 180.0 / math.pi)


def _se3_delta(T_now: torch.Tensor, T_prev: torch.Tensor):
    """
    delta must be inv(prev) @ now (prev->now)
    """
    dT = torch.linalg.inv(T_prev) @ T_now
    t = dT[:3, 3]
    trans = float(torch.linalg.norm(t).item())
    R = dT[:3, :3]
    yaw = abs(_yaw_deg_from_R(R))
    so3 = _so3_angle_deg(R)
    return trans, yaw, so3


def compute_pose_temporal_jitter_rows(batch_data, scene_id, frame_id, prev_T_map, yaw_only=True):
    rows = []
    ego = batch_data.get("ego", {})
    device = ego.get("record_len", None).device if isinstance(ego.get("record_len", None), torch.Tensor) else None

    est_keys = ["transformation_matrix", "noisy_transformation_matrix", "T", "pose", "lidar_pose"]
    found_any = False
    for cav_key, cav_content in batch_data.items():
        if not isinstance(cav_content, dict):
            continue
        T_now = None
        for k in est_keys:
            if k in cav_content:
                T_now = _as_torch_4x4(cav_content.get(k), device=device)
                if T_now is not None:
                    break
        if T_now is None:
            continue

        found_any = True
        key = (str(scene_id), str(cav_key))
        T_prev = prev_T_map.get(key, None)
        prev_T_map[key] = T_now.detach()

        if T_prev is None:
            continue

        jt, jyaw, jso3 = _se3_delta(T_now, T_prev)
        jrot = jyaw if yaw_only else jso3
        rows.append([str(scene_id), int(frame_id), str(cav_key), float(jt), float(jrot)])

    if found_any:
        return rows

    pair = ego.get("pairwise_t_matrix", None)
    record_len = ego.get("record_len", None)
    if pair is None or record_len is None:
        return rows

    if isinstance(pair, torch.Tensor) and pair.ndim == 5:
        pair = pair[0]
    L = int(record_len[0].item()) if isinstance(record_len, torch.Tensor) else int(record_len[0])

    for cav_idx in range(L):
        T_now = _as_torch_4x4(pair[cav_idx, 0], device=device)
        if T_now is None:
            continue

        cav_key = "ego" if cav_idx == 0 else str(cav_idx)
        key = (str(scene_id), cav_key)
        T_prev = prev_T_map.get(key, None)
        prev_T_map[key] = T_now.detach()

        if T_prev is None:
            continue

        jt, jyaw, jso3 = _se3_delta(T_now, T_prev)
        jrot = jyaw if yaw_only else jso3
        rows.append([str(scene_id), int(frame_id), cav_key, float(jt), float(jrot)])

    return rows


def compute_pose_cycle_consistency_rows(batch_data, scene_id, frame_id, yaw_only=True, max_pairs=16):
    rows = []
    ego = batch_data.get("ego", {})
    pair = ego.get("pairwise_t_matrix", None)
    record_len = ego.get("record_len", None)

    if pair is None or record_len is None:
        return rows

    device = record_len.device if isinstance(record_len, torch.Tensor) else None

    if isinstance(pair, torch.Tensor) and pair.ndim == 5:
        pair = pair[0]
    L = int(record_len[0].item()) if isinstance(record_len, torch.Tensor) else int(record_len[0])

    pairs = []
    for ii in range(1, L):
        for jj in range(1, L):
            if ii != jj:
                pairs.append((ii, jj))
    if len(pairs) == 0:
        return rows
    pairs = pairs[:max_pairs]

    I = torch.eye(4, device=device, dtype=torch.float32)

    for (ii, jj) in pairs:
        T0i = _as_torch_4x4(pair[0, ii], device=device)
        Tij = _as_torch_4x4(pair[ii, jj], device=device)
        Tj0 = _as_torch_4x4(pair[jj, 0], device=device)
        if T0i is None or Tij is None or Tj0 is None:
            continue

        E = T0i @ Tij @ Tj0
        trans, yaw, so3 = _se3_delta(E, I)
        rot = yaw if yaw_only else so3
        rows.append([str(scene_id), int(frame_id), int(ii), int(jj), float(trans), float(rot)])

    return rows


# ============================================================
# POSE PROXY FROM BOXES (LATE): GT-vs-Pred -> best-fit SE(2)
# ============================================================
def _boxes_to_centers_xy(boxes_any):
    """boxes_any: (N,8,3) corners -> centers (N,2)"""
    if boxes_any is None:
        return np.zeros((0, 2), dtype=np.float32)
    if isinstance(boxes_any, torch.Tensor):
        b = boxes_any.detach().cpu().numpy()
    else:
        b = np.asarray(boxes_any)
    if b.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    c = b.mean(axis=1)[:, :2]
    return c.astype(np.float32)


def _corners8_to_bev_poly(corners8x3):
    z = corners8x3[:, 2]
    idx = np.argsort(z)[:4]
    pts = corners8x3[idx][:, :2]
    cent = pts.mean(axis=0)
    ang = np.arctan2(pts[:, 1] - cent[1], pts[:, 0] - cent[0])
    return pts[np.argsort(ang)]


def _poly_area(poly):
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _poly_intersection_area(subject, clip):
    """Sutherland–Hodgman clipping (convex)."""
    def inside(p, a, b):
        return (b[0]-a[0])*(p[1]-a[1]) - (b[1]-a[1])*(p[0]-a[0]) >= 0

    def intersection(s, e, a, b):
        dc = a - b
        dp = s - e
        n1 = a[0]*b[1] - a[1]*b[0]
        n2 = s[0]*e[1] - s[1]*e[0]
        denom = dc[0]*dp[1] - dc[1]*dp[0]
        if abs(denom) < 1e-9:
            return e
        x = (n1*dp[0] - n2*dc[0]) / denom
        y = (n1*dp[1] - n2*dc[1]) / denom
        return np.array([x, y], dtype=np.float64)

    output = subject
    cp1 = clip[-1]
    for cp2 in clip:
        inp = output
        output = []
        if len(inp) == 0:
            break
        s = inp[-1]
        for e in inp:
            if inside(e, cp1, cp2):
                if not inside(s, cp1, cp2):
                    output.append(intersection(s, e, cp1, cp2))
                output.append(e)
            elif inside(s, cp1, cp2):
                output.append(intersection(s, e, cp1, cp2))
            s = e
        cp1 = cp2

    output = np.array(output, dtype=np.float64)
    if output.shape[0] < 3:
        return 0.0
    return _poly_area(output)


def _bev_iou_from_corners(c1_8x3, c2_8x3):
    p1 = _corners8_to_bev_poly(c1_8x3)
    p2 = _corners8_to_bev_poly(c2_8x3)
    a1 = _poly_area(p1)
    a2 = _poly_area(p2)
    inter = _poly_intersection_area(p1, p2)
    union = a1 + a2 - inter
    return 0.0 if union <= 0 else float(inter / union)


def _greedy_match_by_iou(pred_np, gt_np, iou_thr=0.3):
    """Return matches [(p_idx, g_idx, iou), ...]"""
    P, G = pred_np.shape[0], gt_np.shape[0]
    if P == 0 or G == 0:
        return []
    ious = np.zeros((P, G), dtype=np.float32)
    for i in range(P):
        for j in range(G):
            ious[i, j] = _bev_iou_from_corners(pred_np[i], gt_np[j])

    matches = []
    used_p, used_g = set(), set()
    while True:
        i, j = np.unravel_index(np.argmax(ious), ious.shape)
        best = float(ious[i, j])
        if best < iou_thr:
            break
        if i in used_p or j in used_g:
            ious[i, j] = -1.0
            continue
        matches.append((int(i), int(j), best))
        used_p.add(int(i)); used_g.add(int(j))
        ious[i, :] = -1.0
        ious[:, j] = -1.0
    return matches


def _estimate_se2_from_points(P_xy, G_xy):
    """
    Umeyama 2D, no scale. Returns (theta_deg, tx, ty).
    """
    muP = P_xy.mean(axis=0)
    muG = G_xy.mean(axis=0)
    X = P_xy - muP
    Y = G_xy - muG
    S = (X.T @ Y) / max(len(P_xy), 1)

    U, _, Vt = np.linalg.svd(S)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    t = muG - (R @ muP)
    theta = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
    theta = (theta + 180.0) % 360.0 - 180.0
    return float(theta), float(t[0]), float(t[1])


# ============================================================
# CLI
# ============================================================
def test_parser():
    parser = argparse.ArgumentParser(description="OpenCOOD inference (late/early/intermediate)")
    parser.add_argument('--model_dir', type=str, required=True, help='Continued training path')
    parser.add_argument('--fusion_method', required=True, type=str, default='late',
                        help='late, early or intermediate')

    parser.add_argument('--show_vis', action='store_true')
    parser.add_argument('--show_sequence', action='store_true')
    parser.add_argument('--save_vis', action='store_true')

    parser.add_argument('--save_npy', action='store_true')
    parser.add_argument('--global_sort_detections', action='store_true')

    parser.add_argument('--save_raw_pcd', action='store_true',
                        help='save raw per-CAV point clouds as .pcd files (late only)')

    # LATE: per-cav detection KPI
    parser.add_argument('--save_per_cav_iou', action='store_true',
                        help='save per-CAV mean IoU per frame (late only)')

    # LATE: transform-based pose inconsistency KPIs (no GT needed)
    parser.add_argument('--save_pose_inconsistency', action='store_true',
                        help='save pose inconsistency KPIs for late fusion (temporal jitter + cycle)')
    parser.add_argument('--pose_yaw_only', action='store_true',
                        help='use yaw-only rotation error; otherwise full SO(3) angle')

    # LATE: pose proxy from GT-vs-Pred boxes
    parser.add_argument('--save_pose_proxy_from_boxes', action='store_true',
                        help='save pose proxy + jitter computed from matching pred boxes to GT boxes (late only)')
    parser.add_argument('--pose_proxy_iou_thr', type=float, default=0.3,
                        help='IoU threshold for matching pred<->gt boxes when estimating pose proxy (default=0.3)')
    parser.add_argument('--pose_proxy_min_matches', type=int, default=3,
                        help='minimum matched boxes needed to estimate SE(2) proxy (default=3)')

    return parser.parse_args()


# ============================================================
# Main
# ============================================================
def main():
    opt = test_parser()
    assert opt.fusion_method in ['late', 'early', 'intermediate']
    assert not (opt.show_vis and opt.show_sequence)

    if opt.show_sequence or opt.show_vis or opt.save_vis or opt.save_raw_pcd:
        import open3d as o3d  # noqa: F401
    if opt.show_sequence:
        from opencood.visualization import vis_utils  # noqa: F401

    print("Loading hypes...", flush=True)
    hypes = yaml_utils.load_yaml(None, opt)

    print("Dataset Building", flush=True)
    opencood_dataset = build_dataset(hypes, visualize=True, train=False)
    print(f"{len(opencood_dataset)} samples found.", flush=True)

    data_loader = DataLoader(
        opencood_dataset,
        batch_size=1,
        num_workers=16,
        collate_fn=opencood_dataset.collate_batch_test,
        shuffle=False,
        pin_memory=False,
        drop_last=False
    )
    print(f"DataLoader iters: {len(data_loader)}", flush=True)

    print("Creating Model", flush=True)
    model = train_utils.create_model(hypes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    print("Loading Model from checkpoint", flush=True)
    _, model = train_utils.load_saved_model(opt.model_dir, model)
    model.eval()

    criterion = train_utils.create_loss(hypes)

    # Bookkeeping
    scene_losses = defaultdict(list)
    scene_ious = defaultdict(list)
    scene_counts = defaultdict(int)

    log_dir = os.path.join(opt.model_dir, "tb_inference")
    writer = SummaryWriter(log_dir=log_dir)

    # LATE: per-cav IoU rows
    per_cav_iou_rows = []
    per_cav_iou_rows_added = 0
    per_cav_iou_empty_frames = 0

    # LATE: transform-based pose inconsistency rows
    pose_jitter_rows = []
    pose_cycle_rows = []
    prev_T_map = {}

    # LATE: pose proxy from boxes rows + jitter
    pose_proxy_rows = []
    pose_proxy_jitter_rows = []
    prev_pose_proxy = None  # (scene_id, theta, tx, ty)

    # Result stats for AP
    result_stat = {
        0.3: {'tp': [], 'fp': [], 'gt': 0, 'score': []},
        0.5: {'tp': [], 'fp': [], 'gt': 0, 'score': []},
        0.7: {'tp': [], 'fp': [], 'gt': 0, 'score': []}
    }

    print("Starting inference loop...", flush=True)
    for i, batch_data in tqdm(enumerate(data_loader), total=len(data_loader)):
        with torch.no_grad():
            batch_data = train_utils.to_device(batch_data, device)

            scene_id = extract_scene_id(batch_data, i, dataset=opencood_dataset)
            scene_counts[scene_id] += 1
            scene_frame_id = scene_counts[scene_id] - 1  # per-scene index

            # Save raw per-CAV point clouds (LATE only)
            if opt.save_raw_pcd and opt.fusion_method == "late":
                pcd_root = os.path.join(opt.model_dir, "pcd_per_cav")
                save_raw_pcd_late_like_intermediate(
                    batch_data=batch_data,
                    scene_id=scene_id,
                    sample_idx=i,
                    out_root=pcd_root
                )

            # Forward for loss (ego path)
            output = model(batch_data['ego'])
            loss = criterion(output, batch_data['ego']['label_dict'])
            loss_val = float(loss.item())
            scene_losses[scene_id].append(loss_val)
            writer.add_scalar("loss/frame", loss_val, i)

            # LATE: transform-based pose inconsistency (temporal jitter + cycle)
            if opt.fusion_method == "late" and getattr(opt, "save_pose_inconsistency", False):
                yaw_only = bool(opt.pose_yaw_only)
                pose_jitter_rows.extend(
                    compute_pose_temporal_jitter_rows(
                        batch_data=batch_data,
                        scene_id=scene_id,
                        frame_id=scene_frame_id,
                        prev_T_map=prev_T_map,
                        yaw_only=yaw_only
                    )
                )
                pose_cycle_rows.extend(
                    compute_pose_cycle_consistency_rows(
                        batch_data=batch_data,
                        scene_id=scene_id,
                        frame_id=scene_frame_id,
                        yaw_only=yaw_only,
                        max_pairs=16
                    )
                )

            # Detection inference
            if opt.fusion_method == 'late':
                pred, score, gt = inference_utils.inference_late_fusion(
                    batch_data, model, opencood_dataset, return_per_cav=False
                )

                # per-CAV IoU
                if getattr(opt, "save_per_cav_iou", False):
                    ret = inference_utils.inference_late_fusion(
                        batch_data, model, opencood_dataset, return_per_cav=True
                    )
                    if isinstance(ret, (list, tuple)) and len(ret) >= 4:
                        gt_for_rows = ret[2]
                        per_cav_pred = ret[3]
                        rows = compute_per_cav_iou_rows(per_cav_pred, gt_for_rows, scene_id, scene_frame_id)
                        per_cav_iou_rows.extend(rows)
                        per_cav_iou_rows_added += len(rows)
                        if len(rows) == 0:
                            per_cav_iou_empty_frames += 1
                    else:
                        per_cav_iou_empty_frames += 1

                # -----------------------------
                # ✅ Pose proxy from GT vs Pred boxes (late only)
                # -----------------------------
                if getattr(opt, "save_pose_proxy_from_boxes", False):
                    pred_np = common_utils.torch_tensor_to_numpy(pred) if pred is not None else np.zeros((0, 8, 3), dtype=np.float32)
                    gt_np = common_utils.torch_tensor_to_numpy(gt) if gt is not None else np.zeros((0, 8, 3), dtype=np.float32)

                    matches = _greedy_match_by_iou(pred_np, gt_np, iou_thr=float(opt.pose_proxy_iou_thr))

                    if len(matches) >= int(opt.pose_proxy_min_matches):
                        P = _boxes_to_centers_xy(pred_np)
                        G = _boxes_to_centers_xy(gt_np)

                        Pm = np.stack([P[p] for (p, g, iou) in matches], axis=0)
                        Gm = np.stack([G[g] for (p, g, iou) in matches], axis=0)

                        theta_deg, tx, ty = _estimate_se2_from_points(Pm, Gm)
                        t_norm = float(np.hypot(tx, ty))
                        mean_match_iou = float(np.mean([m[2] for m in matches]))

                        pose_proxy_rows.append([
                            str(scene_id), int(scene_frame_id),
                            theta_deg, tx, ty, t_norm,
                            int(len(matches)), mean_match_iou,
                            int(pred_np.shape[0]), int(gt_np.shape[0]),
                        ])

                        # temporal jitter (within same scene only)
                        if prev_pose_proxy is not None and prev_pose_proxy[0] == str(scene_id):
                            prev_theta, prev_tx, prev_ty = prev_pose_proxy[1], prev_pose_proxy[2], prev_pose_proxy[3]
                            dtheta = (theta_deg - prev_theta + 180.0) % 360.0 - 180.0
                            dtrans = float(np.hypot(tx - prev_tx, ty - prev_ty))
                            pose_proxy_jitter_rows.append([
                                str(scene_id), int(scene_frame_id),
                                float(abs(dtheta)), dtrans
                            ])

                        prev_pose_proxy = (str(scene_id), theta_deg, tx, ty)
                    else:
                        # log NaNs so you can see frames skipped
                        pose_proxy_rows.append([
                            str(scene_id), int(scene_frame_id),
                            float("nan"), float("nan"), float("nan"), float("nan"),
                            int(len(matches)), float("nan"),
                            int(pred_np.shape[0]) if pred_np is not None else 0,
                            int(gt_np.shape[0]) if gt_np is not None else 0,
                        ])

            elif opt.fusion_method == 'early':
                pred, score, gt = inference_utils.inference_early_fusion(
                    batch_data, model, opencood_dataset
                )
            else:
                pred, score, gt, _ = inference_utils.inference_intermediate_fusion(
                    batch_data, model, opencood_dataset, return_per_cav=True
                )

            # mean IoU diagnostic
            mean_iou, _, _ = mean_det_to_gt_iou(pred, gt)
            if mean_iou is not None:
                mean_iou = float(mean_iou)
                scene_ious[scene_id].append(mean_iou)
                writer.add_scalar("iou/frame_mean", mean_iou, i)

            # TP/FP stats for AP
            eval_utils.caluclate_tp_fp(pred, score, gt, result_stat, 0.3)
            eval_utils.caluclate_tp_fp(pred, score, gt, result_stat, 0.5)
            eval_utils.caluclate_tp_fp(pred, score, gt, result_stat, 0.7)

            # Save npy
            if opt.save_npy:
                npy_save_path = os.path.join(opt.model_dir, 'npy')
                os.makedirs(npy_save_path, exist_ok=True)
                inference_utils.save_prediction_gt(
                    pred, gt, batch_data['ego']['origin_lidar'][0], i, npy_save_path
                )

    # ---------------------------------------------------------
    # Save LATE outputs
    # ---------------------------------------------------------
    if opt.fusion_method == "late" and getattr(opt, "save_per_cav_iou", False):
        per_cav_iou_csv = os.path.join(opt.model_dir, "per_cav_iou.csv")
        with open(per_cav_iou_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["scene_id", "frame_id", "cav_key", "mean_iou", "num_pred", "num_gt"])
            w.writerows(per_cav_iou_rows)
        print(f"[INFO] Saved: {per_cav_iou_csv}", flush=True)
        print(f"[INFO] per-cav IoU rows added: {per_cav_iou_rows_added}", flush=True)
        print(f"[INFO] frames with no per-cav IoU rows: {per_cav_iou_empty_frames}", flush=True)

    if opt.fusion_method == "late" and getattr(opt, "save_pose_inconsistency", False):
        pose_jitter_csv = os.path.join(opt.model_dir, "pose_temporal_jitter.csv")
        with open(pose_jitter_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["scene_id", "frame_id", "cav_key", "jitter_trans_m", "jitter_rot_deg"])
            w.writerows(pose_jitter_rows)
        print(f"[INFO] Saved: {pose_jitter_csv}", flush=True)

        pose_cycle_csv = os.path.join(opt.model_dir, "pose_cycle_error.csv")
        with open(pose_cycle_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["scene_id", "frame_id", "i", "j", "cycle_trans_m", "cycle_rot_deg"])
            w.writerows(pose_cycle_rows)
        print(f"[INFO] Saved: {pose_cycle_csv}", flush=True)

    if opt.fusion_method == "late" and getattr(opt, "save_pose_proxy_from_boxes", False):
        out1 = os.path.join(opt.model_dir, "pose_proxy_from_boxes.csv")
        with open(out1, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "scene_id", "frame_id",
                "theta_deg", "tx", "ty", "t_norm",
                "num_matches", "mean_match_iou",
                "num_pred", "num_gt"
            ])
            w.writerows(pose_proxy_rows)
        print(f"[INFO] Saved: {out1} (rows={len(pose_proxy_rows)})", flush=True)

        out2 = os.path.join(opt.model_dir, "pose_proxy_jitter.csv")
        with open(out2, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["scene_id", "frame_id", "delta_theta_deg", "delta_trans_m"])
            w.writerows(pose_proxy_jitter_rows)
        print(f"[INFO] Saved: {out2} (rows={len(pose_proxy_jitter_rows)})", flush=True)

    # ---------------------------------------------------------
    # Save scene metrics
    # ---------------------------------------------------------
    scene_csv_path = os.path.join(opt.model_dir, "scene_metrics.csv")
    with open(scene_csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scene_id", "num_samples", "mean_loss", "mean_iou"])
        for s in sorted(scene_counts.keys()):
            w.writerow([
                s,
                int(scene_counts[s]),
                float(np.mean(scene_losses[s])) if scene_losses[s] else float("nan"),
                float(np.mean(scene_ious[s])) if scene_ious[s] else float("nan")
            ])
    print(f"[INFO] Scene metrics saved to: {scene_csv_path}", flush=True)

    # Final eval
    eval_utils.eval_final_results(result_stat, opt.model_dir, opt.global_sort_detections)

    writer.close()
    print(f"Inference complete. TensorBoard logs at: {log_dir}", flush=True)


if __name__ == '__main__':
    main()
