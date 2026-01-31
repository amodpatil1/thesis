# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>, Hao Xiang <haxiang@g.ucla.edu>, Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib

import argparse
import os
import time
from collections import defaultdict
import csv

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

import torch
import open3d as o3d
import resource

# Increase max open files (must run before DataLoader workers start)
soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, hard))

import torch.multiprocessing as mp
mp.set_sharing_strategy("file_system")

from torch.utils.data import DataLoader

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils, inference_utils
from opencood.data_utils.datasets import build_dataset
from opencood.utils import eval_utils, common_utils
from opencood.visualization import vis_utils


# ==========================================================
# ✅ TIMESTAMP-FREE REAL CAV-ID RESOLUTION
# (works even when scene_id is chunk_0000)
# ==========================================================
def _first_str(x):
    if isinstance(x, str):
        return x
    if isinstance(x, (list, tuple)) and len(x) > 0 and isinstance(x[0], str):
        return x[0]
    return None


def cav_id_from_path_digits(p: str):
    """
    Extract cav id by scanning path components from the end
    and returning the first all-digit directory name.
    """
    if not isinstance(p, str) or not p:
        return None
    parts = os.path.normpath(p).split(os.sep)
    for comp in reversed(parts):
        if comp.isdigit():
            return comp
    return None


def scenario_dir_from_any_path(p: str):
    """
    Walk upward from a path until we find a folder that contains >=2 numeric subfolders.
    That folder is treated as the scenario folder containing cav-id folders.
    """
    if not isinstance(p, str) or not p:
        return None

    cur = os.path.abspath(p)
    if os.path.isfile(cur):
        cur = os.path.dirname(cur)

    while True:
        if os.path.isdir(cur):
            try:
                subdirs = [
                    d for d in os.listdir(cur)
                    if d.isdigit() and os.path.isdir(os.path.join(cur, d))
                ]
                if len(subdirs) >= 2:
                    return cur
            except Exception:
                pass

        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent

    return None


def build_late_cav_map(batch_data):
    """
    Late fusion: map cav_key ('ego','2167',...) -> real numeric cav id.
    We pull any path field per cav and parse the nearest numeric folder.
    """
    out = {}
    for cav_key, cav_content in batch_data.items():
        if not isinstance(cav_content, dict):
            continue

        p = _first_str(cav_content.get("lidar_path", None))
        if p is None:
            for k in ["pcd_path", "yaml_path", "file_path", "path", "filename"]:
                p = _first_str(cav_content.get(k, None))
                if p:
                    break

        cav_id = cav_id_from_path_digits(p) if p else None
        if cav_id:
            out[str(cav_key)] = cav_id
    return out


def build_intermediate_idx_to_cavid(batch_data):
    """
    Intermediate fusion: per_cav keys are ego/cav_1/cav_2...
    We derive cav ids by:
      1) finding scenario dir from any ego path
      2) listing numeric cav folders inside it
      3) truncating to record_len
    Returns {0:'2149', 1:'2158', 2:'2167', ...}
    """
    ego = batch_data.get("ego", {})
    if not isinstance(ego, dict):
        return {}

    # find any path-like string to locate scenario directory
    p = None
    for k in ["yaml_path", "lidar_path", "pcd_path", "file_path", "path", "filename"]:
        p = _first_str(ego.get(k, None))
        if p:
            break

    scene_dir = scenario_dir_from_any_path(p) if p else None
    if scene_dir is None:
        return {}

    cav_ids = sorted([
        d for d in os.listdir(scene_dir)
        if d.isdigit() and os.path.isdir(os.path.join(scene_dir, d))
    ])

    # record_len tells how many cavs in this sample
    rl = ego.get("record_len", None)
    L = None
    try:
        if isinstance(rl, torch.Tensor):
            L = int(rl[0].item())
        elif isinstance(rl, (list, tuple, np.ndarray)):
            L = int(rl[0])
        elif rl is not None:
            L = int(rl)
    except Exception:
        L = None

    if L is not None:
        cav_ids = cav_ids[:L]

    return {idx: cav_id for idx, cav_id in enumerate(cav_ids)}


# ==========================================================
# CLI
# ==========================================================
def test_parser():
    parser = argparse.ArgumentParser(description="OpenCOOD inference (late/early/intermediate)")
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Continued training path')
    parser.add_argument('--fusion_method', required=True, type=str,
                        default='late',
                        help='late, early or intermediate')
    parser.add_argument('--show_vis', action='store_true',
                        help='whether to show image visualization result')
    parser.add_argument('--show_sequence', action='store_true',
                        help='whether to show video visualization result '
                             '(cannot be set true with show_vis together)')
    parser.add_argument('--save_vis', action='store_true',
                        help='whether to save visualization result')
    parser.add_argument('--save_npy', action='store_true',
                        help='whether to save prediction and gt result in npy folder')
    parser.add_argument('--global_sort_detections', action='store_true',
                        help='whether to globally sort detections by confidence score')

    parser.add_argument('--save_raw_pcd', action='store_true',
                        help='save raw per-CAV point clouds as .pcd files (late fusion only)')

    opt = parser.parse_args()
    return opt


def extract_scene_id(batch_data, batch_idx, dataset=None):
    ego = batch_data.get('ego', {})
    if isinstance(ego, dict):
        path_keys = ['lidar_path', 'pcd_path', 'yaml_path', 'file_path', 'filename', 'path']
        for k in path_keys:
            if k in ego:
                path_val = ego[k]
                if isinstance(path_val, (list, tuple)) and len(path_val) > 0:
                    path_val = path_val[0]
                if isinstance(path_val, str):
                    scene_dir = os.path.dirname(path_val)
                    scene_root = os.path.dirname(scene_dir)
                    scene_name = os.path.basename(scene_root)
                    return scene_name

    chunk = batch_idx // 100
    return f"chunk_{chunk:04d}"


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


def _to_np(x):
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _tensor_to_xyz_np(x):
    if x is None:
        return None
    arr = _to_np(x)
    if arr is None:
        return None
    if arr.ndim == 3:
        arr = arr[0]
    if arr.ndim != 2:
        return None
    if arr.shape[1] < 3:
        return None
    return arr[:, :3]


def _save_pcd_xyz(path, xyz_np):
    if xyz_np is None or xyz_np.shape[0] == 0:
        return
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz_np.astype(np.float64))
    o3d.io.write_point_cloud(path, pcd, write_ascii=False)


def _apply_4x4_to_boxes_corners(boxes_8_3: torch.Tensor, T_4x4: torch.Tensor) -> torch.Tensor:
    if boxes_8_3 is None:
        return boxes_8_3
    if isinstance(boxes_8_3, torch.Tensor) and boxes_8_3.shape[0] == 0:
        return boxes_8_3

    if not isinstance(boxes_8_3, torch.Tensor):
        boxes_8_3 = torch.as_tensor(boxes_8_3)

    if not isinstance(T_4x4, torch.Tensor):
        T_4x4 = torch.as_tensor(T_4x4, device=boxes_8_3.device, dtype=boxes_8_3.dtype)
    else:
        T_4x4 = T_4x4.to(device=boxes_8_3.device, dtype=boxes_8_3.dtype)

    N = boxes_8_3.shape[0]
    pts = boxes_8_3.reshape(-1, 3)
    ones = torch.ones((pts.shape[0], 1), device=pts.device, dtype=pts.dtype)
    pts_h = torch.cat([pts, ones], dim=1)
    pts_out = (T_4x4 @ pts_h.t()).t()[:, :3]
    return pts_out.reshape(N, 8, 3)


# ==========================================================
# MAIN
# ==========================================================
def main():
    opt = test_parser()
    assert opt.fusion_method in ['late', 'early', 'intermediate']
    assert not (opt.show_vis and opt.show_sequence), \
        'you can only visualize results in single image mode or video mode'

    hypes = yaml_utils.load_yaml(None, opt)

    print('Dataset Building')
    opencood_dataset = build_dataset(hypes, visualize=True, train=False)
    print(f"{len(opencood_dataset)} samples found.")
    data_loader = DataLoader(opencood_dataset,
                             batch_size=1,
                             num_workers=16,
                             collate_fn=opencood_dataset.collate_batch_test,
                             shuffle=False,
                             pin_memory=False,
                             drop_last=False)

    print('Creating Model')
    model = train_utils.create_model(hypes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        model.to(device)

    print('Loading Model from checkpoint')
    saved_path = opt.model_dir
    _, model = train_utils.load_saved_model(saved_path, model)
    model.eval()

    criterion = train_utils.create_loss(hypes)

    scene_losses = defaultdict(list)
    scene_ious = defaultdict(list)
    scene_counts = defaultdict(int)

    log_dir = os.path.join(opt.model_dir, "tb_inference")
    print(f"TensorBoard logs will be written to: {log_dir}")
    writer = SummaryWriter(log_dir=log_dir)

    per_cav_rows = []

    result_stat = {
        0.3: {'tp': [], 'fp': [], 'gt': 0, 'score': []},
        0.5: {'tp': [], 'fp': [], 'gt': 0, 'score': []},
        0.7: {'tp': [], 'fp': [], 'gt': 0, 'score': []}
    }

    det_root = os.path.join(opt.model_dir, "per_chunk_dets")
    os.makedirs(det_root, exist_ok=True)

    # ---------- INFERENCE LOOP ----------
    for step, batch_data in tqdm(enumerate(data_loader), total=len(opencood_dataset)):
        with torch.no_grad():
            per_cav = None
            batch_data = train_utils.to_device(batch_data, device)

            scene_id = extract_scene_id(batch_data, step, dataset=opencood_dataset)
            scene_counts[scene_id] += 1

            scene_dir = os.path.join(det_root, str(scene_id))
            os.makedirs(scene_dir, exist_ok=True)

            sample_id = step

            # ---------- LOSS PER SAMPLE ----------
            output_for_loss = model(batch_data['ego'])
            loss_value = criterion(output_for_loss, batch_data['ego']['label_dict'])
            loss_scalar = float(loss_value.item())
            scene_losses[scene_id].append(loss_scalar)
            writer.add_scalar("inference/loss", loss_scalar, step)

            # ---------- DETECTION INFERENCE ----------
            if opt.fusion_method == 'late':
                pred_box_tensor, pred_score, gt_box_tensor, per_cav = \
                    inference_utils.inference_late_fusion(
                        batch_data, model, opencood_dataset, return_per_cav=True
                    )
            elif opt.fusion_method == 'intermediate':
                pred_box_tensor, pred_score, gt_box_tensor, per_cav = \
                    inference_utils.inference_intermediate_fusion(
                        batch_data, model, opencood_dataset, return_per_cav=True
                    )
            else:
                pred_box_tensor, pred_score, gt_box_tensor = \
                    inference_utils.inference_early_fusion(
                        batch_data, model, opencood_dataset
                    )

            # ---------- SAVE FUSED ----------
            fused_out = {
                "scene_id": str(scene_id),
                "sample_id": int(sample_id),
                "pred_box_tensor": _to_np(pred_box_tensor),
                "pred_score": _to_np(pred_score),
                "gt_box_tensor": _to_np(gt_box_tensor),
            }
            np.save(os.path.join(scene_dir, f"{sample_id:06d}_fused.npy"), fused_out, allow_pickle=True)

            # =====================================================================
            # ✅ SAVE DETECTIONS PER CAV USING REAL NUMERIC CAV ID
            # =====================================================================
            if opt.fusion_method in ["late", "intermediate"] and per_cav is not None:
                per_cav_dir = os.path.join(scene_dir, "per_cav")
                os.makedirs(per_cav_dir, exist_ok=True)

                # Build cav-id mapping ONCE per sample
                late_map = None
                idx_map = None
                if opt.fusion_method == "late":
                    late_map = build_late_cav_map(batch_data)
                else:
                    idx_map = build_intermediate_idx_to_cavid(batch_data)

                # For optional intermediate IoU in cav frame
                ego_pairwise = None
                ego_record_len = None
                if opt.fusion_method == "intermediate":
                    ego_pairwise = batch_data.get("ego", {}).get("pairwise_t_matrix", None)
                    ego_record_len = batch_data.get("ego", {}).get("record_len", None)

                cav_list = []

                for cav_key, d in per_cav.items():
                    cav_key_str = str(cav_key)

                    if cav_key_str.lower() == "ego":
                        cav_idx = 0
                    elif cav_key_str.startswith("cav_"):
                        cav_idx = int(cav_key_str.split("_", 1)[1])
                    else:
                        continue

                    # Resolve real cav id
                    # Resolve real cav id with fallbacks
                    real_cav_id = None

                    if opt.fusion_method == "late":
                        # Try to get from map
                        real_cav_id = late_map.get(cav_key_str, None) if late_map else None
                        
                        # Fallback 1: If key itself is numeric, use it
                        if real_cav_id is None and cav_key_str.isdigit():
                            real_cav_id = cav_key_str
                        
                        # Fallback 2: Use indexed naming
                        if real_cav_id is None:
                            real_cav_id = f"cav_{cav_idx}"
                    else:  # intermediate
                        # Try to get from map
                        real_cav_id = idx_map.get(cav_idx, None) if idx_map else None
                        
                        # Fallback: Use indexed naming
                        if real_cav_id is None:
                            real_cav_id = f"cav_{cav_idx}"

                    # Skip only if we still couldn't determine an ID (shouldn't happen with fallbacks)
                    if real_cav_id is None:
                        continue

                    cav_list.append(str(real_cav_id))

                    b = d.get("boxes", None)
                    s = d.get("scores", None)

                    cav_mean_iou = None
                    cav_num_pred = int(b.shape[0]) if isinstance(b, torch.Tensor) else 0
                    cav_num_gt = int(gt_box_tensor.shape[0]) if isinstance(gt_box_tensor, torch.Tensor) else 0

                    if opt.fusion_method == "late":
                        cav_mean_iou, cav_num_pred, cav_num_gt = mean_det_to_gt_iou(b, gt_box_tensor)
                    else:
                        if (ego_pairwise is not None) and (ego_record_len is not None) and isinstance(gt_box_tensor, torch.Tensor):
                            try:
                                L = int(ego_record_len[0])
                            except Exception:
                                L = None
                            if L is not None and 0 <= cav_idx < L:
                                T_ego_to_cav = ego_pairwise[0, 0, cav_idx]
                                gt_cav = _apply_4x4_to_boxes_corners(gt_box_tensor, T_ego_to_cav)
                                cav_mean_iou, cav_num_pred, cav_num_gt = mean_det_to_gt_iou(b, gt_cav)

                    per_cav_rows.append([
                        str(scene_id), int(sample_id), str(real_cav_id),
                        -1.0 if cav_mean_iou is None else float(cav_mean_iou),
                        int(cav_num_pred), int(cav_num_gt)
                    ])

                    out = {
                        "scene_id": str(scene_id),
                        "sample_id": int(sample_id),
                        "cav_id": str(real_cav_id),
                        "pred_box_tensor": _to_np(b),
                        "pred_score": _to_np(s),
                        "gt_box_tensor": _to_np(gt_box_tensor),
                        "mean_iou_det_to_gt": cav_mean_iou,
                    }
                    np.save(os.path.join(per_cav_dir, f"{sample_id:06d}_{real_cav_id}_pred.npy"),
                            out, allow_pickle=True)

                with open(os.path.join(per_cav_dir, f"{sample_id:06d}_cavs.txt"), "w") as f:
                    f.write("\n".join(cav_list))

            # ---------- TP/FP stats ----------
            if opt.fusion_method in ["late", "intermediate"]:
                eval_utils.caluclate_tp_fp(pred_box_tensor, pred_score, gt_box_tensor, result_stat, 0.3)
                eval_utils.caluclate_tp_fp(pred_box_tensor, pred_score, gt_box_tensor, result_stat, 0.5)
                eval_utils.caluclate_tp_fp(pred_box_tensor, pred_score, gt_box_tensor, result_stat, 0.7)

    # ---------- WRITE PER-CAV IOU CSV ----------
    per_cav_csv_path = os.path.join(opt.model_dir, "per_cav_iou.csv")
    with open(per_cav_csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scene_id", "sample_id", "cav_id", "mean_iou_det_to_gt", "num_pred", "num_gt"])
        w.writerows(per_cav_rows)
    print(f"[INFO] Per-CAV IoU saved to: {per_cav_csv_path}")

    # ---------- FINAL EVAL ----------
    eval_utils.eval_final_results(result_stat, opt.model_dir, opt.global_sort_detections)

    writer.close()
    print(f"TensorBoard writer closed. Logs at: {log_dir}")


if __name__ == '__main__':
    main()
