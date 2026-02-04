# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>, Hao Xiang <haxiang@g.ucla.edu>, Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib

"""
opencood/tools/inference.py

This version:
- Runs inference for late/early/intermediate fusion
- Computes + saves feature misalignment metrics (INTERMEDIATE only):
    1) frame_feature_misalignment.csv
    2) per_cav_feature_misalignment.csv
    3) chunk_report.csv   (mean feature misalignment per chunk)
- Saves basic scene_metrics.csv (mean loss / mean IoU per scene)

Notes:
- Feature misalignment requires your inference_utils.compute_feature_misalignment_intermediate(...)
  to return (frame_mean, per_cav_dict) or (None,{}) when not computable.
- This script keeps Open3D/vis optional and imports them lazily.
"""

print("executing", flush=True)

import argparse
import csv
import os
import re
import time
from collections import defaultdict

import numpy as np
import resource
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.data_utils.datasets import build_dataset
from opencood.tools import inference_utils, train_utils
from opencood.utils import common_utils, eval_utils

# Increase max open files (must run before DataLoader workers start)
soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, hard))

mp.set_sharing_strategy("file_system")



def pose_error_from_T(T_gt, T_est):
    """
    Compare two 4x4 SE(3) matrices (cav -> ego)
    Returns: translation error (m), rotation error (deg)
    """
    dT = np.linalg.inv(T_gt) @ T_est
    dt = dT[:3, 3]
    et = float(np.linalg.norm(dt))

    R = dT[:3, :3]
    cos_angle = (np.trace(R) - 1.0) / 2.0
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    er = float(np.degrees(np.arccos(cos_angle)))
    return et, er


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

    # raw pcd export
    parser.add_argument('--save_raw_pcd', action='store_true',
                        help='save raw per-CAV point clouds as .pcd files')

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

    if dataset is not None:
        for attr in ['data_list', 'scenario_database', 'opv2v_database']:
            if hasattr(dataset, attr):
                db = getattr(dataset, attr)
                entry = None
                try:
                    entry = db[batch_idx]
                except Exception:
                    entry = None

                if isinstance(entry, dict):
                    for k in ['lidar_path', 'pcd_path', 'yaml_path', 'path', 'file_path']:
                        if k in entry and isinstance(entry[k], str):
                            scene_dir = os.path.dirname(entry[k])
                            scene_root = os.path.dirname(scene_dir)
                            scene_name = os.path.basename(scene_root)
                            return scene_name

                if isinstance(db, (list, tuple)) and isinstance(entry, str):
                    scene_dir = os.path.dirname(entry)
                    scene_root = os.path.dirname(scene_dir)
                    scene_name = os.path.basename(scene_root)
                    return scene_name

    chunk = batch_idx // 100
    return f"chunk_{chunk:04d}"


def mean_det_to_gt_iou(det_boxes_tensor, gt_boxes_tensor):
    """
    Mean (max-over-GT) IoU per predicted box, averaged across predictions.
    Returns: (mean_iou_or_None, num_pred, num_gt)
    """
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


def _save_pcd_xyz(o3d, path, xyz_np):
    """Save xyz points as .pcd using Open3D (passed in)."""
    if xyz_np is None or xyz_np.shape[0] == 0:
        return
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz_np.astype(np.float64))
    o3d.io.write_point_cloud(path, pcd, write_ascii=False)


# ==========================================================
# ✅ Real cav-id mapping: derive ID from each cav's own path
# ==========================================================
_SCENE_TS_RE = re.compile(r"\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2}")


def _first_str(x):
    if isinstance(x, str):
        return x
    if isinstance(x, (list, tuple)) and len(x) > 0 and isinstance(x[0], str):
        return x[0]
    return None


def _find_path_in_cav_dict(cav_dict):
    """Return a path-like string from a single cav dict (ego or cav_*)."""
    if not isinstance(cav_dict, dict):
        return None
    for k in ["lidar_path", "pcd_path", "yaml_path", "file_path", "path", "filename"]:
        p = _first_str(cav_dict.get(k, None))
        if p:
            return p
    return None


def _cav_id_from_any_path(p):
    """
    Extract cav_id from a path like:
      .../validate/2021_08_20_21_48_35/2167/000123.pcd
    Returns '2167' or None.
    """
    if not p or not isinstance(p, str):
        return None

    m = _SCENE_TS_RE.search(p)
    if not m:
        return None

    ts = m.group(0)
    parts = os.path.normpath(p).split(os.sep)

    # find index of timestamp folder, cav_id is the next component
    for i in range(len(parts) - 1):
        if parts[i] == ts and i + 1 < len(parts):
            cav_id = parts[i + 1]
            if cav_id.isdigit():
                return cav_id
            return None
    return None


def build_cav_key_to_real_id(batch_data):
    """
    Returns dict keyed by cav_key string:
      {'ego':'2149', 'cav_1':'2167', 'cav_2':'2158', ...}

    Robust: uses each cav's own file path so ordering never breaks.
    """
    out = {}
    for cav_key, cav_content in batch_data.items():
        if not isinstance(cav_content, dict):
            continue
        p = _find_path_in_cav_dict(cav_content)
        cav_id = _cav_id_from_any_path(p)
        if cav_id is not None:
            out[str(cav_key)] = cav_id
    return out


def main():
    # ---------------------------------------------------------
    # Parse args early so -h/--help always works
    # ---------------------------------------------------------
    opt = test_parser()

    assert opt.fusion_method in ['late', 'early', 'intermediate']
    assert not (opt.show_vis and opt.show_sequence), \
        'you can only visualize results in single image mode or video mode'

    # Lazy imports
    import matplotlib
    matplotlib.use("Agg")

    o3d = None
    if opt.show_sequence or opt.show_vis or opt.save_vis or opt.save_raw_pcd:
        import open3d as o3d  # noqa: F401

    vis_utils = None
    if opt.show_sequence:
        from opencood.visualization import vis_utils  # noqa: F401

    # ---------------------------------------------------------
    # Dataset + model
    # ---------------------------------------------------------
    print("Loading hypes...", flush=True)
    hypes = yaml_utils.load_yaml(None, opt)

    print('Dataset Building', flush=True)
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

    print('Creating Model', flush=True)
    model = train_utils.create_model(hypes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    print('Loading Model from checkpoint', flush=True)
    _, model = train_utils.load_saved_model(opt.model_dir, model)
    model.eval()

    criterion = train_utils.create_loss(hypes)

    # ---------------------------------------------------------
    # Pose error helper
    # ---------------------------------------------------------
    def pose_error_from_T(T_gt_np, T_est_np):
        """
        Compare two 4x4 SE(3) matrices (cav -> ego).
        Returns: translation error (m), rotation error (deg)
        """
        dT = np.linalg.inv(T_gt_np) @ T_est_np
        dt = dT[:3, 3]
        et = float(np.linalg.norm(dt))

        R = dT[:3, :3]
        cos_angle = (np.trace(R) - 1.0) / 2.0
        cos_angle = float(np.clip(cos_angle, -1.0, 1.0))
        er = float(np.degrees(np.arccos(cos_angle)))
        return et, er

    # ---------------------------------------------------------
    # Bookkeeping
    # ---------------------------------------------------------
    scene_losses = defaultdict(list)
    scene_ious = defaultdict(list)
    scene_counts = defaultdict(int)

    log_dir = os.path.join(opt.model_dir, "tb_inference")
    writer = SummaryWriter(log_dir=log_dir)

    # Feature misalignment storage (INTERMEDIATE only)
    frame_mis_rows = []
    per_cav_mis_rows = []
    chunk_rows = []
    chunk_vals = []
    chunk_size = 100

    # ✅ Pose error storage (LATE only)
    pose_rows = []

    # Result stats for AP
    result_stat = {
        0.3: {'tp': [], 'fp': [], 'gt': 0, 'score': []},
        0.5: {'tp': [], 'fp': [], 'gt': 0, 'score': []},
        0.7: {'tp': [], 'fp': [], 'gt': 0, 'score': []}
    }

    # Debug: detect if per-agent features ever appear
    saw_per_agent_feats = False

    # ---------------------------------------------------------
    # Inference loop
    # ---------------------------------------------------------
    print("Starting inference loop...", flush=True)
    for i, batch_data in tqdm(enumerate(data_loader), total=len(data_loader)):
        with torch.no_grad():
            batch_data = train_utils.to_device(batch_data, device)

            # ✅ scene_id must be defined BEFORE pose logging uses it
            scene_id = extract_scene_id(batch_data, i, dataset=opencood_dataset)
            scene_counts[scene_id] += 1

            # -------------------------------------------------
            # ✅ POSE ERROR (late fusion only)
            # -------------------------------------------------
            if opt.fusion_method == "late":
                for cav_key, cav_content in batch_data.items():
                    if not isinstance(cav_content, dict):
                        continue

                    T_gt = cav_content.get("transformation_matrix", None)
                    if T_gt is None:
                        continue

                    # NOTE: currently identical -> errors will be ~0 unless you also store a noisy/estimated matrix
                    T_est = cav_content.get("transformation_matrix", None)
                    if T_est is None:
                        continue

                    T_gt_np = T_gt.detach().cpu().numpy()
                    T_est_np = T_est.detach().cpu().numpy()

                    et, er = pose_error_from_T(T_gt_np, T_est_np)

                    pose_rows.append([
                        str(scene_id),
                        int(i),
                        str(cav_key),
                        float(et),
                        float(er)
                    ])

            # ---------- Forward (ego) for loss + features ----------
            output = model(batch_data['ego'])
            loss = criterion(output, batch_data['ego']['label_dict'])
            loss_val = float(loss.item())
            scene_losses[scene_id].append(loss_val)
            writer.add_scalar("loss/frame", loss_val, i)

            # ---------- FEATURE MISALIGNMENT (INTERMEDIATE only) ----------
            if opt.fusion_method == "intermediate" and isinstance(output, dict):
                feats_pa = output.get("spatial_features_2d_per_agent", None)

                if feats_pa is not None:
                    saw_per_agent_feats = True

                    frame_mis, per_cav_mis = inference_utils.compute_feature_misalignment_intermediate(
                        batch_data, feats_pa
                    )

                    if frame_mis is not None:
                        frame_mis = float(frame_mis)
                        writer.add_scalar("misalign/frame_mean", frame_mis, i)
                        chunk_vals.append(frame_mis)

                        frame_mis_rows.append([
                            str(scene_id),
                            int(i),
                            frame_mis,
                            int(len(per_cav_mis))
                        ])

                        for cav_idx, mis in per_cav_mis.items():
                            per_cav_mis_rows.append([
                                str(scene_id),
                                int(i),
                                int(cav_idx),
                                float(mis)
                            ])

            # ---------- Detection inference ----------
            if opt.fusion_method == 'late':
                pred, score, gt = inference_utils.inference_late_fusion(
                    batch_data, model, opencood_dataset, return_per_cav=False
                )
            elif opt.fusion_method == 'early':
                pred, score, gt = inference_utils.inference_early_fusion(
                    batch_data, model, opencood_dataset
                )
            else:
                pred, score, gt, _ = inference_utils.inference_intermediate_fusion(
                    batch_data, model, opencood_dataset, return_per_cav=True
                )

            # ---------- IoU diagnostic ----------
            mean_iou, _, _ = mean_det_to_gt_iou(pred, gt)
            if mean_iou is not None:
                mean_iou = float(mean_iou)
                scene_ious[scene_id].append(mean_iou)
                writer.add_scalar("iou/frame_mean", mean_iou, i)

            # ---------- TP/FP stats for AP ----------
            eval_utils.caluclate_tp_fp(pred, score, gt, result_stat, 0.3)
            eval_utils.caluclate_tp_fp(pred, score, gt, result_stat, 0.5)
            eval_utils.caluclate_tp_fp(pred, score, gt, result_stat, 0.7)

            # ---------- Chunk aggregation (misalignment) ----------
            if opt.fusion_method == "intermediate" and ((i + 1) % chunk_size == 0):
                chunk_id = i // chunk_size
                chunk_rows.append([
                    int(chunk_id),
                    int(i - chunk_size + 1),
                    int(i),
                    float(np.mean(chunk_vals)) if chunk_vals else float("nan"),
                    int(len(chunk_vals))
                ])
                chunk_vals = []

            # ---------- Save NPY (OpenCOOD default) ----------
            if opt.save_npy:
                npy_save_path = os.path.join(opt.model_dir, 'npy')
                os.makedirs(npy_save_path, exist_ok=True)
                inference_utils.save_prediction_gt(
                    pred,
                    gt,
                    batch_data['ego']['origin_lidar'][0],
                    i,
                    npy_save_path
                )

    # ---------------------------------------------------------
    # Flush last chunk
    # ---------------------------------------------------------
    if opt.fusion_method == "intermediate" and chunk_vals:
        last_i = len(opencood_dataset) - 1
        chunk_id = last_i // chunk_size
        chunk_rows.append([
            int(chunk_id),
            int(chunk_id * chunk_size),
            int(last_i),
            float(np.mean(chunk_vals)),
            int(len(chunk_vals))
        ])

    # ---------------------------------------------------------
    # ✅ Save pose errors ONCE (late fusion)
    # ---------------------------------------------------------
    if opt.fusion_method == "late":
        pose_csv = os.path.join(opt.model_dir, "pose_error_per_cav.csv")
        with open(pose_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["scene_id", "frame_id", "cav_key",
                        "translation_error_m", "rotation_error_deg"])
            w.writerows(pose_rows)
        print(f"[INFO] Pose error saved to: {pose_csv}", flush=True)

    # ---------------------------------------------------------
    # Save feature misalignment CSVs
    # ---------------------------------------------------------
    if opt.fusion_method == "intermediate":
        if not saw_per_agent_feats:
            print(
                "[WARN] spatial_features_2d_per_agent was NEVER found in model output. "
                "Your model/backbone must be modified to output per-agent BEV features. "
                "Misalignment CSVs will be empty.",
                flush=True
            )

        frame_csv_path = os.path.join(opt.model_dir, "frame_feature_misalignment.csv")
        with open(frame_csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["scene_id", "sample_id", "frame_misalign_mean", "num_agents_used"])
            w.writerows(frame_mis_rows)
        print(f"[INFO] Frame feature misalignment saved to: {frame_csv_path}", flush=True)

        per_cav_csv_path = os.path.join(opt.model_dir, "per_cav_feature_misalignment.csv")
        with open(per_cav_csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["scene_id", "sample_id", "cav_idx", "misalign"])
            w.writerows(per_cav_mis_rows)
        print(f"[INFO] Per-CAV feature misalignment saved to: {per_cav_csv_path}", flush=True)

        chunk_csv_path = os.path.join(opt.model_dir, "chunk_report.csv")
        with open(chunk_csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["chunk_id", "start_sample", "end_sample", "feature_misalignment", "num_frames"])
            w.writerows(chunk_rows)
        print(f"[INFO] Chunk report saved to: {chunk_csv_path}", flush=True)

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

    # ---------------------------------------------------------
    # Final eval (AP etc.)
    # ---------------------------------------------------------
    eval_utils.eval_final_results(result_stat, opt.model_dir, opt.global_sort_detections)

    writer.close()
    print(f"Inference complete. TensorBoard logs at: {log_dir}", flush=True)


if __name__ == '__main__':
    main()
