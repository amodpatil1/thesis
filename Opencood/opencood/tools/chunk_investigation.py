# -*- coding: utf-8 -*-
# Chunk-level inference analysis for OpenCOOD
#
# Drop this file into: opencood/tools/chunk_analysis.py
#
# Example:
#   cd /OpenCOOD
#   python opencood/tools/chunk_analysis.py \
#       --model_dir opencood/trained/pointpillar_attentive_fusion \
#       --fusion_method intermediate \
#       --chunk_size 10 \
#       --target_chunk chunk_0003 \
#       --save_failed_vis \
#       --failed_iou_thresh 0.5 \
#       --save_npy
#
# Notes:
# - If you do NOT pass --hypes_yaml, this script will try:
#     <model_dir>/config.yaml, <model_dir>/config.yml, <model_dir>/hypes.yaml, <model_dir>/hypes.yml
# - Failed visualizations are saved into: <model_dir>/chunk_debug_vis/
# - Debug JSON (if enabled) is saved alongside each PNG.
# - NPY dumps (if enabled) are saved into: <model_dir>/chunk_debug_npy/
#     * <chunk>_pred_<frame>.npy
#     * <chunk>_score_<frame>.npy   <-- ADDED
#     * <chunk>_gt_<frame>.npy
# - Combined Loss+IoU plot per chunk is saved into: <model_dir>/<chunk>_loss_iou_timeline.png  <-- ADDED

import argparse
import os
import time
import json
import csv
from collections import defaultdict

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

import torch
import open3d as o3d

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless servers
import matplotlib.pyplot as plt

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils, inference_utils
from opencood.data_utils.datasets import build_dataset
from opencood.utils import eval_utils, common_utils
from opencood.visualization import vis_utils


# -------------------------- helpers -------------------------- #

def build_parser():
    parser = argparse.ArgumentParser(
        description="Chunk-level inference analysis for OpenCOOD"
    )
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Path to trained model directory')
    parser.add_argument('--fusion_method', type=str, required=True,
                        choices=['late', 'early', 'intermediate'],
                        help='Fusion method')

    # Prefer providing hypes explicitly; otherwise we try to infer from model_dir
    parser.add_argument('--hypes_yaml', type=str, default=None,
                        help='Path to hypes yaml (optional). If not provided, will try to load from model_dir.')

    parser.add_argument('--chunk_size', type=int, default=100,
                        help='Number of frames per chunk')
    parser.add_argument('--show_sequence', action='store_true',
                        help='Open3D sequence visualization')
    parser.add_argument('--save_npy', action='store_true',
                        help='Save prediction (boxes+scores) and gt as numpy arrays')  # updated help
    parser.add_argument('--global_sort_detections', action='store_true',
                        help='Globally sort detections when computing AP')

    parser.add_argument('--target_chunk', type=str, default=None,
                        help='Chunk ID to debug (e.g. chunk_0011). '
                             'If set, we will print more info and optionally save vis.')

    # Vis / debug selection
    parser.add_argument('--save_failed_vis', action='store_true',
                        help='Save BEV visualization for selected frames in target_chunk.')
    parser.add_argument('--failed_iou_thresh', type=float, default=0.5,
                        help='IoU threshold for saving failed visualizations (only used if --save_worst_k=0).')
    parser.add_argument('--save_worst_k', type=int, default=0,
                        help='If >0, save only worst-K frames (lowest IoU) within target_chunk. '
                             'Overrides failed_iou_thresh filtering.')

    parser.add_argument('--dump_debug_json', action='store_true',
                        help='Save per-frame debug JSON alongside saved PNG.')
    return parser


def find_hypes_yaml(model_dir: str) -> str:
    candidates = [
        os.path.join(model_dir, "config.yaml"),
        os.path.join(model_dir, "config.yml"),
        os.path.join(model_dir, "hypes.yaml"),
        os.path.join(model_dir, "hypes.yml"),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(
        "Could not find a hypes/config yaml in model_dir. "
        "Please pass --hypes_yaml explicitly."
    )


def get_chunk_id(global_sample_idx: int, chunk_size: int) -> str:
    """
    Map global sample index to chunk ID.
    Example: chunk_size=100 -> 0-99: chunk_0000, 100-199: chunk_0001, ...
    """
    chunk_index = global_sample_idx // chunk_size
    return f"chunk_{chunk_index:04d}"


def to_list(x):
    """Safely convert torch/numpy to Python lists for JSON."""
    if x is None:
        return None
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


def compute_sample_mean_iou(pred_box_tensor, gt_box_tensor):
    """
    Compute mean IoU for a single sample (diagnostic).
    - For each predicted polygon, compute IoU vs all GT polygons, take max
    - Return mean across predictions
    Returns None if not computable.
    """
    if (pred_box_tensor is None) or (gt_box_tensor is None):
        return None

    if hasattr(pred_box_tensor, "numel") and pred_box_tensor.numel() == 0:
        return None
    if hasattr(gt_box_tensor, "numel") and gt_box_tensor.numel() == 0:
        return None

    det_boxes_np = common_utils.torch_tensor_to_numpy(pred_box_tensor)
    gt_boxes_np = common_utils.torch_tensor_to_numpy(gt_box_tensor)

    det_polygons = list(common_utils.convert_format(det_boxes_np))
    gt_polygons = list(common_utils.convert_format(gt_boxes_np))

    if len(det_polygons) == 0 or len(gt_polygons) == 0:
        return None

    sample_ious = []
    for det_poly in det_polygons:
        ious = common_utils.compute_iou(det_poly, gt_polygons)
        if len(ious) > 0:
            sample_ious.append(float(np.max(ious)))

    if len(sample_ious) == 0:
        return None

    return float(np.mean(sample_ious))


def count_boxes(box_tensor):
    """Robust box counting across possible shapes."""
    if box_tensor is None:
        return 0
    if not hasattr(box_tensor, "ndim"):
        return 0
    # common shapes: (N, 7) or (1, N, 7)
    if box_tensor.ndim == 2:
        return int(box_tensor.shape[0])
    if box_tensor.ndim >= 3:
        return int(box_tensor.shape[1])
    return 0


def try_get_pairwise_t_matrix(batch_data):
    """
    Try common locations for pairwise transform matrices.
    Different OpenCOOD datasets/pipelines may put it in:
      batch_data['ego']['pairwise_t_matrix'] or batch_data['pairwise_t_matrix']
    """
    try:
        if isinstance(batch_data, dict):
            if 'ego' in batch_data and isinstance(batch_data['ego'], dict):
                if 'pairwise_t_matrix' in batch_data['ego']:
                    return batch_data['ego']['pairwise_t_matrix']
            if 'pairwise_t_matrix' in batch_data:
                return batch_data['pairwise_t_matrix']
    except Exception:
        pass
    return None


def plot_loss_iou_timeline(indices, losses, ious, chunk_id, save_path):
    """
    Combined Loss + IoU timeline with twin y-axes.
    X-axis: global frame index
    Left Y: loss
    Right Y: IoU
    """
    fig, ax1 = plt.subplots(figsize=(10, 4))

    ax1.plot(indices, losses, label="Loss")
    ax1.set_xlabel("Global frame index")
    ax1.set_ylabel("Loss")
    ax1.tick_params(axis="y")

    ax2 = ax1.twinx()
    ax2.plot(indices, ious, linestyle="--", label="Mean IoU")
    ax2.set_ylabel("IoU")
    ax2.tick_params(axis="y")

    plt.title(f"{chunk_id}: Loss & IoU timeline")

    # combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


# -------------------------- main -------------------------- #

def main():
    opt = build_parser().parse_args()

    # Resolve hypes yaml
    hypes_path = opt.hypes_yaml if opt.hypes_yaml is not None else find_hypes_yaml(opt.model_dir)

    print('----------------- Building dataset -----------------')
    hypes = yaml_utils.load_yaml(hypes_path, opt)
    opencood_dataset = build_dataset(hypes, visualize=True, train=False)
    print(f"[INFO] {len(opencood_dataset)} samples found.")
    print(f"[INFO] Using hypes: {hypes_path}")

    data_loader = DataLoader(opencood_dataset,
                             batch_size=1,
                             num_workers=8,
                             collate_fn=opencood_dataset.collate_batch_test,
                             shuffle=False,
                             pin_memory=False,
                             drop_last=False)

    print('----------------- Creating model -----------------')
    model = train_utils.create_model(hypes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    print(f'----------------- Loading from {opt.model_dir} -----------------')
    _, model = train_utils.load_saved_model(opt.model_dir, model)
    model.eval()

    # loss function (same as training)
    criterion = train_utils.create_loss(hypes)

    # Chunk-level aggregations
    chunk_losses = defaultdict(list)        # chunk_id -> [loss_i]
    chunk_ious = defaultdict(list)          # chunk_id -> [mean_iou_i]
    chunk_counts = defaultdict(int)         # chunk_id -> num samples

    # Per-frame timelines inside chunks
    chunk_frame_ious = defaultdict(list)    # chunk_id -> [iou per frame]
    chunk_frame_losses = defaultdict(list)  # chunk_id -> [loss per frame]
    chunk_frame_indices = defaultdict(list) # chunk_id -> [global indices]

    # TensorBoard writer
    log_dir = os.path.join(opt.model_dir, "tb_chunk_analysis")
    os.makedirs(log_dir, exist_ok=True)
    print(f"[INFO] TensorBoard logs at: {log_dir}")
    writer = SummaryWriter(log_dir=log_dir)

    # Stats for AP
    result_stat = {
        0.3: {'tp': [], 'fp': [], 'gt': 0, 'score': []},
        0.5: {'tp': [], 'fp': [], 'gt': 0, 'score': []},
        0.7: {'tp': [], 'fp': [], 'gt': 0, 'score': []}
    }

    # Optional Open3D sequence vis
    if opt.show_sequence:
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        render_option = vis.get_render_option()
        render_option.background_color = [0.05, 0.05, 0.05]
        render_option.point_size = 1.0
        render_option.show_coordinate_frame = True
        vis_pcd = o3d.geometry.PointCloud()
        vis_aabbs_gt = [o3d.geometry.LineSet() for _ in range(50)]
        vis_aabbs_pred = [o3d.geometry.LineSet() for _ in range(50)]
    else:
        vis = vis_pcd = vis_aabbs_gt = vis_aabbs_pred = None

    # Directories for debug outputs
    debug_vis_dir = os.path.join(opt.model_dir, "chunk_debug_vis")
    debug_npy_dir = os.path.join(opt.model_dir, "chunk_debug_npy")
    os.makedirs(debug_vis_dir, exist_ok=True)
    os.makedirs(debug_npy_dir, exist_ok=True)

    # Store records for target chunk so we can pick worst-K (or apply threshold after)
    target_records = []

    # ------------------- INFERENCE LOOP -------------------
    for i, batch_data in tqdm(enumerate(data_loader), total=len(data_loader)):
        with torch.no_grad():
            batch_data = train_utils.to_device(batch_data, device)

            # chunk id based on global index
            chunk_id = get_chunk_id(i, opt.chunk_size)
            chunk_counts[chunk_id] += 1

            # ---------- LOSS PER SAMPLE ----------
            output_for_loss = model(batch_data['ego'])
            loss_value = criterion(output_for_loss,
                                   batch_data['ego']['label_dict'])
            loss_scalar = float(loss_value.item())

            chunk_losses[chunk_id].append(loss_scalar)
            chunk_frame_losses[chunk_id].append(loss_scalar)
            chunk_frame_indices[chunk_id].append(i)
            writer.add_scalar("chunk/loss_per_sample", loss_scalar, i)

            # ---------- DETECTION INFERENCE ----------
            if opt.fusion_method == 'late':
                pred_box_tensor, pred_score, gt_box_tensor = \
                    inference_utils.inference_late_fusion(
                        batch_data, model, opencood_dataset
                    )
            elif opt.fusion_method == 'early':
                pred_box_tensor, pred_score, gt_box_tensor = \
                    inference_utils.inference_early_fusion(
                        batch_data, model, opencood_dataset
                    )
            elif opt.fusion_method == 'intermediate':
                pred_box_tensor, pred_score, gt_box_tensor = \
                    inference_utils.inference_intermediate_fusion(
                        batch_data, model, opencood_dataset
                    )
            else:
                raise NotImplementedError

            # ---------- PER-SAMPLE MEAN IoU ----------
            mean_iou_sample = compute_sample_mean_iou(
                pred_box_tensor, gt_box_tensor
            )

            if mean_iou_sample is not None:
                chunk_ious[chunk_id].append(mean_iou_sample)
                chunk_frame_ious[chunk_id].append(mean_iou_sample)
                writer.add_scalar("chunk/mean_iou_sample", mean_iou_sample, i)
            else:
                chunk_frame_ious[chunk_id].append(np.nan)

            # ---------- TP/FP STATS FOR AP ----------
            eval_utils.caluclate_tp_fp(pred_box_tensor,
                                       pred_score,
                                       gt_box_tensor,
                                       result_stat,
                                       0.3)
            eval_utils.caluclate_tp_fp(pred_box_tensor,
                                       pred_score,
                                       gt_box_tensor,
                                       result_stat,
                                       0.5)
            eval_utils.caluclate_tp_fp(pred_box_tensor,
                                       pred_score,
                                       gt_box_tensor,
                                       result_stat,
                                       0.7)

            # ---------- SIMPLE LOGGING ----------
            num_pred = count_boxes(pred_box_tensor)
            num_gt = count_boxes(gt_box_tensor)
            writer.add_scalar("chunk/num_pred_boxes", num_pred, i)
            writer.add_scalar("chunk/num_gt_boxes", num_gt, i)

            # Print detailed info only for target_chunk (if set)
            if opt.target_chunk is None or chunk_id == opt.target_chunk:
                print(
                    f"[Sample {i:05d}] "
                    f"chunk={chunk_id} | "
                    f"loss={loss_scalar:.4f} | "
                    f"mean_IoU={mean_iou_sample if mean_iou_sample is not None else 'N/A'} | "
                    f"#pred={num_pred} | #gt={num_gt}"
                )

            # ---------- SAVE NPY (optional) ----------
            if opt.save_npy and (opt.target_chunk is None or chunk_id == opt.target_chunk):
                np.save(os.path.join(debug_npy_dir, f"{chunk_id}_pred_{i:05d}.npy"),
                        pred_box_tensor.detach().cpu().numpy() if pred_box_tensor is not None else None)

                # ADDED: save scores too
                np.save(os.path.join(debug_npy_dir, f"{chunk_id}_score_{i:05d}.npy"),
                        pred_score.detach().cpu().numpy() if pred_score is not None else None)

                np.save(os.path.join(debug_npy_dir, f"{chunk_id}_gt_{i:05d}.npy"),
                        gt_box_tensor.detach().cpu().numpy() if gt_box_tensor is not None else None)

            # ---------- COLLECT TARGET CHUNK RECORDS (for worst-K / threshold saves) ----------
            if opt.target_chunk is not None and chunk_id == opt.target_chunk:
                rec = {
                    "i": int(i),
                    "iou": float(mean_iou_sample) if mean_iou_sample is not None else np.nan,
                    "loss": float(loss_scalar),
                    "pred_box_tensor": pred_box_tensor.detach().cpu() if pred_box_tensor is not None else None,
                    "pred_score": pred_score.detach().cpu() if pred_score is not None else None,
                    "gt_box_tensor": gt_box_tensor.detach().cpu() if gt_box_tensor is not None else None,
                    "origin_lidar": batch_data['ego']['origin_lidar'].detach().cpu(),
                    "pairwise_t_matrix": try_get_pairwise_t_matrix(batch_data),
                }
                if rec["pairwise_t_matrix"] is not None and hasattr(rec["pairwise_t_matrix"], "detach"):
                    rec["pairwise_t_matrix"] = rec["pairwise_t_matrix"].detach().cpu()
                target_records.append(rec)

            # ---------- OPEN3D SEQUENCE VIS (optional) ----------
            if opt.show_sequence:
                pcd, pred_o3d_box, gt_o3d_box = \
                    vis_utils.visualize_inference_sample_dataloader(
                        pred_box_tensor,
                        gt_box_tensor,
                        batch_data['ego']['origin_lidar'],
                        vis_pcd,
                        mode='constant'
                    )
                if i == 0:
                    vis.add_geometry(pcd)
                    vis_utils.linset_assign_list(
                        vis, vis_aabbs_pred, pred_o3d_box, update_mode='add'
                    )
                    vis_utils.linset_assign_list(
                        vis, vis_aabbs_gt, gt_o3d_box, update_mode='add'
                    )

                vis_utils.linset_assign_list(vis, vis_aabbs_pred, pred_o3d_box)
                vis_utils.linset_assign_list(vis, vis_aabbs_gt, gt_o3d_box)
                vis.update_geometry(pcd)
                vis.poll_events()
                vis.update_renderer()
                time.sleep(0.001)

    # ------------------- SAVE SELECTED FAILED VIS FOR TARGET CHUNK -------------------
    if opt.target_chunk is not None and opt.save_failed_vis:
        valid = [r for r in target_records if not np.isnan(r["iou"])]

        if len(valid) == 0:
            print(f"[WARN] No valid IoU values found in {opt.target_chunk}; nothing to save.")
        else:
            if opt.save_worst_k and opt.save_worst_k > 0:
                chosen = sorted(valid, key=lambda r: r["iou"])[:opt.save_worst_k]
                print(f"[INFO] Saving worst-{opt.save_worst_k} frames (lowest IoU) in {opt.target_chunk}")
            else:
                chosen = [r for r in valid if r["iou"] < opt.failed_iou_thresh]
                print(f"[INFO] Saving frames in {opt.target_chunk} with IoU < {opt.failed_iou_thresh}")

            if len(chosen) == 0:
                print(f"[INFO] No frames matched the selection criteria in {opt.target_chunk}.")
            else:
                for r in chosen:
                    gi = r["i"]
                    vis_save_path = os.path.join(debug_vis_dir, f"{opt.target_chunk}_frame_{gi:05d}.png")

                    # visualize_result works fine with CPU tensors
                    opencood_dataset.visualize_result(
                        r["pred_box_tensor"],
                        r["gt_box_tensor"],
                        r["origin_lidar"],
                        show_vis=False,
                        save_path=vis_save_path,
                        dataset=opencood_dataset
                    )
                    print(f"[DEBUG] Saved failed vis: {vis_save_path}")

                    if opt.dump_debug_json:
                        dbg = {
                            "chunk": opt.target_chunk,
                            "global_frame": int(gi),
                            "iou": float(r["iou"]),
                            "loss": float(r["loss"]),
                            "num_pred": int(count_boxes(r["pred_box_tensor"])),
                            "num_gt": int(count_boxes(r["gt_box_tensor"])),
                            "pairwise_t_matrix": to_list(r["pairwise_t_matrix"]),
                            "fused_pred_boxes": to_list(r["pred_box_tensor"]),
                            "fused_pred_scores": to_list(r["pred_score"]),
                            "gt_boxes": to_list(r["gt_box_tensor"]),
                        }
                        json_path = vis_save_path.replace(".png", ".json")
                        with open(json_path, "w") as f:
                            json.dump(dbg, f, indent=2)
                        print(f"[DEBUG] Saved debug JSON: {json_path}")

    # ------------------- FINAL EVAL -------------------
    try:
        # newer forks may accept a third arg
        eval_utils.eval_final_results(result_stat,
                                      opt.model_dir,
                                      opt.global_sort_detections)
    except TypeError:
        # your current OpenCOOD expects only (result_stat, model_dir)
        eval_utils.eval_final_results(result_stat, opt.model_dir)

    # ------------------- SUMMARY PER CHUNK -------------------
    chunks = sorted(chunk_counts.keys())

    mean_loss = {}
    mean_iou = {}
    min_iou = {}
    max_iou = {}

    for c in chunks:
        losses = np.array(chunk_losses[c], dtype=float) if chunk_losses[c] else np.array([np.nan])
        ious = np.array(chunk_ious[c], dtype=float) if chunk_ious[c] else np.array([np.nan])

        mean_loss[c] = float(np.nanmean(losses))
        mean_iou[c] = float(np.nanmean(ious))
        min_iou[c] = float(np.nanmin(ious)) if not np.isnan(ious).all() else np.nan
        max_iou[c] = float(np.nanmax(ious)) if not np.isnan(ious).all() else np.nan

    # CSV report
    csv_path = os.path.join(opt.model_dir, "chunk_report.csv")
    with open(csv_path, "w", newline="") as f:
        writer_csv = csv.writer(f)
        writer_csv.writerow(["chunk_id", "num_samples",
                             "mean_loss", "mean_iou",
                             "min_iou", "max_iou"])
        for c in chunks:
            writer_csv.writerow([
                c,
                chunk_counts[c],
                mean_loss[c],
                mean_iou[c],
                min_iou[c],
                max_iou[c]
            ])
    print(f"[INFO] Chunk report saved to: {csv_path}")

    # Global bar plots over chunks
    if chunks:
        # Loss bar
        losses_plot = [mean_loss[c] for c in chunks]
        plt.figure(figsize=(max(10, len(chunks) * 0.5), 5))
        plt.bar(range(len(chunks)), losses_plot)
        plt.xticks(range(len(chunks)), chunks, rotation=90)
        plt.ylabel("Mean Loss")
        plt.title("Per-chunk Loss During Inference")
        plt.tight_layout()
        save_path_loss = os.path.join(opt.model_dir, "chunk_loss_overview.png")
        plt.savefig(save_path_loss, dpi=250)
        plt.close()
        print(f"[INFO] Chunk loss overview saved to: {save_path_loss}")

        # IoU bar
        ious_plot = [mean_iou[c] for c in chunks]
        plt.figure(figsize=(max(10, len(chunks) * 0.5), 5))
        plt.bar(range(len(chunks)), ious_plot)
        plt.xticks(range(len(chunks)), chunks, rotation=90)
        plt.ylabel("Mean IoU")
        plt.title("Per-chunk Mean IoU During Inference (diagnostic)")
        plt.tight_layout()
        save_path_iou = os.path.join(opt.model_dir, "chunk_iou_overview.png")
        plt.savefig(save_path_iou, dpi=250)
        plt.close()
        print(f"[INFO] Chunk IoU overview saved to: {save_path_iou}")

    # Per-chunk timelines (COMBINED LOSS + IOU)  <-- UPDATED
    for c in chunks:
        indices = chunk_frame_indices[c]
        ious = chunk_frame_ious[c]
        losses = chunk_frame_losses[c]

        path_combined = os.path.join(opt.model_dir, f"{c}_loss_iou_timeline.png")
        plot_loss_iou_timeline(indices, losses, ious, c, path_combined)

    # Log chunk-level aggregates to TB
    for idx, c in enumerate(chunks):
        writer.add_scalar("chunk/mean_loss", mean_loss[c], idx)
        if not np.isnan(mean_iou[c]):
            writer.add_scalar("chunk/mean_iou", mean_iou[c], idx)
        writer.add_text("chunk/chunk_id", f"{idx}: {c}", idx)

    if opt.show_sequence and vis is not None:
        vis.destroy_window()

    writer.close()
    print("[INFO] Chunk analysis finished.")


if __name__ == "__main__":
    main()
