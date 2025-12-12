# -*- coding: utf-8 -*-
# Chunk-level inference analysis for OpenCOOD
#
# Drop this file into: opencood/tools/chunk_analysis.py
# Run e.g.:
#   cd /OpenCOOD
#   python opencood/tools/chunk_analysis.py \
#       --model_dir opencood/trained/pointpillar_attentive_fusion \
#       --fusion_method intermediate \
#       --chunk_size 100 \
#       --target_chunk chunk_0011 \
#       --save_failed_vis \
#       --failed_iou_thresh 0.5

import argparse
import os
import time
from collections import defaultdict
import csv

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

import torch
import open3d as o3d
from torch.utils.data import DataLoader

import numpy as np
import matplotlib
matplotlib.use("Agg")   # important for headless servers
import matplotlib.pyplot as plt

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils, inference_utils
from opencood.data_utils.datasets import build_dataset
from opencood.utils import eval_utils, common_utils
from opencood.visualization import vis_utils


def build_parser():
    parser = argparse.ArgumentParser(
        description="Chunk-level inference analysis for OpenCOOD"
    )
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Path to trained model directory')
    parser.add_argument('--fusion_method', type=str, required=True,
                        choices=['late', 'early', 'intermediate'],
                        help='Fusion method')
    parser.add_argument('--chunk_size', type=int, default=100,
                        help='Number of frames per chunk')
    parser.add_argument('--show_sequence', action='store_true',
                        help='Open3D sequence visualization')
    parser.add_argument('--save_npy', action='store_true',
                        help='Save prediction and gt as numpy arrays')
    parser.add_argument('--global_sort_detections', action='store_true',
                        help='Globally sort detections when computing AP')
    parser.add_argument('--target_chunk', type=str, default=None,
                        help='Chunk ID to debug (e.g. chunk_0011). '
                             'If set, we will print more info and optionally save vis.')
    parser.add_argument('--save_failed_vis', action='store_true',
                        help='Save BEV visualization for frames in target_chunk '
                             'with IoU below threshold.')
    parser.add_argument('--failed_iou_thresh', type=float, default=0.5,
                        help='IoU threshold for saving failed visualizations')
    return parser


def get_chunk_id(batch_idx: int, chunk_size: int) -> str:
    """
    Map global sample index to chunk ID.
    Example: chunk_size=100 -> 0-99: chunk_0000, 100-199: chunk_0001, ...
    """
    chunk_index = batch_idx // chunk_size
    return f"chunk_{chunk_index:04d}"


def compute_sample_mean_iou(pred_box_tensor, gt_box_tensor):
    """
    Compute mean IoU for a single sample (diagnostic).
    Returns None if not computable.
    """
    if (pred_box_tensor is None) or (gt_box_tensor is None):
        return None

    if pred_box_tensor.numel() == 0 or gt_box_tensor.numel() == 0:
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


def main():
    opt = build_parser().parse_args()

    print('----------------- Building dataset -----------------')
    hypes = yaml_utils.load_yaml(None, opt)
    opencood_dataset = build_dataset(hypes, visualize=True, train=False)
    print(f"[INFO] {len(opencood_dataset)} samples found.")

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
    if torch.cuda.is_available():
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
                writer.add_scalar("chunk/mean_iou_sample",
                                  mean_iou_sample, i)
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
            num_pred = 0
            num_gt = 0
            if pred_box_tensor is not None:
                num_pred = pred_box_tensor.shape[1] if pred_box_tensor.ndim >= 2 else pred_box_tensor.shape[0]
            if gt_box_tensor is not None:
                num_gt = gt_box_tensor.shape[1] if gt_box_tensor.ndim >= 2 else gt_box_tensor.shape[0]

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
                        pred_box_tensor.cpu().numpy() if pred_box_tensor is not None else None)
                np.save(os.path.join(debug_npy_dir, f"{chunk_id}_gt_{i:05d}.npy"),
                        gt_box_tensor.cpu().numpy() if gt_box_tensor is not None else None)

            # ---------- SAVE FAILED VIS (optional) ----------
            if (opt.target_chunk is not None
                and chunk_id == opt.target_chunk
                and opt.save_failed_vis
                and mean_iou_sample is not None
                and mean_iou_sample < opt.failed_iou_thresh):

                vis_save_path = os.path.join(
                    debug_vis_dir, f"{chunk_id}_frame_{i:05d}.png"
                )
                opencood_dataset.visualize_result(
                    pred_box_tensor,
                    gt_box_tensor,
                    batch_data['ego']['origin_lidar'],
                    show_vis=False,
                    save_path=vis_save_path,
                    dataset=opencood_dataset
                )
                print(f"[DEBUG] Saved failed vis: {vis_save_path}")

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

    # ------------------- FINAL EVAL -------------------
    eval_utils.eval_final_results(result_stat,
                                  opt.model_dir,
                                  opt.global_sort_detections)

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

    # Per-chunk timelines
    for c in chunks:
        indices = chunk_frame_indices[c]
        ious = chunk_frame_ious[c]
        losses = chunk_frame_losses[c]

        # IoU timeline
        plt.figure(figsize=(10, 4))
        plt.plot(indices, ious, marker='o', linewidth=1)
        plt.xlabel("Global frame index")
        plt.ylabel("IoU")
        plt.title(f"IoU timeline for {c}")
        plt.tight_layout()
        path_timeline_iou = os.path.join(opt.model_dir, f"{c}_iou_timeline.png")
        plt.savefig(path_timeline_iou, dpi=200)
        plt.close()

        # Loss timeline
        plt.figure(figsize=(10, 4))
        plt.plot(indices, losses, marker='o', linewidth=1)
        plt.xlabel("Global frame index")
        plt.ylabel("Loss")
        plt.title(f"Loss timeline for {c}")
        plt.tight_layout()
        path_timeline_loss = os.path.join(opt.model_dir, f"{c}_loss_timeline.png")
        plt.savefig(path_timeline_loss, dpi=200)
        plt.close()

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
