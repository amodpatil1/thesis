# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>, Hao Xiang <haxiang@g.ucla.edu>, Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib
# Extended: per-CAV saving for late fusion (CloudCompare workflow)

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
import matplotlib.pyplot as plt
matplotlib.use("Agg")

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils, inference_utils
from opencood.data_utils.datasets import build_dataset
from opencood.utils import eval_utils, common_utils
from opencood.visualization import vis_utils


def test_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Continued training path')
    parser.add_argument('--fusion_method', required=True, type=str,
                        default='late',
                        choices=['late', 'early', 'intermediate'],
                        help='late, early or intermediate')
    parser.add_argument('--show_vis', action='store_true',
                        help='whether to show image visualization result')
    parser.add_argument('--show_sequence', action='store_true',
                        help='whether to show video visualization result.'
                             'it can note be set true with show_vis together ')
    parser.add_argument('--save_vis', action='store_true',
                        help='whether to save visualization result')
    parser.add_argument('--save_npy', action='store_true',
                        help='whether to save prediction and gt result in npy files')
    parser.add_argument('--global_sort_detections', action='store_true',
                        help='whether to globally sort detections by confidence score')
    opt = parser.parse_args()
    return opt


def extract_scene_id(batch_data, batch_idx, dataset=None):
    ego = batch_data.get('ego', {})
    if isinstance(ego, dict):
        path_keys = ['lidar_path', 'pcd_path', 'yaml_path', 'file_path', 'filename', 'path']
        for k in path_keys:
            if k in ego:
                path_val = ego[k]
                if isinstance(path_val, (list, tuple)):
                    path_val = path_val[0]
                if isinstance(path_val, str):
                    scene_dir = os.path.dirname(path_val)
                    scene_root = os.path.dirname(scene_dir)
                    return os.path.basename(scene_root)

    if dataset is not None:
        for attr in ['data_list', 'scenario_database', 'opv2v_database']:
            if hasattr(dataset, attr):
                db = getattr(dataset, attr)
                try:
                    entry = db[batch_idx]
                except Exception:
                    entry = None

                if isinstance(entry, dict):
                    for k in ['lidar_path', 'pcd_path', 'yaml_path', 'path', 'file_path']:
                        if k in entry and isinstance(entry[k], str):
                            scene_dir = os.path.dirname(entry[k])
                            scene_root = os.path.dirname(scene_dir)
                            return os.path.basename(scene_root)

                if isinstance(db, (list, tuple)) and isinstance(entry, str):
                    scene_dir = os.path.dirname(entry)
                    scene_root = os.path.dirname(scene_dir)
                    return os.path.basename(scene_root)

    chunk = batch_idx // 100
    return f"chunk_{chunk:04d}"


def main():
    opt = test_parser()
    assert opt.fusion_method in ['late', 'early', 'intermediate']
    assert not (opt.show_vis and opt.show_sequence), \
        'you can only visualize the results in single image mode or video mode'

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

    result_stat = {
        0.3: {'tp': [], 'fp': [], 'gt': 0, 'score': []},
        0.5: {'tp': [], 'fp': [], 'gt': 0, 'score': []},
        0.7: {'tp': [], 'fp': [], 'gt': 0, 'score': []}
    }

    if opt.show_sequence:
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        ro = vis.get_render_option()
        ro.background_color = [0.05, 0.05, 0.05]
        ro.point_size = 1.0
        ro.show_coordinate_frame = True
        vis_pcd = o3d.geometry.PointCloud()
        vis_aabbs_gt = [o3d.geometry.LineSet() for _ in range(50)]
        vis_aabbs_pred = [o3d.geometry.LineSet() for _ in range(50)]

    seen_scenes = set()
    prev_scene = None

    # ---------- INFERENCE LOOP ----------
    for i, batch_data in tqdm(enumerate(data_loader)):
        with torch.no_grad():
            batch_data = train_utils.to_device(batch_data, device)

            scene_id = extract_scene_id(batch_data, i, dataset=opencood_dataset)
            scene_counts[scene_id] += 1
            if scene_id not in seen_scenes:
                print(f"[SCENE] New scene encountered: {scene_id}")
                seen_scenes.add(scene_id)
            if scene_id != prev_scene:
                writer.add_text("inference/scene_change",
                                f"step {i}: scene {scene_id}", i)
                prev_scene = scene_id

            # ---------- LOSS ----------
            output_for_loss = model(batch_data['ego'])
            loss_value = criterion(output_for_loss, batch_data['ego']['label_dict'])
            loss_scalar = loss_value.item()
            scene_losses[scene_id].append(loss_scalar)
            writer.add_scalar("inference/loss", loss_scalar, i)

            # ---------- DETECTION INFERENCE ----------
            per_cav = None

            if opt.fusion_method == 'late':
                ret = inference_utils.inference_late_fusion(batch_data, model, opencood_dataset)
                if len(ret) == 3:
                    pred_box_tensor, pred_score, gt_box_tensor = ret
                    per_cav = None
                else:
                    pred_box_tensor, pred_score, gt_box_tensor, per_cav = ret

            elif opt.fusion_method == 'early':
                pred_box_tensor, pred_score, gt_box_tensor = \
                    inference_utils.inference_early_fusion(batch_data, model, opencood_dataset)

            elif opt.fusion_method == 'intermediate':
                pred_box_tensor, pred_score, gt_box_tensor = \
                    inference_utils.inference_intermediate_fusion(batch_data, model, opencood_dataset)

            else:
                raise NotImplementedError('Only early, late and intermediate fusion is supported.')

            # ---------- PER-SAMPLE MEAN IoU (diagnostic) ----------
            mean_iou_sample = None
            if (pred_box_tensor is not None) and (gt_box_tensor is not None) \
                    and pred_box_tensor.shape[0] > 0 and gt_box_tensor.shape[0] > 0:

                det_boxes_np = common_utils.torch_tensor_to_numpy(pred_box_tensor)
                gt_boxes_np = common_utils.torch_tensor_to_numpy(gt_box_tensor)

                det_polygons = list(common_utils.convert_format(det_boxes_np))
                gt_polygons = list(common_utils.convert_format(gt_boxes_np))

                sample_ious = []
                for det_poly in det_polygons:
                    if len(gt_polygons) == 0:
                        break
                    ious = common_utils.compute_iou(det_poly, gt_polygons)
                    if len(ious) > 0:
                        sample_ious.append(float(np.max(ious)))

                if len(sample_ious) > 0:
                    mean_iou_sample = float(np.mean(sample_ious))
                    scene_ious[scene_id].append(mean_iou_sample)
                    writer.add_scalar("inference/mean_iou_sample", mean_iou_sample, i)

            # ---------- TP/FP STATS ----------
            for thr in (0.3, 0.5, 0.7):
                eval_utils.caluclate_tp_fp(pred_box_tensor, pred_score, gt_box_tensor,
                                           result_stat, thr)

            # ---------- NUM BOXES ----------
            num_pred = int(pred_box_tensor.shape[0]) if pred_box_tensor is not None else 0
            num_gt = int(gt_box_tensor.shape[0]) if gt_box_tensor is not None else 0
            writer.add_scalar("inference/num_pred_boxes", num_pred, i)
            writer.add_scalar("inference/num_gt_boxes", num_gt, i)

            if mean_iou_sample is not None:
                print(f"[Sample {i:05d}] scene={scene_id} | loss={loss_scalar:.4f} | "
                      f"mean_IoU={mean_iou_sample:.4f} | #pred={num_pred} | #gt={num_gt}")
            else:
                print(f"[Sample {i:05d}] scene={scene_id} | loss={loss_scalar:.4f} | "
                      f"mean_IoU=N/A | #pred={num_pred} | #gt={num_gt}")

            # ---------- SAVE NPY ----------
            if opt.save_npy:
                fused_dir = os.path.join(opt.model_dir, 'npy')
                os.makedirs(fused_dir, exist_ok=True)

                inference_utils.save_prediction_gt(
                    pred_box_tensor,
                    gt_box_tensor,
                    batch_data['ego']['origin_lidar'][0],
                    i,
                    fused_dir
                )

                # per-CAV saving (late only, and only if available)
                if opt.fusion_method == 'late' and per_cav is not None:
                    per_cav_dir = os.path.join(opt.model_dir, 'npy_per_cav')
                    os.makedirs(per_cav_dir, exist_ok=True)

                    for cav_id, (cav_pred_box, cav_pred_score) in per_cav.items():
                        safe_id = str(cav_id).replace("/", "_")
                        np.save(os.path.join(per_cav_dir, f"{i:04d}_cav{safe_id}_pred.npy"),
                                cav_pred_box.detach().cpu().numpy())

            # ---------- VIS ----------
            if opt.show_vis or opt.save_vis:
                vis_save_path = ''
                if opt.save_vis:
                    vis_dir = os.path.join(opt.model_dir, 'vis')
                    os.makedirs(vis_dir, exist_ok=True)
                    vis_save_path = os.path.join(vis_dir, f'{i:05d}.png')

                opencood_dataset.visualize_result(
                    pred_box_tensor,
                    gt_box_tensor,
                    batch_data['ego']['origin_lidar'],
                    opt.show_vis,
                    vis_save_path,
                    dataset=opencood_dataset
                )

            # ---------- SEQUENCE VIS ----------
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
                    vis_utils.linset_assign_list(vis, vis_aabbs_pred, pred_o3d_box, update_mode='add')
                    vis_utils.linset_assign_list(vis, vis_aabbs_gt, gt_o3d_box, update_mode='add')

                vis_utils.linset_assign_list(vis, vis_aabbs_pred, pred_o3d_box)
                vis_utils.linset_assign_list(vis, vis_aabbs_gt, gt_o3d_box)
                vis.update_geometry(pcd)
                vis.poll_events()
                vis.update_renderer()
                time.sleep(0.001)

    # ---------- FINAL EVAL ----------
    eval_utils.eval_final_results(result_stat, opt.model_dir, opt.global_sort_detections)

    # ---------- PER-SCENE METRICS CSV ----------
    scenes = sorted(scene_counts.keys())

    mean_loss = {s: float(np.mean(scene_losses[s])) if scene_losses[s] else float('nan') for s in scenes}
    mean_iou = {s: float(np.mean(scene_ious[s])) if scene_ious[s] else float('nan') for s in scenes}

    csv_path = os.path.join(opt.model_dir, "scene_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scene_id", "num_samples", "mean_loss", "mean_iou"])
        for s in scenes:
            w.writerow([s, scene_counts[s], mean_loss[s], mean_iou[s]])
    print(f"[INFO] Scene metrics saved to: {csv_path}")

    # ---------- PLOTS ----------
    if scenes:
        losses_plot = [mean_loss[s] for s in scenes]
        plt.figure(figsize=(max(10, len(scenes) * 0.4), 5))
        plt.bar(range(len(scenes)), losses_plot)
        plt.xticks(range(len(scenes)), scenes, rotation=90)
        plt.ylabel("Mean Loss")
        plt.title("Per-scene Loss During Inference")
        plt.tight_layout()
        plt.savefig(os.path.join(opt.model_dir, "scene_loss_analysis.png"), dpi=250)
        plt.close()

        ious_plot = [mean_iou[s] for s in scenes]
        plt.figure(figsize=(max(10, len(scenes) * 0.4), 5))
        plt.bar(range(len(scenes)), ious_plot)
        plt.xticks(range(len(scenes)), scenes, rotation=90)
        plt.ylabel("Mean IoU")
        plt.title("Per-scene Mean IoU During Inference (diagnostic)")
        plt.tight_layout()
        plt.savefig(os.path.join(opt.model_dir, "scene_iou_analysis.png"), dpi=250)
        plt.close()

        for idx, s in enumerate(scenes):
            writer.add_scalar("inference/scene_mean_loss", mean_loss[s], idx)
            if not np.isnan(mean_iou[s]):
                writer.add_scalar("inference/scene_mean_iou", mean_iou[s], idx)
            writer.add_text("inference/scene_id", f"{idx}: {s}", idx)

    if opt.show_sequence:
        vis.destroy_window()

    writer.close()
    print(f"TensorBoard writer closed. Logs at: {log_dir}")


if __name__ == '__main__':
    main()
