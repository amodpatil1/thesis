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
                        help='late, early or intermediate')
    parser.add_argument('--show_vis', action='store_true',
                        help='whether to show image visualization result')
    parser.add_argument('--show_sequence', action='store_true',
                        help='whether to show video visualization result.'
                             'it can note be set true with show_vis together ')
    parser.add_argument('--save_vis', action='store_true',
                        help='whether to save visualization result')
    parser.add_argument('--save_npy', action='store_true',
                        help='whether to save prediction and gt result'
                             'in npy_test file')
    parser.add_argument('--global_sort_detections', action='store_true',
                        help='whether to globally sort detections by confidence score.'
                             'If set to True, it is the mainstream AP computing method,'
                             'but would increase the tolerance for FP (False Positives).')
    opt = parser.parse_args()
    return opt


def extract_scene_id(batch_data, batch_idx, dataset=None):
    """
    Try to get a stable scene identifier.

    Priority:
    1) Use any file path stored in batch_data['ego'] (preferred).
    2) Use dataset meta (dataset.data_list or dataset.opv2v_database).
    3) Fall back to a synthetic chunk name based on batch_idx.
    """
    # --- 1) Try to use a path stored inside batch_data['ego'] ---
    ego = batch_data.get('ego', {})
    if isinstance(ego, dict):
        # common key names that might store paths
        path_keys = [
            'lidar_path',
            'pcd_path',
            'yaml_path',
            'file_path',
            'filename',
            'path'
        ]
        for k in path_keys:
            if k in ego:
                path_val = ego[k]
                # often stored as list of length 1
                if isinstance(path_val, (list, tuple)):
                    path_val = path_val[0]
                # sometimes tensor of strings is impossible, so focus on str
                if isinstance(path_val, str):
                    # e.g. .../train/testoutput_CAV_data_2022-03-15-09-54-40_0/0/000120.yaml
                    # scene name := parent folder of the frame index folder
                    scene_dir = os.path.dirname(path_val)          # .../0
                    scene_root = os.path.dirname(scene_dir)        # .../testoutput_CAV_data_2022-03-15-09-54-40_0
                    scene_name = os.path.basename(scene_root)      # testoutput_CAV_data_2022-03-15-09-54-40_0
                    return scene_name

    # --- 2) Try to infer from the dataset object, if provided ---
    if dataset is not None:
        # Many OpenCOOD datasets store a list of file entries.
        # Try some common attribute names:
        for attr in ['data_list', 'scenario_database', 'opv2v_database']:
            if hasattr(dataset, attr):
                db = getattr(dataset, attr)
                try:
                    entry = db[batch_idx]
                except Exception:
                    entry = None

                if isinstance(entry, dict):
                    # Try to find a path inside the entry
                    for k in ['lidar_path', 'pcd_path', 'yaml_path', 'path', 'file_path']:
                        if k in entry:
                            path_val = entry[k]
                            if isinstance(path_val, str):
                                scene_dir = os.path.dirname(path_val)
                                scene_root = os.path.dirname(scene_dir)
                                scene_name = os.path.basename(scene_root)
                                return scene_name

                # If entries are raw strings (direct paths)
                if isinstance(db, (list, tuple)) and isinstance(entry, str):
                    path_val = entry
                    scene_dir = os.path.dirname(path_val)
                    scene_root = os.path.dirname(scene_dir)
                    scene_name = os.path.basename(scene_root)
                    return scene_name

    # --- 3) Fallback: group by chunks of 100 frames ---
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

    # loss function (same as training)
    criterion = train_utils.create_loss(hypes)

    # per-scene aggregation
    scene_losses = defaultdict(list)   # scene_id -> [loss_i]
    scene_ious = defaultdict(list)     # scene_id -> [mean_iou_i]
    scene_counts = defaultdict(int)    # scene_id -> num samples

    # TensorBoard writer
    log_dir = os.path.join(opt.model_dir, "tb_inference")
    print(f"TensorBoard logs will be written to: {log_dir}")
    writer = SummaryWriter(log_dir=log_dir)

    # eval stats
    result_stat = {
        0.3: {'tp': [], 'fp': [], 'gt': 0, 'score': []},
        0.5: {'tp': [], 'fp': [], 'gt': 0, 'score': []},
        0.7: {'tp': [], 'fp': [], 'gt': 0, 'score': []}
    }

    # sequence visualizer
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

    seen_scenes = set()
    prev_scene = None

    # ---------- INFERENCE LOOP ----------
    for i, batch_data in tqdm(enumerate(data_loader)):
        with torch.no_grad():
            batch_data = train_utils.to_device(batch_data, device)

            # get scene id
            scene_id = extract_scene_id(batch_data, i)
            scene_counts[scene_id] += 1
            if scene_id not in seen_scenes:
                print(f"[SCENE] New scene encountered: {scene_id}")
                seen_scenes.add(scene_id)
            if scene_id != prev_scene:
                # log scene change to TB as text
                writer.add_text("inference/scene_change",
                                f"step {i}: scene {scene_id}", i)
                prev_scene = scene_id

            # ---------- LOSS PER SAMPLE ----------
            output_for_loss = model(batch_data['ego'])
            loss_value = criterion(output_for_loss,
                                   batch_data['ego']['label_dict'])
            loss_scalar = loss_value.item()
            scene_losses[scene_id].append(loss_scalar)
            writer.add_scalar("inference/loss", loss_scalar, i)

            # ---------- DETECTION INFERENCE ----------
            if opt.fusion_method == 'late':
            pred_box_tensor, pred_score, gt_box_tensor, per_cav = \
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
                raise NotImplementedError(
                    'Only early, late and intermediate fusion is supported.'
                )

            # ---------- PER-SAMPLE MEAN IoU (diagnostic) ----------
            mean_iou_sample = None
            if (pred_box_tensor is not None) and (gt_box_tensor is not None) \
                    and pred_box_tensor.shape[0] > 0 \
                    and gt_box_tensor.shape[0] > 0:

                det_boxes_np = common_utils.torch_tensor_to_numpy(
                    pred_box_tensor
                )
                gt_boxes_np = common_utils.torch_tensor_to_numpy(
                    gt_box_tensor
                )

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
                    writer.add_scalar("inference/mean_iou_sample",
                                      mean_iou_sample, i)

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

            # ---------- SIMPLE TB LOGGING: NUM BOXES ----------
            num_pred = 0
            num_gt = 0
            if pred_box_tensor is not None:
                num_pred = pred_box_tensor.shape[1] if pred_box_tensor.ndim >= 2 else pred_box_tensor.shape[0]
            if gt_box_tensor is not None:
                num_gt = gt_box_tensor.shape[1] if gt_box_tensor.ndim >= 2 else gt_box_tensor.shape[0]
            writer.add_scalar("inference/num_pred_boxes", num_pred, i)
            writer.add_scalar("inference/num_gt_boxes", num_gt, i)


            if mean_iou_sample is not None:
                print(
                    f"[Sample {i:05d}] "
                    f"scene={scene_id} | "
                    f"loss={loss_scalar:.4f} | "
                    f"mean_IoU={mean_iou_sample:.4f} | "
                    f"#pred={num_pred} | #gt={num_gt}"
                )
            else:
                # no valid IoU (no preds or no GT)
                print(
                    f"[Sample {i:05d}] "
                    f"scene={scene_id} | "
                    f"loss={loss_scalar:.4f} | "
                    f"mean_IoU=N/A | "
                    f"#pred={num_pred} | #gt={num_gt}"
                )

            # ---------- SAVE NPY ----------
            if opt.save_npy:
                npy_save_path = os.path.join(opt.model_dir, 'npy')
                os.makedirs(npy_save_path, exist_ok=True)
                inference_utils.save_prediction_gt(
                    pred_box_tensor,
                    gt_box_tensor,
                    batch_data['ego']['origin_lidar'][0],
                    i,
                    npy_save_path
                )

            # ---------- SINGLE-IMAGE VIS ----------
            if opt.show_vis or opt.save_vis:
                vis_save_path = ''
                if opt.save_vis:
                    vis_save_path = os.path.join(opt.model_dir, 'vis')
                    os.makedirs(vis_save_path, exist_ok=True)
                    vis_save_path = os.path.join(vis_save_path, '%05d.png' % i)

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

    # ---------- FINAL EVAL ----------
    eval_utils.eval_final_results(result_stat,
                                  opt.model_dir,
                                  opt.global_sort_detections)

    # ---------- COLLATE PER-SCENE METRICS ----------
    scenes = sorted(scene_counts.keys())

    # mean loss per scene
    mean_loss = {}
    for s in scenes:
        if scene_losses[s]:
            mean_loss[s] = float(np.mean(scene_losses[s]))
        else:
            mean_loss[s] = float('nan')

    # mean IoU per scene
    mean_iou = {}
    for s in scenes:
        if scene_ious[s]:
            mean_iou[s] = float(np.mean(scene_ious[s]))
        else:
            mean_iou[s] = float('nan')

    # save CSV summary
    csv_path = os.path.join(opt.model_dir, "scene_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer_csv = csv.writer(f)
        writer_csv.writerow(["scene_id", "num_samples", "mean_loss", "mean_iou"])
        for s in scenes:
            writer_csv.writerow([
                s,
                scene_counts[s],
                mean_loss[s],
                mean_iou[s]
            ])
    print(f"[INFO] Scene metrics saved to: {csv_path}")

    # plots
    if scenes:
        # loss plot
        losses_plot = [mean_loss[s] for s in scenes]
        plt.figure(figsize=(max(10, len(scenes) * 0.4), 5))
        plt.bar(range(len(scenes)), losses_plot)
        plt.xticks(range(len(scenes)), scenes, rotation=90)
        plt.ylabel("Mean Loss")
        plt.title("Per-scene Loss During Inference")
        plt.tight_layout()
        save_path_loss = os.path.join(opt.model_dir, "scene_loss_analysis.png")
        plt.savefig(save_path_loss, dpi=250)
        plt.close()
        print(f"[INFO] Per-scene loss graph saved to: {save_path_loss}")

        # IoU plot
        ious_plot = [mean_iou[s] for s in scenes]
        plt.figure(figsize=(max(10, len(scenes) * 0.4), 5))
        plt.bar(range(len(scenes)), ious_plot)
        plt.xticks(range(len(scenes)), scenes, rotation=90)
        plt.ylabel("Mean IoU")
        plt.title("Per-scene Mean IoU During Inference (diagnostic)")
        plt.tight_layout()
        save_path_iou = os.path.join(opt.model_dir, "scene_iou_analysis.png")
        plt.savefig(save_path_iou, dpi=250)
        plt.close()
        print(f"[INFO] Per-scene IoU graph saved to: {save_path_iou}")

        # log per-scene aggregates to TB
        for idx, s in enumerate(scenes):
            writer.add_scalar("inference/scene_mean_loss", mean_loss[s], idx)
            if not np.isnan(mean_iou[s]):
                writer.add_scalar("inference/scene_mean_iou", mean_iou[s], idx)
            writer.add_text("inference/scene_id", f"{idx}: {s}", idx)
        

    # cleanup
    if opt.show_sequence:
        vis.destroy_window()

    writer.close()
    print(f"TensorBoard writer closed. Logs at: {log_dir}")


if __name__ == '__main__':
    main()
