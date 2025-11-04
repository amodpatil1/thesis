# -*- coding: utf-8 -*-
# Author: Runsheng Xu, Hao Xiang, Yifan Lu
# License: TDG-Attribution-NonCommercial-NoDistrib

import argparse
import os
import time
from tqdm import tqdm

import torch
import open3d as o3d
from torch.utils.data import DataLoader

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils, inference_utils
from opencood.data_utils.datasets import build_dataset
from opencood.utils import eval_utils
from opencood.visualization import vis_utils
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# ✅ SAFE CUDA + CUDNN SETTINGS FOR YOUR ENVIRONMENT
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
# ---------------------------------------------------------------------

def _to_fp32(obj):
    if isinstance(obj, torch.Tensor):
        # Keep integer types (coords/indices) intact, cast float types to fp32
        return obj.float() if obj.is_floating_point() else obj
    if isinstance(obj, dict):
        return {k: _to_fp32(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        t = type(obj)
        return t(_to_fp32(v) for v in obj)
    return obj


def test_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Path to trained model directory')
    parser.add_argument('--fusion_method', required=True, type=str,
                        choices=['late', 'early', 'intermediate'])
    parser.add_argument('--show_vis', action='store_true',
                        help='Show image visualization results')
    parser.add_argument('--show_sequence', action='store_true',
                        help='Show video sequence visualization')
    parser.add_argument('--save_vis', action='store_true',
                        help='Save visualized BEV results')
    parser.add_argument('--save_npy', action='store_true',
                        help='Save prediction and GT arrays')
    parser.add_argument('--global_sort_detections', action='store_true',
                        help='Globally sort detections by confidence score')
    return parser.parse_args()


def main():
    opt = test_parser()

    hypes = yaml_utils.load_yaml(None, opt)
    print('Dataset Building')
    opencood_dataset = build_dataset(hypes, visualize=True, train=False)
    print(f"{len(opencood_dataset)} samples found.")

    # Keep DataLoader light and stable
    data_loader = DataLoader(
        opencood_dataset,
        batch_size=1,
        num_workers=1,          # safer than 8
        collate_fn=opencood_dataset.collate_batch_test,
        shuffle=False,
        pin_memory=True,
        drop_last=False
    )

    print('Creating Model')
    model = train_utils.create_model(hypes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        model.cuda()
    print('Loading Model from checkpoint')
    _, model = train_utils.load_saved_model(opt.model_dir, model)
    model = model.float()       # ✅ Ensure full precision (FP32)
    model.eval()

    # Evaluation results container
    result_stat = {thr: {'tp': [], 'fp': [], 'gt': 0, 'score': []}
                   for thr in (0.3, 0.5, 0.7)}

    # Visualization setup (headless safe)
    do_viz = False
    vis = None
    vis_pcd, vis_aabbs_gt, vis_aabbs_pred = None, None, None

    if opt.show_sequence:
        try:
            _vis = o3d.visualization.Visualizer()
            ok = _vis.create_window(visible=True)
            if not ok:
                print("[WARN] Open3D failed to create a GL context. Running headless.")
            else:
                vis = _vis
                do_viz = True
                ro = vis.get_render_option()
                ro.background_color = [0.05, 0.05, 0.05]
                ro.point_size = 1.0
                ro.show_coordinate_frame = True
                vis_pcd = o3d.geometry.PointCloud()
                vis_aabbs_gt = [o3d.geometry.LineSet() for _ in range(50)]
                vis_aabbs_pred = [o3d.geometry.LineSet() for _ in range(50)]
        except Exception as e:
            print(f"[WARN] Open3D init failed: {e}. Running without visualization.")

    # ---------------- INFERENCE LOOP -----------------
    for i, batch_data in tqdm(enumerate(data_loader)):
        with torch.no_grad():
            batch_data = train_utils.to_device(batch_data, device)
    
    for i, batch_data in tqdm(enumerate(data_loader)):
        with torch.no_grad():
            batch_data = train_utils.to_device(batch_data, device)

        # NEW: enforce float32 everywhere in the batch
        batch_data = _to_fp32(batch_data)

        with torch.cuda.amp.autocast(enabled=False):
            if opt.fusion_method == 'late':
                pred_box_tensor, pred_score, gt_box_tensor = \
                    inference_utils.inference_late_fusion(batch_data, model, opencood_dataset)
            elif opt.fusion_method == 'early':
                pred_box_tensor, pred_score, gt_box_tensor = \
                    inference_utils.inference_early_fusion(batch_data, model, opencood_dataset)
            else:  # intermediate
                pred_box_tensor, pred_score, gt_box_tensor = \
                    inference_utils.inference_intermediate_fusion(batch_data, model, opencood_dataset)


            # ✅ Run in pure FP32 (disable autocast everywhere)
            with torch.cuda.amp.autocast(enabled=False):
                if opt.fusion_method == 'late':
                    pred_box_tensor, pred_score, gt_box_tensor = \
                        inference_utils.inference_late_fusion(batch_data, model, opencood_dataset)
                elif opt.fusion_method == 'early':
                    pred_box_tensor, pred_score, gt_box_tensor = \
                        inference_utils.inference_early_fusion(batch_data, model, opencood_dataset)
                elif opt.fusion_method == 'intermediate':
                    pred_box_tensor, pred_score, gt_box_tensor = \
                        inference_utils.inference_intermediate_fusion(batch_data, model, opencood_dataset)

            # Eval metrics
            for thr in (0.3, 0.5, 0.7):
                eval_utils.caluclate_tp_fp(pred_box_tensor, pred_score, gt_box_tensor, result_stat, thr)

            # Optional saving
            if opt.save_npy:
                npy_save_path = os.path.join(opt.model_dir, 'npy')
                os.makedirs(npy_save_path, exist_ok=True)
                inference_utils.save_prediction_gt(
                    pred_box_tensor, gt_box_tensor,
                    batch_data['ego']['origin_lidar'][0], i, npy_save_path
                )

            if opt.show_vis or opt.save_vis:
                vis_save_path = ''
                if opt.save_vis:
                    vis_dir = os.path.join(opt.model_dir, 'vis')
                    os.makedirs(vis_dir, exist_ok=True)
                    vis_save_path = os.path.join(vis_dir, f'{i:05d}.png')
                opencood_dataset.visualize_result(
                    pred_box_tensor, gt_box_tensor,
                    batch_data['ego']['origin_lidar'],
                    opt.show_vis, vis_save_path, dataset=opencood_dataset
                )

            # Live sequence visualization (if possible)
            if opt.show_sequence and do_viz:
                pcd, pred_o3d_box, gt_o3d_box = vis_utils.visualize_inference_sample_dataloader(
                    pred_box_tensor, gt_box_tensor,
                    batch_data['ego']['origin_lidar'],
                    vis_pcd, mode='constant'
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
    # -------------------------------------------------

    eval_utils.eval_final_results(result_stat, opt.model_dir, opt.global_sort_detections)
    if opt.show_sequence and do_viz and vis is not None:
        vis.destroy_window()


if __name__ == '__main__':
    main()
