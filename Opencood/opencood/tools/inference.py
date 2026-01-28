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

    # fallback: group by chunks of 100 frames
    chunk = batch_idx // 100
    return f"chunk_{chunk:04d}"


def mean_det_to_gt_iou(det_boxes_tensor, gt_boxes_tensor):
    """
    det_boxes_tensor: torch.Tensor (N,8,3) in ego frame
    gt_boxes_tensor : torch.Tensor (M,8,3) in ego frame

    Returns:
      mean_iou: float or None
      num_pred: int
      num_gt: int

    Metric:
      For each det, take max IoU with any GT, then average across dets.
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

    # loss function (same as training)
    criterion = train_utils.create_loss(hypes)

    # per-scene aggregation
    scene_losses = defaultdict(list)
    scene_ious = defaultdict(list)
    scene_counts = defaultdict(int)

    # TensorBoard writer
    log_dir = os.path.join(opt.model_dir, "tb_inference")
    print(f"TensorBoard logs will be written to: {log_dir}")
    writer = SummaryWriter(log_dir=log_dir)

    # per-cav iou csv rows
    per_cav_rows = []

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

    def _to_np(x):
        if x is None:
            return None
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    # ---------- INFERENCE LOOP ----------
    for i, batch_data in tqdm(enumerate(data_loader)):
        with torch.no_grad():
            batch_data = train_utils.to_device(batch_data, device)

            # get scene id
            scene_id = extract_scene_id(batch_data, i, dataset=opencood_dataset)
            scene_counts[scene_id] += 1
            if scene_id not in seen_scenes:
                print(f"[SCENE] New scene encountered: {scene_id}")
                seen_scenes.add(scene_id)
            if scene_id != prev_scene:
                writer.add_text("inference/scene_change",
                                f"step {i}: scene {scene_id}", i)
                prev_scene = scene_id

            # ---------- LOSS PER SAMPLE ----------
            output_for_loss = model(batch_data['ego'])
            loss_value = criterion(output_for_loss,
                                   batch_data['ego']['label_dict'])
            loss_scalar = float(loss_value.item())
            scene_losses[scene_id].append(loss_scalar)
            writer.add_scalar("inference/loss", loss_scalar, i)

            # ---------- DETECTION INFERENCE ----------
            per_cav = None
            if opt.fusion_method == 'late':
                pred_box_tensor, pred_score, gt_box_tensor, per_cav = \
                    inference_utils.inference_late_fusion(
                        batch_data,
                        model,
                        opencood_dataset,
                        return_per_cav=True
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
                raise NotImplementedError('Only early, late and intermediate fusion is supported.')

            # ---------- SAVE DETECTIONS (PER SCENE FOLDER) ----------
            det_root = os.path.join(opt.model_dir, "per_chunk_dets")
            os.makedirs(det_root, exist_ok=True)
            scene_dir = os.path.join(det_root, str(scene_id))
            os.makedirs(scene_dir, exist_ok=True)

            sample_id = i  # replace with real frame id if you have it

            # ---- Save fused output ----
            fused_out = {
                "scene_id": str(scene_id),
                "sample_id": int(sample_id),
                "pred_box_tensor": _to_np(pred_box_tensor),
                "pred_score": _to_np(pred_score),
                "gt_box_tensor": _to_np(gt_box_tensor),
            }
            np.save(os.path.join(scene_dir, f"{sample_id:06d}_fused.npy"),
                    fused_out, allow_pickle=True)

            # ---- Save per-CAV preds as separate files + compute per-cav IoU ----
            if opt.fusion_method == "late" and per_cav is not None:
                cav_list = []

                for cav_key, d in per_cav.items():
                    cav_str = str(cav_key)
                    # keep "ego" as "ego", but normalize numeric ids
                    if cav_str.isdigit():
                        cav_str = f"cav_{cav_str}"
                    elif cav_str.startswith("cav_"):
                        pass
                    elif cav_str.lower() == "ego":
                        cav_str = "ego"
                    else:
                        # fallback
                        cav_str = cav_str

                    cav_list.append(cav_str)

                    b = d.get("boxes", None)   # torch (N,8,3) ego-frame
                    s = d.get("scores", None)  # torch (N,)

                    # --- per-cav IoU vs GT (ego-frame) ---
                    cav_mean_iou, cav_num_pred, cav_num_gt = mean_det_to_gt_iou(b, gt_box_tensor)

                    # store row (use -1.0 when not computable)
                    per_cav_rows.append([
                        str(scene_id),
                        int(sample_id),
                        cav_str,
                        -1.0 if cav_mean_iou is None else float(cav_mean_iou),
                        int(cav_num_pred),
                        int(cav_num_gt)
                    ])

                    # optional TB logging per cav
                    if cav_mean_iou is not None:
                        writer.add_scalar(f"inference/per_cav_mean_iou/{cav_str}", cav_mean_iou, i)

                    out = {
                        "scene_id": str(scene_id),
                        "sample_id": int(sample_id),
                        "cav_id": cav_str,
                        "pred_box_tensor": _to_np(b),
                        "pred_score": _to_np(s),
                        "gt_box_tensor": _to_np(gt_box_tensor),
                        "mean_iou_det_to_gt": cav_mean_iou,
                    }

                    np.save(os.path.join(scene_dir, f"{sample_id:06d}_{cav_str}_pred.npy"),
                            out, allow_pickle=True)

                # write which cavs were present
                with open(os.path.join(scene_dir, f"{sample_id:06d}_cavs.txt"), "w") as f:
                    f.write("\n".join(cav_list))

            # ---------- PER-SAMPLE MEAN IoU (diagnostic, fused) ----------
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

            # ---------- TP/FP STATS FOR AP ----------
            eval_utils.caluclate_tp_fp(pred_box_tensor, pred_score, gt_box_tensor, result_stat, 0.3)
            eval_utils.caluclate_tp_fp(pred_box_tensor, pred_score, gt_box_tensor, result_stat, 0.5)
            eval_utils.caluclate_tp_fp(pred_box_tensor, pred_score, gt_box_tensor, result_stat, 0.7)

            # ---------- SIMPLE TB LOGGING: NUM BOXES ----------
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

            # ---------- SAVE NPY (OpenCOOD default) ----------
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
                    vis_utils.linset_assign_list(vis, vis_aabbs_pred, pred_o3d_box, update_mode='add')
                    vis_utils.linset_assign_list(vis, vis_aabbs_gt, gt_o3d_box, update_mode='add')

                vis_utils.linset_assign_list(vis, vis_aabbs_pred, pred_o3d_box)
                vis_utils.linset_assign_list(vis, vis_aabbs_gt, gt_o3d_box)
                vis.update_geometry(pcd)
                vis.poll_events()
                vis.update_renderer()
                time.sleep(0.001)

    # ---------- WRITE PER-CAV IOU CSV ----------
    per_cav_csv_path = os.path.join(opt.model_dir, "per_cav_iou.csv")
    with open(per_cav_csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scene_id", "sample_id", "cav_id", "mean_iou_det_to_gt", "num_pred", "num_gt"])
        w.writerows(per_cav_rows)
    print(f"[INFO] Per-CAV IoU saved to: {per_cav_csv_path}")

    # ---------- FINAL EVAL ----------
    eval_utils.eval_final_results(result_stat, opt.model_dir, opt.global_sort_detections)

    # ---------- COLLATE PER-SCENE METRICS ----------
    scenes = sorted(scene_counts.keys())

    mean_loss = {s: (float(np.mean(scene_losses[s])) if scene_losses[s] else float('nan')) for s in scenes}
    mean_iou = {s: (float(np.mean(scene_ious[s])) if scene_ious[s] else float('nan')) for s in scenes}

    # save CSV summary
    csv_path = os.path.join(opt.model_dir, "scene_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer_csv = csv.writer(f)
        writer_csv.writerow(["scene_id", "num_samples", "mean_loss", "mean_iou"])
        for s in scenes:
            writer_csv.writerow([s, scene_counts[s], mean_loss[s], mean_iou[s]])
    print(f"[INFO] Scene metrics saved to: {csv_path}")

    # plots
    if scenes:
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
