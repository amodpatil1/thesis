# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

import os
from collections import OrderedDict

import numpy as np
import torch

from opencood.utils.common_utils import torch_tensor_to_numpy


def _safe_post_process_late(dataset, batch_data, output_dict, want_per_cav: bool):
    """
    Try dataset.post_process with return_per_cav (if supported).
    If not supported / returns only fused outputs, fall back to per-cav by single-agent post_process.
    """
    # ---------------------------
    # 1) Get fused results first
    # ---------------------------
    fused_ret = None
    if want_per_cav:
        # Some OpenCOOD versions accept return_per_cav and return (pred, score, gt, per_cav)
        try:
            fused_ret = dataset.post_process(batch_data, output_dict, return_per_cav=True)
        except TypeError:
            fused_ret = None

        if fused_ret is not None:
            # If it already returned per_cav, we're done
            if isinstance(fused_ret, (list, tuple)) and len(fused_ret) == 4:
                return fused_ret

            # If it returned only fused results, keep them and we will build per_cav ourselves
            if isinstance(fused_ret, (list, tuple)) and len(fused_ret) == 3:
                pred_box_tensor, pred_score, gt_box_tensor = fused_ret
            else:
                # unexpected format; re-run without return_per_cav
                pred_box_tensor, pred_score, gt_box_tensor = dataset.post_process(batch_data, output_dict)
    else:
        pred_box_tensor, pred_score, gt_box_tensor = dataset.post_process(batch_data, output_dict)
        return pred_box_tensor, pred_score, gt_box_tensor

    # ---------------------------------------------
    # 2) Build per-cav predictions via fallback path
    # ---------------------------------------------
    per_cav = OrderedDict()

    # We do per-agent postprocess by temporarily treating each cav as "ego".
    # This is robust across many OpenCOOD code paths because post_process usually expects "ego".
    for cav_id, cav_content in batch_data.items():
        # Only dict entries are valid CAVs
        if not isinstance(cav_content, dict):
            continue
        if cav_id not in output_dict:
            continue

        single_batch = OrderedDict()
        single_out = OrderedDict()

        # Treat this cav as ego to reuse existing post_process code paths
        single_batch["ego"] = cav_content
        single_out["ego"] = output_dict[cav_id]

        try:
            pb, ps, _gt = dataset.post_process(single_batch, single_out)
        except Exception:
            # If something fails, store empty so caller can still save files consistently
            pb = torch.zeros((0, 8, 3), device=pred_box_tensor.device if pred_box_tensor is not None else "cpu")
            ps = torch.zeros((0,), device=pb.device)

        per_cav[str(cav_id)] = {"boxes": pb, "scores": ps}

    return pred_box_tensor, pred_score, gt_box_tensor, per_cav


def inference_late_fusion(batch_data, model, dataset, return_per_cav=False):
    """
    Late fusion inference.
    Always returns:
      - (pred_box_tensor, pred_score, gt_box_tensor) if return_per_cav=False
      - (pred_box_tensor, pred_score, gt_box_tensor, per_cav) if return_per_cav=True

    per_cav format:
      per_cav[cav_key_str] = {"boxes": (N,8,3) torch, "scores": (N,) torch}
    """
    output_dict = OrderedDict()
    for cav_id, cav_content in batch_data.items():
        if not isinstance(cav_content, dict):
            continue
        output_dict[cav_id] = model(cav_content)

    return _safe_post_process_late(dataset, batch_data, output_dict, want_per_cav=return_per_cav)


def inference_early_fusion(batch_data, model, dataset):
    """
    Model inference for early fusion.
    """
    output_dict = OrderedDict()
    cav_content = batch_data['ego']
    output_dict['ego'] = model(cav_content)

    pred_box_tensor, pred_score, gt_box_tensor = dataset.post_process(batch_data, output_dict)
    return pred_box_tensor, pred_score, gt_box_tensor


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
    pts = boxes_8_3.reshape(-1, 3)  # (N*8, 3)
    ones = torch.ones((pts.shape[0], 1), device=pts.device, dtype=pts.dtype)
    pts_h = torch.cat([pts, ones], dim=1)  # (N*8, 4)
    pts_out = (T_4x4 @ pts_h.t()).t()[:, :3]
    return pts_out.reshape(N, 8, 3)


def inference_intermediate_fusion(batch_data, model, dataset, return_per_cav=False):
    """
    Works with IntermediateFusionDataset + voxel_postprocessor which expects output_dict keyed by cav_id.
    """

    # 1) forward on ego-packed intermediate input
    ego_out = model(batch_data['ego'])

    # wrap outputs under cav id key expected by postprocessor
    output_dict = {"ego": ego_out}

    # 2) post process
    pred_box_tensor, pred_score, gt_box_tensor = dataset.post_process(batch_data, output_dict)

    if not return_per_cav:
        return pred_box_tensor, pred_score, gt_box_tensor

    # 3) build per-cav predictions (NOTE: this is NOT true per-cav detection;
    #    it transforms fused ego predictions into each cav frame for saving/analysis.)
    ego_dict = batch_data.get("ego", {})
    pairwise = ego_dict.get("pairwise_t_matrix", None)
    record_len = ego_dict.get("record_len", None)

    per_cav = OrderedDict()
    if pairwise is None or record_len is None:
        return pred_box_tensor, pred_score, gt_box_tensor, per_cav

    if pairwise.shape[0] != 1:
        raise RuntimeError("This implementation assumes batch_size=1.")

    L = int(record_len[0])  # number of cavs incl ego

    for cav_idx in range(L):
        # convention: pairwise[b, src, tgt] = T_src->tgt
        T_ego_to_cav = pairwise[0, 0, cav_idx]
        boxes_cav = _apply_4x4_to_boxes_corners(pred_box_tensor, T_ego_to_cav)

        key = "ego" if cav_idx == 0 else f"cav_{cav_idx}"
        per_cav[key] = {"boxes": boxes_cav, "scores": pred_score}

    return pred_box_tensor, pred_score, gt_box_tensor, per_cav


def save_prediction_gt(pred_tensor, gt_tensor, pcd, timestamp, save_path):
    """
    Save prediction and gt tensor to npy file.
    """
    pred_np = torch_tensor_to_numpy(pred_tensor)
    gt_np = torch_tensor_to_numpy(gt_tensor)
    pcd_np = torch_tensor_to_numpy(pcd)

    np.save(os.path.join(save_path, '%04d_pcd.npy' % timestamp), pcd_np)
    np.save(os.path.join(save_path, '%04d_pred.npy' % timestamp), pred_np)
    np.save(os.path.join(save_path, '%04d_gt.npy_test' % timestamp), gt_np)
