# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

import os
from collections import OrderedDict

import numpy as np
import torch

from opencood.utils.common_utils import torch_tensor_to_numpy


def inference_late_fusion(batch_data, model, dataset):
    """
    Model inference for late fusion.

    Returns
    -------
    pred_box_tensor : torch.Tensor
        Fused prediction bounding boxes after NMS.
    pred_score : torch.Tensor
        Fused prediction scores.
    gt_box_tensor : torch.Tensor
        Ground-truth boxes.
    per_cav : OrderedDict
        per_cav[cav_id] = (cav_pred_box_tensor, cav_pred_score_tensor)
        Each is the prediction result if we "post_process" only that single CAV.
    """
    output_dict = OrderedDict()

    # 1) Forward pass for each CAV
    for cav_id, cav_content in batch_data.items():
        output_dict[cav_id] = model(cav_content)

    # 2) Fused result (original behavior) - requires full batch_data + full output_dict
    pred_box_tensor, pred_score, gt_box_tensor = dataset.post_process(batch_data, output_dict)

    # 3) Per-CAV results (NEW)
    # IMPORTANT:
    # voxel_postprocessor asserts that for every cav_id in data_dict, cav_id must exist in output_dict.
    # So for per-cav, we must pass matching single-cav data_dict and output_dict.
    per_cav = OrderedDict()
    for cav_id in output_dict.keys():
        single_output = OrderedDict({cav_id: output_dict[cav_id]})
        single_data = OrderedDict({cav_id: batch_data[cav_id]})  # ✅ FIX: keys match

        cav_pred_box, cav_pred_score, _ = dataset.post_process(single_data, single_output)
        per_cav[cav_id] = (cav_pred_box, cav_pred_score)

    return pred_box_tensor, pred_score, gt_box_tensor, per_cav


def inference_early_fusion(batch_data, model, dataset):
    """
    Model inference for early fusion.

    Parameters
    ----------
    batch_data : dict
    model : opencood.object
    dataset : opencood.EarlyFusionDataset

    Returns
    -------
    pred_box_tensor : torch.Tensor
        The tensor of prediction bounding box after NMS.
    pred_score : torch.Tensor
        The tensor of prediction scores.
    gt_box_tensor : torch.Tensor
        The tensor of gt bounding box.
    """
    output_dict = OrderedDict()
    cav_content = batch_data['ego']

    output_dict['ego'] = model(cav_content)

    pred_box_tensor, pred_score, gt_box_tensor = dataset.post_process(batch_data, output_dict)
    return pred_box_tensor, pred_score, gt_box_tensor


def inference_intermediate_fusion(batch_data, model, dataset):
    """
    Model inference for intermediate fusion.
    In OpenCOOD, intermediate uses the same pipeline as early here.
    """
    return inference_early_fusion(batch_data, model, dataset)


def save_prediction_gt(pred_tensor, gt_tensor, pcd, timestamp, save_path):
    """
    Save prediction and gt tensor to npy files.

    Saved files:
      <timestamp>_pcd.npy
      <timestamp>_pred.npy
      <timestamp>_gt.npy_test.npy  (matches your existing naming)
    """
    pred_np = torch_tensor_to_numpy(pred_tensor)
    gt_np = torch_tensor_to_numpy(gt_tensor)
    pcd_np = torch_tensor_to_numpy(pcd)

    np.save(os.path.join(save_path, '%04d_pcd.npy' % timestamp), pcd_np)
    np.save(os.path.join(save_path, '%04d_pred.npy' % timestamp), pred_np)
    np.save(os.path.join(save_path, '%04d_gt.npy_test' % timestamp), gt_np)
