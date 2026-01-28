# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>, OpenPCDet
# License: TDG-Attribution-NonCommercial-NoDistrib

"""
3D Anchor Generator for Voxel
"""
import math
import sys

import numpy as np
import torch
import torch.nn.functional as F

from opencood.data_utils.post_processor.base_postprocessor import BasePostprocessor
from opencood.utils import box_utils
from opencood.utils.box_overlaps import bbox_overlaps
from opencood.visualization import vis_utils


class VoxelPostprocessor(BasePostprocessor):
    def __init__(self, anchor_params, train):
        super(VoxelPostprocessor, self).__init__(anchor_params, train)
        self.anchor_num = self.params['anchor_args']['num']

    def generate_anchor_box(self):
        W = self.params['anchor_args']['W']
        H = self.params['anchor_args']['H']

        l = self.params['anchor_args']['l']
        w = self.params['anchor_args']['w']
        h = self.params['anchor_args']['h']
        r = self.params['anchor_args']['r']

        assert self.anchor_num == len(r)
        r = [math.radians(ele) for ele in r]

        vh = self.params['anchor_args']['vh']
        vw = self.params['anchor_args']['vw']

        xrange = [self.params['anchor_args']['cav_lidar_range'][0],
                  self.params['anchor_args']['cav_lidar_range'][3]]
        yrange = [self.params['anchor_args']['cav_lidar_range'][1],
                  self.params['anchor_args']['cav_lidar_range'][4]]

        if 'feature_stride' in self.params['anchor_args']:
            feature_stride = self.params['anchor_args']['feature_stride']
        else:
            feature_stride = 2

        x = np.linspace(xrange[0] + vw, xrange[1] - vw, W // feature_stride)
        y = np.linspace(yrange[0] + vh, yrange[1] - vh, H // feature_stride)

        cx, cy = np.meshgrid(x, y)
        cx = np.tile(cx[..., np.newaxis], self.anchor_num)
        cy = np.tile(cy[..., np.newaxis], self.anchor_num)
        cz = np.ones_like(cx) * -1.0

        w = np.ones_like(cx) * w
        l = np.ones_like(cx) * l
        h = np.ones_like(cx) * h

        r_ = np.ones_like(cx)
        for i in range(self.anchor_num):
            r_[..., i] = r[i]

        if self.params['order'] == 'hwl':
            anchors = np.stack([cx, cy, cz, h, w, l, r_], axis=-1)
        elif self.params['order'] == 'lhw':
            anchors = np.stack([cx, cy, cz, l, h, w, r_], axis=-1)
        else:
            sys.exit('Unknown bbx order.')

        return anchors

    def generate_label(self, **kwargs):
        """
        Generate targets for training.

        Parameters
        ----------
        gt_box_center:(max_num, 7), anchor:(H, W, anchor_num, 7)

        Returns
        -------
        label_dict : dict
        """
        assert self.params['order'] == 'hwl', 'Currently Voxel only support hwl bbx order.'
        gt_box_center = kwargs['gt_box_center']     # (max_num, 7)
        anchors = kwargs['anchors']                 # (H, W, anchor_num, 7)
        masks = kwargs['mask']                      # (max_num)

        feature_map_shape = anchors.shape[:2]       # (H, W)

        anchors = anchors.reshape(-1, 7)            # (H*W*anchor_num, 7)
        anchors_d = np.sqrt(anchors[:, 4] ** 2 + anchors[:, 5] ** 2)

        pos_equal_one = np.zeros((*feature_map_shape, self.anchor_num))
        neg_equal_one = np.zeros((*feature_map_shape, self.anchor_num))
        targets = np.zeros((*feature_map_shape, self.anchor_num * 7))

        gt_box_center_valid = gt_box_center[masks == 1]  # (n, 7)

        gt_box_corner_valid = box_utils.boxes_to_corners_3d(
            gt_box_center_valid, self.params['order']
        )
        anchors_corner = box_utils.boxes_to_corners_3d(
            anchors, order=self.params['order']
        )

        anchors_standup_2d = box_utils.corner2d_to_standup_box(anchors_corner)
        gt_standup_2d = box_utils.corner2d_to_standup_box(gt_box_corner_valid)

        iou = bbox_overlaps(
            np.ascontiguousarray(anchors_standup_2d).astype(np.float32),
            np.ascontiguousarray(gt_standup_2d).astype(np.float32),
        )

        id_highest = np.argmax(iou.T, axis=1)
        id_highest_gt = np.arange(iou.T.shape[0])

        mask = iou.T[id_highest_gt, id_highest] > 0
        id_highest, id_highest_gt = id_highest[mask], id_highest_gt[mask]

        id_pos, id_pos_gt = np.where(iou > self.params['target_args']['pos_threshold'])
        id_neg = np.where(
            np.sum(iou < self.params['target_args']['neg_threshold'], axis=1) == iou.shape[1]
        )[0]

        id_pos = np.concatenate([id_pos, id_highest])
        id_pos_gt = np.concatenate([id_pos_gt, id_highest_gt])
        id_pos, index = np.unique(id_pos, return_index=True)
        id_pos_gt = id_pos_gt[index]
        id_neg.sort()

        index_x, index_y, index_z = np.unravel_index(id_pos, (*feature_map_shape, self.anchor_num))
        pos_equal_one[index_x, index_y, index_z] = 1

        targets[index_x, index_y, np.array(index_z) * 7] = \
            (gt_box_center[id_pos_gt, 0] - anchors[id_pos, 0]) / anchors_d[id_pos]
        targets[index_x, index_y, np.array(index_z) * 7 + 1] = \
            (gt_box_center[id_pos_gt, 1] - anchors[id_pos, 1]) / anchors_d[id_pos]
        targets[index_x, index_y, np.array(index_z) * 7 + 2] = \
            (gt_box_center[id_pos_gt, 2] - anchors[id_pos, 2]) / anchors[id_pos, 3]
        targets[index_x, index_y, np.array(index_z) * 7 + 3] = np.log(
            gt_box_center[id_pos_gt, 3] / anchors[id_pos, 3])
        targets[index_x, index_y, np.array(index_z) * 7 + 4] = np.log(
            gt_box_center[id_pos_gt, 4] / anchors[id_pos, 4])
        targets[index_x, index_y, np.array(index_z) * 7 + 5] = np.log(
            gt_box_center[id_pos_gt, 5] / anchors[id_pos, 5])
        targets[index_x, index_y, np.array(index_z) * 7 + 6] = \
            (gt_box_center[id_pos_gt, 6] - anchors[id_pos, 6])

        index_x, index_y, index_z = np.unravel_index(id_neg, (*feature_map_shape, self.anchor_num))
        neg_equal_one[index_x, index_y, index_z] = 1

        index_x, index_y, index_z = np.unravel_index(id_highest, (*feature_map_shape, self.anchor_num))
        neg_equal_one[index_x, index_y, index_z] = 0

        label_dict = {
            'pos_equal_one': pos_equal_one,
            'neg_equal_one': neg_equal_one,
            'targets': targets
        }

        # your debug print (kept)
        pos = label_dict['pos_equal_one']
        neg = label_dict['neg_equal_one']
        print("DEBUG LABELS: num_pos =", pos.sum().item(), " num_neg =", neg.sum().item())

        return label_dict

    @staticmethod
    def collate_batch(label_batch_list):
        pos_equal_one = []
        neg_equal_one = []
        targets = []

        for i in range(len(label_batch_list)):
            pos_equal_one.append(label_batch_list[i]['pos_equal_one'])
            neg_equal_one.append(label_batch_list[i]['neg_equal_one'])
            targets.append(label_batch_list[i]['targets'])

        pos_equal_one = torch.from_numpy(np.array(pos_equal_one))
        neg_equal_one = torch.from_numpy(np.array(neg_equal_one))
        targets = torch.from_numpy(np.array(targets))

        return {
            'targets': targets,
            'pos_equal_one': pos_equal_one,
            'neg_equal_one': neg_equal_one
        }

    def post_process(self, data_dict, output_dict, return_per_cav=False):
        """
        Convert each CAV output to boxes, project to ego, then fuse by NMS.

        If return_per_cav=True, also returns:
          per_cav["cav_2149"] = {"boxes": (Ni,8,3), "scores": (Ni,)}
        where boxes are ego-frame, pre-fusion.
        """
        per_cav = {}
        pred_box3d_list = []
        pred_box2d_list = []

        for cav_id, cav_content in data_dict.items():
            assert cav_id in output_dict

            # ✅ comes from LateFusionDataset.get_item_test: selected_cav_processed['cav_id_str']
            real_id = cav_content.get("cav_id_str", str(cav_id))

            transformation_matrix = cav_content['transformation_matrix']
            anchor_box = cav_content['anchor_box']

            # classification probability
            prob = output_dict[cav_id]['psm']
            prob = torch.sigmoid(prob.permute(0, 2, 3, 1))
            prob = prob.reshape(1, -1)

            reg = output_dict[cav_id]['rm']
            batch_box3d = self.delta_to_boxes3d(reg, anchor_box)

            mask = torch.gt(prob, self.params['target_args']['score_threshold'])
            mask = mask.view(1, -1)
            mask_reg = mask.unsqueeze(2).repeat(1, 1, 7)

            assert batch_box3d.shape[0] == 1
            boxes3d = torch.masked_select(batch_box3d[0], mask_reg[0]).view(-1, 7)
            scores = torch.masked_select(prob[0], mask[0])

            # no boxes for this cav
            if boxes3d.numel() == 0:
                if return_per_cav:
                    per_cav[real_id] = {
                        "boxes": torch.empty((0, 8, 3), device=prob.device),
                        "scores": torch.empty((0,), device=prob.device)
                    }
                continue

            boxes3d_corner = box_utils.boxes_to_corners_3d(
                boxes3d, order=self.params['order']
            )

            projected_boxes3d = box_utils.project_box3d(
                boxes3d_corner, transformation_matrix
            )

            projected_boxes2d = box_utils.corner_to_standup_box_torch(projected_boxes3d)
            boxes2d_score = torch.cat((projected_boxes2d, scores.unsqueeze(1)), dim=1)

            pred_box2d_list.append(boxes2d_score)
            pred_box3d_list.append(projected_boxes3d)

            if return_per_cav:
                per_cav[real_id] = {
                    "boxes": projected_boxes3d.detach(),
                    "scores": scores.detach()
                }

        if len(pred_box2d_list) == 0 or len(pred_box3d_list) == 0:
            if return_per_cav:
                return None, None, per_cav
            return None, None

        pred_box2d_list = torch.vstack(pred_box2d_list)
        scores_all = pred_box2d_list[:, -1]
        pred_box3d_tensor = torch.vstack(pred_box3d_list)

        keep_index_1 = box_utils.remove_large_pred_bbx(pred_box3d_tensor)
        keep_index_2 = box_utils.remove_bbx_abnormal_z(pred_box3d_tensor)
        keep_index = torch.logical_and(keep_index_1, keep_index_2)

        pred_box3d_tensor = pred_box3d_tensor[keep_index]
        scores_all = scores_all[keep_index]

        keep_index = box_utils.nms_rotated(
            pred_box3d_tensor,
            scores_all,
            self.params['nms_thresh']
        )

        pred_box3d_tensor = pred_box3d_tensor[keep_index]
        scores_all = scores_all[keep_index]

        mask_range = box_utils.get_mask_for_boxes_within_range_torch(pred_box3d_tensor)
        pred_box3d_tensor = pred_box3d_tensor[mask_range, :, :]
        scores_all = scores_all[mask_range]

        assert scores_all.shape[0] == pred_box3d_tensor.shape[0]

        if return_per_cav:
            return pred_box3d_tensor, scores_all, per_cav
        else:
            return pred_box3d_tensor, scores_all

    @staticmethod
    def delta_to_boxes3d(deltas, anchors, channel_swap=True):
        """
        Convert the output delta to 3d bbx.

        Parameters
        ----------
        deltas : torch.Tensor
            (N, W, L, 14) or (N, W, L, 7) depending on swap
        anchors : torch.Tensor
            (W, L, 2, 7) -> xyzhwlr
        """
        N = deltas.shape[0]
        if channel_swap:
            deltas = deltas.permute(0, 2, 3, 1).contiguous().view(N, -1, 7)
        else:
            deltas = deltas.contiguous().view(N, -1, 7)

        boxes3d = torch.zeros_like(deltas)
        if deltas.is_cuda:
            anchors = anchors.cuda()
            boxes3d = boxes3d.cuda()

        anchors_reshaped = anchors.view(-1, 7).float()
        anchors_d = torch.sqrt(anchors_reshaped[:, 4] ** 2 + anchors_reshaped[:, 5] ** 2)
        anchors_d = anchors_d.repeat(N, 2, 1).transpose(1, 2)
        anchors_reshaped = anchors_reshaped.repeat(N, 1, 1)

        boxes3d[..., [0, 1]] = torch.mul(deltas[..., [0, 1]], anchors_d) + anchors_reshaped[..., [0, 1]]
        boxes3d[..., [2]] = torch.mul(deltas[..., [2]], anchors_reshaped[..., [3]]) + anchors_reshaped[..., [2]]
        boxes3d[..., [3, 4, 5]] = torch.exp(deltas[..., [3, 4, 5]]) * anchors_reshaped[..., [3, 4, 5]]
        boxes3d[..., 6] = deltas[..., 6] + anchors_reshaped[..., 6]

        return boxes3d

    @staticmethod
    def visualize(pred_box_tensor, gt_tensor, pcd, show_vis, save_path, dataset=None):
        """
        Visualize the prediction, ground truth with point cloud together.
        """
        vis_utils.visualize_single_sample_output_gt(
            pred_box_tensor,
            gt_tensor,
            pcd,
            show_vis,
            save_path
        )
