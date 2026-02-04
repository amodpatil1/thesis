# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

import torch
import torch.nn as nn

from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.sub_modules.att_bev_backbone import AttBEVBackbone


class PointPillarIntermediate(nn.Module):
    def __init__(self, args):
        super(PointPillarIntermediate, self).__init__()

        # Pillar VFE
        self.pillar_vfe = PillarVFE(
            args['pillar_vfe'],
            num_point_features=4,
            voxel_size=args['voxel_size'],
            point_cloud_range=args['lidar_range']
        )
        self.scatter = PointPillarScatter(args['point_pillar_scatter'])
        self.backbone = AttBEVBackbone(args['base_bev_backbone'], 64)

        self.cls_head = nn.Conv2d(128 * 3, args['anchor_number'], kernel_size=1)
        self.reg_head = nn.Conv2d(128 * 3, 7 * args['anchor_num'], kernel_size=1)

    def forward(self, data_dict):
        voxel_features = data_dict['processed_lidar']['voxel_features']
        voxel_coords = data_dict['processed_lidar']['voxel_coords']
        voxel_num_points = data_dict['processed_lidar']['voxel_num_points']
        record_len = data_dict['record_len']

        batch_dict = {
            'voxel_features': voxel_features,
            'voxel_coords': voxel_coords,
            'voxel_num_points': voxel_num_points,
            'record_len': record_len
        }

        batch_dict = self.pillar_vfe(batch_dict)
        batch_dict = self.scatter(batch_dict)
        batch_dict = self.backbone(batch_dict)

        # ---------------------------------------------------
        # fused feature (used for heads)
        # ---------------------------------------------------
        spatial_features_2d = batch_dict['spatial_features_2d']

        # ---------------------------------------------------
        # ✅ per-agent feature stack (needed for feature misalignment)
        # We only attach it to output_dict if it exists.
        #
        # Expected from modified AttBEVBackbone:
        #   spatial_features_2d_per_agent: [sumL, C, H, W]
        #
        # Some variants might output:
        #   [1, sumL, C, H, W]  (batch dim)
        # We'll normalize it to [sumL, C, H, W] for convenience.
        # ---------------------------------------------------
        spatial_features_2d_per_agent = batch_dict.get('spatial_features_2d_per_agent', None)
        if spatial_features_2d_per_agent is not None and isinstance(spatial_features_2d_per_agent, torch.Tensor):
            if spatial_features_2d_per_agent.ndim == 5:
                # [B, sumL, C, H, W] -> assume B=1 at inference
                spatial_features_2d_per_agent = spatial_features_2d_per_agent[0]
            # else: already [sumL, C, H, W]

        # heads
        psm = self.cls_head(spatial_features_2d)
        rm = self.reg_head(spatial_features_2d)

        # ---------------------------------------------------
        # output dict (keep backwards compatibility)
        # ---------------------------------------------------
        output_dict = {
            'psm': psm,
            'rm': rm,
            'spatial_features_2d': spatial_features_2d,
        }

        # Only add per-agent tensor if it exists (prevents None everywhere)
        if spatial_features_2d_per_agent is not None:
            output_dict['spatial_features_2d_per_agent'] = spatial_features_2d_per_agent

        return output_dict

