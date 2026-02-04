import numpy as np
import torch
import torch.nn as nn

from opencood.models.fuse_modules.self_attn import AttFusion
from opencood.models.sub_modules.auto_encoder import AutoEncoder


class AttBEVBackbone(nn.Module):
    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg
        self.compress = False

        if 'compression' in model_cfg and model_cfg['compression'] > 0:
            self.compress = True
            self.compress_layer = model_cfg['compression']

        if 'layer_nums' in self.model_cfg:

            assert len(self.model_cfg['layer_nums']) == \
                   len(self.model_cfg['layer_strides']) == \
                   len(self.model_cfg['num_filters'])

            layer_nums = self.model_cfg['layer_nums']
            layer_strides = self.model_cfg['layer_strides']
            num_filters = self.model_cfg['num_filters']
        else:
            layer_nums = layer_strides = num_filters = []

        if 'upsample_strides' in self.model_cfg:
            assert len(self.model_cfg['upsample_strides']) \
                   == len(self.model_cfg['num_upsample_filter'])

            num_upsample_filters = self.model_cfg['num_upsample_filter']
            upsample_strides = self.model_cfg['upsample_strides']

        else:
            upsample_strides = num_upsample_filters = []

        num_levels = len(layer_nums)
        c_in_list = [input_channels, *num_filters[:-1]]

        self.blocks = nn.ModuleList()
        self.fuse_modules = nn.ModuleList()
        self.deblocks = nn.ModuleList()

        if self.compress:
            self.compression_modules = nn.ModuleList()

        for idx in range(num_levels):
            cur_layers = [
                nn.ZeroPad2d(1),
                nn.Conv2d(
                    c_in_list[idx], num_filters[idx], kernel_size=3,
                    stride=layer_strides[idx], padding=0, bias=False
                ),
                nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                nn.ReLU()
            ]

            fuse_network = AttFusion(num_filters[idx])
            self.fuse_modules.append(fuse_network)
            if self.compress and self.compress_layer - idx > 0:
                self.compression_modules.append(AutoEncoder(num_filters[idx],
                                                            self.compress_layer-idx))

            for k in range(layer_nums[idx]):
                cur_layers.extend([
                    nn.Conv2d(num_filters[idx], num_filters[idx],
                              kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                    nn.ReLU()
                ])

            self.blocks.append(nn.Sequential(*cur_layers))
            if len(upsample_strides) > 0:
                stride = upsample_strides[idx]
                if stride >= 1:
                    self.deblocks.append(nn.Sequential(
                        nn.ConvTranspose2d(
                            num_filters[idx], num_upsample_filters[idx],
                            upsample_strides[idx],
                            stride=upsample_strides[idx], bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx],
                                       eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))
                else:
                    stride = np.round(1 / stride).astype(np.int)
                    self.deblocks.append(nn.Sequential(
                        nn.Conv2d(
                            num_filters[idx], num_upsample_filters[idx],
                            stride,
                            stride=stride, bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3,
                                       momentum=0.01),
                        nn.ReLU()
                    ))

        c_in = sum(num_upsample_filters)
        if len(upsample_strides) > num_levels:
            self.deblocks.append(nn.Sequential(
                nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1],
                                   stride=upsample_strides[-1], bias=False),
                nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ))

        self.num_bev_features = c_in

    def forward(self, data_dict):
        """
        Inputs:
        data_dict['spatial_features']: [sumL, C, H, W]   (stacked across agents)
        data_dict['record_len']:       e.g. tensor([L]) when batch_size=1

        Outputs:
        data_dict['spatial_features_2d']:            fused BEV feature [1, C_out, H_out, W_out]
        data_dict['spatial_features_2d_per_agent']:  per-agent BEV feature [sumL, C_out, H_out, W_out]  (NEW)
        """
        spatial_features = data_dict['spatial_features']   # [sumL, C, H, W]
        record_len = data_dict['record_len']               # tensor([L]) when batch=1

        ups_fused = []
        ups_per_agent = []   # NEW: per-agent upsampled features at each scale

        ret_dict = {}
        x = spatial_features

        # Toggle: if you ever want to turn it off from caller, set this flag in data_dict
        save_per_agent = data_dict.get("save_per_agent_features", True)

        for i in range(len(self.blocks)):
            # ---------------------------------------------------
            # 1) Backbone block (still per-agent stacked)
            # ---------------------------------------------------
            x = self.blocks[i](x)  # [sumL, C_i, H_i, W_i]

            if self.compress and i < len(self.compression_modules):
                x = self.compression_modules[i](x)

            # Save per-level per-agent features (optional debug)
            stride = int(spatial_features.shape[2] / x.shape[2])
            ret_dict[f"spatial_features_{stride}x_per_agent"] = x

            # ---------------------------------------------------
            # 2) Fuse per level
            # ---------------------------------------------------
            x_fuse = self.fuse_modules[i](x, record_len)  # usually [B=1, C_i, H_i, W_i]

            # ---------------------------------------------------
            # 3) Upsample both fused and per-agent tensors
            # ---------------------------------------------------
            if len(self.deblocks) > 0:
                ups_fused.append(self.deblocks[i](x_fuse))

                if save_per_agent:
                    # Apply same deblock to the stacked per-agent tensor
                    ups_per_agent.append(self.deblocks[i](x))
            else:
                ups_fused.append(x_fuse)

                if save_per_agent:
                    ups_per_agent.append(x)

        # ---------------------------------------------------
        # 4) Concatenate multi-scale fused features
        # ---------------------------------------------------
        if len(ups_fused) > 1:
            x_out = torch.cat(ups_fused, dim=1)
        elif len(ups_fused) == 1:
            x_out = ups_fused[0]
        else:
            x_out = x  # should not happen

        if len(self.deblocks) > len(self.blocks):
            x_out = self.deblocks[-1](x_out)

        data_dict['spatial_features_2d'] = x_out  # fused final feature

        # ---------------------------------------------------
        # 5) Concatenate multi-scale per-agent features  (NEW)
        # ---------------------------------------------------
        if save_per_agent:
            if len(ups_per_agent) > 1:
                x_pa = torch.cat(ups_per_agent, dim=1)  # [sumL, C_out, H_out, W_out]
            elif len(ups_per_agent) == 1:
                x_pa = ups_per_agent[0]
            else:
                x_pa = x

            if len(self.deblocks) > len(self.blocks):
                x_pa = self.deblocks[-1](x_pa)

            data_dict['spatial_features_2d_per_agent'] = x_pa

        # Keep debug features (optional)
        data_dict.update(ret_dict)

        return data_dict


