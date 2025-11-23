# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

import os
import argparse
from torch.utils.data import DataLoader

from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.visualization import vis_utils
from opencood.data_utils.datasets.early_fusion_vis_dataset import EarlyFusionVisDataset
from opencood.data_utils.datasets.early_fusion_dataset import EarlyFusionDataset as EarlyAny
from opencood.data_utils.datasets.intermediate_fusion_dataset import IntermediateFusionDataset as IntermediateAny
from opencood.data_utils.datasets.late_fusion_dataset import LateFusionDataset as LateAny

NAME2CLS = {
    'EarlyFusionDataset': EarlyFusionVisDataset,   # prefer the *vis* version you have
    'IntermediateFusionDataset': IntermediateAny,
    'LateFusionDataset': LateAny,
}

def vis_parser():
    parser = argparse.ArgumentParser(description="data visualization")
    parser.add_argument(
        '--color_mode',
        type=str,
        default="intensity",
        help='lidar color rendering mode, e.g. intensity, z-value or constant.'
    )
    # If you ONLY want YAML, you don’t need fusion_method in the parser.
    opt = parser.parse_args()
    return opt


if __name__ == '__main__':
    # 1) Load YAML
    current_path = os.path.dirname(os.path.realpath(__file__))
    params = load_yaml(os.path.join(current_path, '../hypes_yaml/visualization.yaml'))

    # 2) Parse CLI args (only color_mode)
    opt = vis_parser()

    # 3) Read fusion method from YAML
    dataset_key = params['fusion']['core_method']   # e.g. "IntermediateFusionDataset"
    print(f"[Visualizer] Using dataset: {dataset_key} for visualization.")

    # 4) Map YAML fusion name -> actual dataset class
    if dataset_key not in NAME2CLS:
        raise ValueError(f"Dataset {dataset_key} not supported for visualization.")
    dataset_cls = NAME2CLS[dataset_key]

    # 5) Choose correct lidar key for visualizer
    lidar_key = "stacked_lidar" if dataset_key == "EarlyFusionDataset" else "origin_lidar"
    print(f"[Visualizer] Using LiDAR key: {lidar_key}")

    # 6) Create dataset (handle older ctor signatures)
    try:
        opencda_dataset = dataset_cls(params, visualize=True, train=False)
    except TypeError:
        opencda_dataset = dataset_cls(params, train=False)

    # 7) Build DataLoader
    data_loader = DataLoader(
        opencda_dataset,
        batch_size=1,
        num_workers=2,
        collate_fn=opencda_dataset.collate_batch_train,
        shuffle=False,
        pin_memory=False
    )

    # 8) Run visualizer
    try:
        vis_utils.visualize_sequence_dataloader(
            data_loader,
            params['postprocess']['order'],
            color_mode=opt.color_mode,
            key=lidar_key,
            window_name=f"OpenCOOD • {dataset_key.replace('FusionDataset','')} Fusion"
        )
    except TypeError:
        # older vis_utils without key/window_name
        vis_utils.visualize_sequence_dataloader(
            data_loader,
            params['postprocess']['order'],
            color_mode=opt.color_mode
        )