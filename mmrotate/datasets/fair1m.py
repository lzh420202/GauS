# Copyright (c) OpenMMLab. All rights reserved.
import glob
import os.path as osp
from typing import List, Tuple

from mmengine.dataset import BaseDataset

from mmrotate.registry import DATASETS
try:
    from .dota import DOTADataset
except:
    from mmrotate.datasets import DOTADataset


@DATASETS.register_module()
class FAIR1MDataset(DOTADataset):
    METAINFO = {
        # 37 categories
        # 'a': (
        #     'small-car', 'van', 'dump-truck', 'cargo-truck', 'motorboat',
        #     'other-vehicle', 'dry-cargo-ship', 'intersection', 'other-ship',
        #     'fishing-boat', 'liquid-cargo-ship', 'truck-tractor', 'other-airplane',
        #     'engineering-ship', 'bus', 'tennis-court', 'trailer', 'excavator',
        #     'a220', 'passenger-ship', 'football-field', 'boeing737', 'warship',
        #     'tugboat', 'baseball-field', 'a321', 'boeing787', 'basketball-court',
        #     'boeing747', 'a330', 'boeing777', 'tractor', 'bridge', 'a350', 'c919',
        #     'arj21', 'roundabout'
        # ),
        'classes':
        ('boeing737', 'boeing777', 'boeing747', 'boeing787', 'a321',
         'a220', 'a330', 'a350', 'c919', 'arj21', 'other-airplane',
         'passenger-ship', 'motorboat', 'fishing-boat', 'tugboat', 'engineering-ship',
         'liquid-cargo-ship', 'dry-cargo-ship', 'warship', 'other-ship',
         'small-car', 'bus', 'cargo-truck', 'dump-truck', 'van', 'trailer',
         'tractor', 'truck-tractor', 'excavator', 'other-vehicle',
         'baseball-field', 'basketball-court', 'football-field', 'tennis-court',
         'roundabout', 'intersection', 'bridge'),
        # palette is a list of color tuples, which is used for visualization.
        'palette': [(62, 38, 168), (65, 45, 189), (68, 52, 208), (70, 61, 224),
                    (71, 70, 235), (72, 80, 243), (71, 90, 249), (69, 100, 253),
                    (63, 110, 255), (53, 121, 253), (46, 131, 249), (45, 140, 243),
                    (40, 149, 236), (36, 158, 230), (31, 166, 226), (25, 173, 220),
                    (11, 179, 210), (0, 185, 199), (13, 189, 188), (35, 193, 175),
                    (47, 197, 163), (57, 201, 148), (75, 203, 132), (97, 205, 114),
                    (121, 204, 94), (147, 202, 75), (171, 199, 57), (194, 195, 44),
                    (214, 190, 39), (232, 187, 46), (247, 186, 61), (254, 193, 58),
                    (253, 203, 50), (249, 214, 45), (245, 225, 40), (245, 236, 34),
                    (248, 247, 26)]
    }

# if __name__ == '__main__':
#     dataset = FAIR1MDataset.METAINFO
