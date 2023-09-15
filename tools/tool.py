import copy
import os
import os.path as osp
import re
import tempfile
import zipfile
from collections import OrderedDict, defaultdict
from typing import List, Optional, Sequence, Union

import numpy as np
import torch
from mmcv.ops import nms_quadri, nms_rotated
from mmengine.evaluator import BaseMetric
from mmengine.fileio import dump
from mmengine.logging import MMLogger

from mmrotate.evaluation import eval_rbbox_map
from mmrotate.registry import METRICS
from mmrotate.structures.bbox import rbox2qbox
from mmrotate.datasets import (DOTADataset, DOTAv15Dataset, DOTAv2Dataset, FAIR1MDataset)
from typing import Any, List, Optional, Sequence, Union
from mmengine.fileio import dump
from mmengine.logging import print_log
from mmengine.evaluator import BaseMetric
from mmengine.evaluator.metric import _to_cpu
# from mmrotate.structures.bbox import rbox2qbox
from mmrotate.post_processing import postprocess_rotated
from mmdet.structures.bbox import get_box_tensor
from tqdm import tqdm

CLASS_NAME = dict(DOTADataset=DOTADataset.METAINFO['classes'],
                  DOTAv15Dataset=DOTAv15Dataset.METAINFO['classes'],
                  DOTAv2Dataset=DOTAv2Dataset.METAINFO['classes'],
                  FAIR1MDataset=FAIR1MDataset.METAINFO['classes'])

MAX_CUDA_LENGTH = 40000


def print_color_str(string, color='r'):
    if color.lower() in ['r', 'red']:
        code = '31'
    elif color.lower() in ['g', 'green']:
        code = '32'
    elif color.lower() in ['y', 'yellow']:
        code = '33'
    elif color.lower() in ['b', 'blue']:
        code = '34'
    elif color.lower() in ['m', 'magenta']:
        code = '35'
    elif color.lower() in ['c', 'cyan']:
        code = '36'
    else:
        raise NotImplementedError('Unsupported color code.')
    print(f'\033[1;{code}m {string}\033[0m')


def merge_results(results: Sequence[dict], dataset: str, test_cfg, only_cpu=False):
    collector = defaultdict(list)
    print_color_str('Collect result.', 'b')
    for idx, result in enumerate(tqdm(results)):
        img_id = result.get('img_id', idx)
        splitname = img_id.split('__')
        oriname = splitname[0]
        pattern1 = re.compile(r'__\d+___\d+')
        x_y = re.findall(pattern1, img_id)
        x_y_2 = re.findall(r'\d+', x_y[0])
        x, y = int(x_y_2[0]), int(x_y_2[1])
        labels = result['pred_instances']['labels'].numpy()
        bboxes = result['pred_instances']['bboxes'].numpy()
        scores = result['pred_instances']['scores'].numpy()
        ori_bboxes = bboxes.copy()

        ori_bboxes[..., :2] = ori_bboxes[..., :2] + np.array(
            [x, y], dtype=np.float32)

        label_dets = np.concatenate(
            [labels[:, np.newaxis], ori_bboxes, scores[:, np.newaxis]],
            axis=1)
        collector[oriname].append(label_dets)

    print_color_str('Merging...', 'b')
    id_list, dets_list = [], []
    for oriname, label_dets_list in tqdm(collector.items()):
        big_img_results = []
        label_dets = np.concatenate(label_dets_list, axis=0)
        labels, dets = label_dets[:, 0], label_dets[:, 1:]
        for i in range(len(CLASS_NAME[dataset])):
            if len(dets[labels == i]) == 0:
                big_img_results.append(np.empty((0, 9), dtype=dets.dtype))
            else:
                try:
                    if len(dets[labels == i]) > MAX_CUDA_LENGTH:
                        print_color_str(f'Image name: {oriname}, dets: {len(dets[labels == i])}', 'c')
                        cls_dets = torch.from_numpy(dets[labels == i]).float()
                    elif only_cpu:
                        cls_dets = torch.from_numpy(dets[labels == i]).float()
                    else:
                        torch.cuda.empty_cache()
                        cls_dets = torch.from_numpy(dets[labels == i]).float().cuda()
                    # cls_dets = torch.from_numpy(dets[labels == i])
                except:  # noqa: E722
                    cls_dets = torch.from_numpy(dets[labels == i])
                try:
                    results = postprocess_rotated(cls_dets[:, :5], cls_dets[:, -1], torch.from_numpy(labels[labels == i]).long(), test_cfg.deepcopy())
                except:
                    print_color_str(f'length: {len(dets[labels == i])}', 'r')
                    raise MemoryError
                # nms_dets, _ = nms_rotated(cls_dets[:, :5], cls_dets[:, -1], 0.1)
                box = get_box_tensor(results.bboxes)
                nms_dets = torch.cat([rbox2qbox(box).cpu(), results.scores[..., None].cpu()], dim=1).double()
                # big_img_results.append(torch.cat([rbox2qbox(nms_dets[:, :5]), nms_dets[:, 5, None]], dim=1).cpu().numpy())
                big_img_results.append(nms_dets.numpy())
        id_list.append(oriname)
        dets_list.append(big_img_results)
    return id_list, dets_list


@METRICS.register_module()
class DumpMergeResults(BaseMetric):
    def __init__(self,
                 out_file_path: str,
                 dataset_type: str,
                 # nms_iou: float = 0.1,
                 test_cfg=None,
                 collect_device: str = 'cpu',
                 collect_dir: Optional[str] = None) -> None:
        try:
            super().__init__(collect_device=collect_device, collect_dir=collect_dir)
        except:
            super().__init__(collect_device=collect_device)
        if not out_file_path.endswith(('.pkl', '.pickle')):
            raise ValueError('The output file must be a pkl file.')
        self.out_file_path = out_file_path
        self.dataset_type = dataset_type
        self.dataset_class = CLASS_NAME[dataset_type]
        # self.iou_thr = nms_iou
        self.test_cfg = test_cfg if test_cfg is not None else dict(score_thr=0.05, nms=dict(type='nms_rotated', iou_threshold=0.1))
        self.test_cfg['max_per_img'] = -1

    def process(self, data_batch: Any, predictions: Sequence[dict]) -> None:
        """transfer tensors in predictions to CPU."""
        self.results.extend(_to_cpu(predictions))

    def compute_metrics(self, results: list) -> dict:
        """dump the prediction results to a pickle file."""
        imageid, merge_result = merge_results(results, self.dataset_type, self.test_cfg)
        dump((imageid, merge_result, self.dataset_class), self.out_file_path)
        print_color_str(f'Dump result to {self.out_file_path}', 'g')
        print_log(
            f'Results has been saved to {self.out_file_path}.',
            logger='current')
        return {}

if __name__ == '__main__':
    a = {'a': 1, 'b':2, 'c':3}
    for i, j in tqdm(a.items()):
        print(f'{i}, {j}')