# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import os
import re
import tempfile
import zipfile
from typing import Sequence
from collections import OrderedDict, defaultdict

import numpy as np
import torch
from mmcv.ops import nms_quadri, nms_rotated
from mmengine.logging import MMLogger

from mmrotate.registry import METRICS
from mmrotate.structures.bbox import rbox2qbox


from .dota_metric import DOTAMetric

@METRICS.register_module()
class NoneMetric(DOTAMetric):
    def compute_metrics(self, results: list) -> dict:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.
        Returns:
            dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """
        logger: MMLogger = MMLogger.get_current_instance()
        gts, preds = zip(*results)

        tmp_dir = None
        if self.outfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            outfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            outfile_prefix = self.outfile_prefix

        eval_results = OrderedDict()
        return eval_results

    def merge_results(self, results: Sequence[dict],
                      outfile_prefix: str) -> str:
        """Merge patches' predictions into full image's results and generate a
        zip file for DOTA online evaluation.

        You can submit it at:
        https://captain-whu.github.io/DOTA/evaluation.html

        Args:
            results (Sequence[dict]): Testing results of the
                dataset.
            outfile_prefix (str): The filename prefix of the zip files. If the
                prefix is "somepath/xxx", the zip files will be named
                "somepath/xxx/xxx.zip".
        """
        collector = defaultdict(list)

        for idx, result in enumerate(results):
            img_id = result.get('img_id', idx)
            splitname = img_id.split('__')
            oriname = splitname[0]
            pattern1 = re.compile(r'__\d+___\d+')
            x_y = re.findall(pattern1, img_id)
            x_y_2 = re.findall(r'\d+', x_y[0])
            x, y = int(x_y_2[0]), int(x_y_2[1])
            labels = result['labels']
            bboxes = result['bboxes']
            scores = result['scores']
            ori_bboxes = bboxes.copy()
            if self.predict_box_type == 'rbox':
                ori_bboxes[..., :2] = ori_bboxes[..., :2] + np.array(
                    [x, y], dtype=np.float32)
            elif self.predict_box_type == 'qbox':
                ori_bboxes[..., :] = ori_bboxes[..., :] + np.array(
                    [x, y, x, y, x, y, x, y], dtype=np.float32)
            else:
                raise NotImplementedError
            label_dets = np.concatenate(
                [labels[:, np.newaxis], ori_bboxes, scores[:, np.newaxis]],
                axis=1)
            collector[oriname].append(label_dets)

        id_list, dets_list = [], []
        for oriname, label_dets_list in collector.items():
            big_img_results = []
            label_dets = np.concatenate(label_dets_list, axis=0)
            labels, dets = label_dets[:, 0], label_dets[:, 1:]
            for i in range(len(self.dataset_meta['classes'])):
                if len(dets[labels == i]) == 0:
                    big_img_results.append(dets[labels == i])
                else:
                    try:
                        cls_dets = torch.from_numpy(dets[labels == i]).cuda()
                    except:  # noqa: E722
                        cls_dets = torch.from_numpy(dets[labels == i])
                    if self.predict_box_type == 'rbox':
                        nms_dets, _ = nms_rotated(cls_dets[:, :5],
                                                  cls_dets[:,
                                                           -1], self.iou_thr)
                    elif self.predict_box_type == 'qbox':
                        nms_dets, _ = nms_quadri(cls_dets[:, :8],
                                                 cls_dets[:, -1], self.iou_thr)
                    else:
                        raise NotImplementedError
                    big_img_results.append(nms_dets.cpu().numpy())
            id_list.append(oriname)
            dets_list.append(big_img_results)

        if not osp.exists(outfile_prefix):
            os.makedirs(outfile_prefix)
        file_list = []
        for img_id, dets_per_cls in zip(id_list, dets_list):
            file_path = osp.join(outfile_prefix, f'{img_id}.txt')
            file_list.append(file_path)
            with open(file_path, 'w') as f:
                for i, dets in enumerate(dets_per_cls):
                    if dets.size == 0:
                        continue
                    th_dets = torch.from_numpy(dets)
                    if self.predict_box_type == 'rbox':
                        rboxes, scores = torch.split(th_dets, (5, 1), dim=-1)
                        qboxes = rbox2qbox(rboxes)
                    elif self.predict_box_type == 'qbox':
                        qboxes, scores = torch.split(th_dets, (8, 1), dim=-1)
                    else:
                        raise NotImplementedError
                    for qbox, score in zip(qboxes, scores):
                        cls = self.dataset_meta['classes'][i]
                        txt_element = [cls, str(round(float(score), 2))] + [f'{p:.2f}' for p in qbox]
                        f.writelines(txt_element)

        target_name = osp.split(outfile_prefix)[-1]
        zip_path = osp.join(outfile_prefix, target_name + '.zip')
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as t:
            for f in file_list:
                t.write(f, osp.split(f)[-1])

        return zip_path
