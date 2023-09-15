from typing import List, Tuple

import torch

from torch import Tensor

from mmcv.ops import batched_nms
from mmdet.models.test_time_augs import DetTTAModel
from mmrotate.registry import MODELS
from mmengine.structures import InstanceData

from mmdet.structures import DetDataSample
from mmrotate.post_processing import postprocess_rotated, multiclass_preprocess
from mmdet.structures.bbox import get_box_tensor


def bbox_flip(bboxes: Tensor,
              img_shape: Tuple[int],
              direction: str = 'horizontal') -> Tensor:
    """Flip bboxes horizontally or vertically.
    Args:
        bboxes (Tensor): Shape (..., 5*k)
        img_shape (Tuple[int]): Image shape.
        direction (str): Flip direction, options are "horizontal", "vertical",
            "diagonal". Default: "horizontal"
    Returns:
        Tensor: Flipped bboxes.
    """
    assert bboxes.shape[-1] % 5 == 0
    assert direction in ['horizontal', 'vertical', 'diagonal']
    flipped = bboxes.clone()
    if direction == 'horizontal':
        flipped[..., 0] = img_shape[1] - flipped[..., 0]
        flipped[..., 4] = -flipped[..., 4]
    elif direction == 'vertical':
        flipped[..., 1] = img_shape[0] - flipped[..., 1]
        flipped[..., 4] = -flipped[..., 4]
    else:
        flipped[..., 0] = img_shape[1] - flipped[..., 0]
        flipped[..., 1] = img_shape[0] - flipped[..., 1]
    return flipped


@MODELS.register_module()
class RotatedTTAModel(DetTTAModel):

    def merge_aug_bboxes(self, aug_bboxes: List[Tensor],
                         aug_scores: List[Tensor],
                         img_metas: List[str]) -> Tuple[Tensor, Tensor]:
        """Merge augmented detection bboxes and scores.
        Args:
            aug_bboxes (list[Tensor]): shape (n, 5*#class)
            aug_scores (list[Tensor] or None): shape (n, #class)
        Returns:
            tuple[Tensor]: ``bboxes`` with shape (n,5), where
            4 represent (x, y, w, h, t)
            and ``scores`` with shape (n,).
        """
        recovered_bboxes = []
        for bboxes, img_info in zip(aug_bboxes, img_metas):
            ori_shape = img_info['ori_shape']
            flip = img_info['flip']
            flip_direction = img_info['flip_direction']
            if flip:
                bboxes = bbox_flip(
                    bboxes=bboxes,
                    img_shape=ori_shape,
                    direction=flip_direction)
            recovered_bboxes.append(bboxes)
        bboxes = torch.cat(recovered_bboxes, dim=0)
        if aug_scores is None:
            return bboxes
        else:
            scores = torch.cat(aug_scores, dim=0)
            return bboxes, scores

    def _merge_single_sample(
            self, data_samples: List[DetDataSample]) -> DetDataSample:
        """Merge predictions which come form the different views of one image
        to one prediction.

        Args:
            data_samples (List[DetDataSample]): List of predictions
            of enhanced data which come form one image.
        Returns:
            List[DetDataSample]: Merged prediction.
        """
        aug_bboxes = []
        aug_scores = []
        aug_labels = []
        img_metas = []
        # TODO: support instance segmentation TTA
        assert data_samples[0].pred_instances.get('masks', None) is None, \
            'TTA of instance segmentation does not support now.'
        for data_sample in data_samples:
            aug_bboxes.append(data_sample.pred_instances.bboxes.cpu())
            aug_scores.append(data_sample.pred_instances.scores.cpu())
            aug_labels.append(data_sample.pred_instances.labels.cpu())
            img_metas.append(data_sample.metainfo)

        merged_bboxes, merged_scores = self.merge_aug_bboxes(
            aug_bboxes, aug_scores, img_metas)
        merged_labels = torch.cat(aug_labels, dim=0)

        if merged_bboxes.numel() == 0:
            return data_samples[0]

        # det_bboxes, keep_idxs = batched_nms(merged_bboxes, merged_scores,
        #                                     merged_labels, self.tta_cfg.nms)

        results = postprocess_rotated(merged_bboxes, merged_scores, merged_labels, self.tta_cfg.deepcopy())
        results.bboxes = get_box_tensor(results.bboxes)
        # det_bboxes = det_bboxes[:self.tta_cfg.max_per_img]
        # det_labels = merged_labels[keep_idxs][:self.tta_cfg.max_per_img]

        # results = InstanceData()
        # _det_bboxes = det_bboxes.clone()
        # results.bboxes = _det_bboxes[:, :-1]
        # results.scores = _det_bboxes[:, -1]
        # results.labels = det_labels
        det_results = data_samples[0]
        det_results.pred_instances = results
        return det_results