# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.models.dense_heads import RetinaHead
from mmdet.structures.bbox import get_box_tensor, get_box_wh, scale_boxes
from torch import Tensor
from mmengine.structures import InstanceData
from mmengine.config import ConfigDict
from typing import Optional
from mmrotate.post_processing import postprocess_rotated
from mmrotate.registry import MODELS


@MODELS.register_module()
class RotatedRetinaHead(RetinaHead):
    """Rotated retina head.

    Args:
        loss_bbox_type (str): Set the input type of ``loss_bbox``.
            Defaults to 'normal'.
    """

    def __init__(self,
                 *args,
                 loss_bbox_type: str = 'normal',
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.loss_bbox_type = loss_bbox_type

    def loss_by_feat_single(self, cls_score: Tensor, bbox_pred: Tensor,
                            anchors: Tensor, labels: Tensor,
                            label_weights: Tensor, bbox_targets: Tensor,
                            bbox_weights: Tensor, avg_factor: int) -> tuple:
        """Calculate the loss of a single scale level based on the features
        extracted by the detection head.

        Args:
            cls_score (Tensor): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W).
            bbox_pred (Tensor): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W).
            anchors (Tensor): Box reference for each scale level with shape
                (N, num_total_anchors, 4).
            labels (Tensor): Labels of each anchors with shape
                (N, num_total_anchors).
            label_weights (Tensor): Label weights of each anchor with shape
                (N, num_total_anchors)
            bbox_targets (Tensor): BBox regression targets of each anchor
                weight shape (N, num_total_anchors, 4).
            bbox_weights (Tensor): BBox regression loss weights of each anchor
                with shape (N, num_total_anchors, 4).
            avg_factor (int): Average factor that is used to average the loss.

        Returns:
            tuple: loss components.
        """
        # classification loss
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        cls_score = cls_score.permute(0, 2, 3,
                                      1).reshape(-1, self.cls_out_channels)
        loss_cls = self.loss_cls(
            cls_score, labels, label_weights, avg_factor=avg_factor)
        # regression loss
        target_dim = bbox_targets.size(-1)
        bbox_targets = bbox_targets.reshape(-1, target_dim)
        bbox_weights = bbox_weights.reshape(-1, target_dim)
        bbox_pred = bbox_pred.permute(0, 2, 3,
                                      1).reshape(-1,
                                                 self.bbox_coder.encode_size)

        if self.reg_decoded_bbox and (self.loss_bbox_type != 'kfiou'):
            # When the regression loss (e.g. `IouLoss`, `GIouLoss`)
            # is applied directly on the decoded bounding boxes, it
            # decodes the already encoded coordinates to absolute format.
            anchors = anchors.reshape(-1, anchors.size(-1))
            bbox_pred = self.bbox_coder.decode(anchors, bbox_pred)
            bbox_pred = get_box_tensor(bbox_pred)

        if self.loss_bbox_type == 'normal':
            loss_bbox = self.loss_bbox(
                bbox_pred, bbox_targets, bbox_weights, avg_factor=avg_factor)
        elif self.loss_bbox_type == 'kfiou':
            # When the regression loss (e.g. `KFLoss`)
            # is applied on both the delta and decoded boxes.
            anchors = anchors.reshape(-1, anchors.size(-1))
            bbox_pred_decode = self.bbox_coder.decode(anchors, bbox_pred)
            bbox_pred_decode = get_box_tensor(bbox_pred_decode)
            bbox_targets_decode = self.bbox_coder.decode(anchors, bbox_targets)
            bbox_targets_decode = get_box_tensor(bbox_targets_decode)
            loss_bbox = self.loss_bbox(
                bbox_pred,
                bbox_targets,
                bbox_weights,
                pred_decode=bbox_pred_decode,
                targets_decode=bbox_targets_decode,
                avg_factor=avg_factor)
        else:
            raise NotImplementedError

        return loss_cls, loss_bbox

    def _bbox_post_process(self,
                           results: InstanceData,
                           cfg: ConfigDict,
                           rescale: bool = False,
                           with_nms: bool = True,
                           img_meta: Optional[dict] = None) -> InstanceData:
        """bbox post-processing method.

        The boxes would be rescaled to the original image scale and do
        the nms operation. Usually `with_nms` is False is used for aug test.

        Args:
            results (:obj:`InstaceData`): Detection instance results,
                each item has shape (num_bboxes, ).
            cfg (ConfigDict): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Default to False.
            with_nms (bool): If True, do nms before return boxes.
                Default to True.
            img_meta (dict, optional): Image meta info. Defaults to None.

        Returns:
            :obj:`InstanceData`: Detection results of each image
            after the post process.
            Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        if rescale:
            assert img_meta.get('scale_factor') is not None
            scale_factor = [1 / s for s in img_meta['scale_factor']]
            results.bboxes = scale_boxes(results.bboxes, scale_factor)

        if hasattr(results, 'score_factors'):
            # TODO： Add sqrt operation in order to be consistent with
            #  the paper.
            score_factors = results.pop('score_factors')
            results.scores = results.scores * score_factors

        # filter small size bboxes
        if cfg.get('min_bbox_size', -1) >= 0:
            w, h = get_box_wh(results.bboxes)
            valid_mask = (w > cfg.min_bbox_size) & (h > cfg.min_bbox_size)
            if not valid_mask.all():
                results = results[valid_mask]

        # TODO: deal with `with_nms` and `nms_cfg=None` in test_cfg
        if with_nms and results.bboxes.numel() > 0:
            bboxes = get_box_tensor(results.bboxes)
            results = postprocess_rotated(bboxes, results.scores, results.labels, cfg.deepcopy())
            # det_bboxes, keep_idxs = batched_nms(bboxes, results.scores,
            #                                     results.labels, cfg.nms)
            # results = results[keep_idxs]
            # # some nms would reweight the score, such as softnms
            # results.scores = det_bboxes[:, -1]
            # results = results[:cfg.max_per_img]

        return results