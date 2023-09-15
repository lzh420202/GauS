# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import torch
from mmdet.models.losses import accuracy
from mmdet.models.roi_heads.bbox_heads import Shared2FCBBoxHead
from mmdet.structures.bbox import get_box_tensor, scale_boxes
from torch import Tensor

from mmrotate.registry import MODELS

import torch.nn.functional as F
from mmengine.config import ConfigDict
from mmengine.structures import InstanceData
from mmdet.models.utils import empty_instances
from mmdet.models.layers import multiclass_nms
from mmdet.models.utils import (filter_scores_and_topk, multi_apply,
                                select_single_mlvl, sigmoid_geometric_mean,
                                unmap)
from mmrotate.post_processing import postprocess_rotated, multiclass_preprocess

@MODELS.register_module()
class RotatedShared2FCBBoxHead(Shared2FCBBoxHead):
    """Rotated Shared2FC RBBox head.

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

    def loss(self,
             cls_score: Tensor,
             bbox_pred: Tensor,
             rois: Tensor,
             labels: Tensor,
             label_weights: Tensor,
             bbox_targets: Tensor,
             bbox_weights: Tensor,
             reduction_override: Optional[str] = None) -> dict:
        """Calculate the loss based on the network predictions and targets.

        Args:
            cls_score (Tensor): Classification prediction
                results of all class, has shape
                (batch_size * num_proposals_single_image, num_classes)
            bbox_pred (Tensor): Regression prediction results,
                has shape
                (batch_size * num_proposals_single_image, 4), the last
                dimension 4 represents [tl_x, tl_y, br_x, br_y].
            rois (Tensor): RoIs with the shape
                (batch_size * num_proposals_single_image, 5) where the first
                column indicates batch id of each RoI.
            labels (Tensor): Gt_labels for all proposals in a batch, has
                shape (batch_size * num_proposals_single_image, ).
            label_weights (Tensor): Labels_weights for all proposals in a
                batch, has shape (batch_size * num_proposals_single_image, ).
            bbox_targets (Tensor): Regression target for all proposals in a
                batch, has shape (batch_size * num_proposals_single_image, 4),
                the last dimension 4 represents [tl_x, tl_y, br_x, br_y].
            bbox_weights (Tensor): Regression weights for all proposals in a
                batch, has shape (batch_size * num_proposals_single_image, 4).
            reduction_override (str, optional): The reduction
                method used to override the original reduction
                method of the loss. Options are "none",
                "mean" and "sum". Defaults to None,

        Returns:
            dict: A dictionary of loss.
        """

        losses = dict()

        if cls_score is not None:
            avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
            if cls_score.numel() > 0:
                loss_cls_ = self.loss_cls(
                    cls_score,
                    labels,
                    label_weights,
                    avg_factor=avg_factor,
                    reduction_override=reduction_override)
                if isinstance(loss_cls_, dict):
                    losses.update(loss_cls_)
                else:
                    losses['loss_cls'] = loss_cls_
                if self.custom_activation:
                    acc_ = self.loss_cls.get_accuracy(cls_score, labels)
                    losses.update(acc_)
                else:
                    losses['acc'] = accuracy(cls_score, labels)
        if bbox_pred is not None:
            bg_class_ind = self.num_classes
            # 0~self.num_classes-1 are FG, self.num_classes is BG
            pos_inds = (labels >= 0) & (labels < bg_class_ind)
            # do not perform bounding box regression for BG anymore.
            if pos_inds.any():
                if self.reg_decoded_bbox and (self.loss_bbox_type != 'kfiou'):
                    # When the regression loss (e.g. `IouLoss`,
                    # `GIouLoss`, `DIouLoss`) is applied directly on
                    # the decoded bounding boxes, it decodes the
                    # already encoded coordinates to absolute format.
                    bbox_pred = self.bbox_coder.decode(rois[:, 1:], bbox_pred)
                    bbox_pred = get_box_tensor(bbox_pred)
                if self.reg_class_agnostic:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), -1)[pos_inds.type(torch.bool)]
                else:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), self.num_classes,
                        -1)[pos_inds.type(torch.bool),
                            labels[pos_inds.type(torch.bool)]]
                if self.loss_bbox_type == 'normal':
                    losses['loss_bbox'] = self.loss_bbox(
                        pos_bbox_pred,
                        bbox_targets[pos_inds.type(torch.bool)],
                        bbox_weights[pos_inds.type(torch.bool)],
                        avg_factor=bbox_targets.size(0),
                        reduction_override=reduction_override)
                elif self.loss_bbox_type == 'kfiou':
                    # When the regression loss (e.g. `KFLoss`)
                    # is applied on both the delta and decoded boxes.
                    bbox_pred_decode = self.bbox_coder.decode(
                        rois[:, 1:], bbox_pred)
                    bbox_pred_decode = get_box_tensor(bbox_pred_decode)
                    bbox_targets_decode = self.bbox_coder.decode(
                        rois[:, 1:], bbox_targets)
                    bbox_targets_decode = get_box_tensor(bbox_targets_decode)

                    if self.reg_class_agnostic:
                        pos_bbox_pred_decode = bbox_pred_decode.view(
                            bbox_pred_decode.size(0),
                            5)[pos_inds.type(torch.bool)]
                    else:
                        pos_bbox_pred_decode = bbox_pred_decode.view(
                            bbox_pred_decode.size(0), -1,
                            5)[pos_inds.type(torch.bool),
                               labels[pos_inds.type(torch.bool)]]

                    losses['loss_bbox'] = self.loss_bbox(
                        pos_bbox_pred,
                        bbox_targets[pos_inds.type(torch.bool)],
                        bbox_weights[pos_inds.type(torch.bool)],
                        pred_decode=pos_bbox_pred_decode,
                        targets_decode=bbox_targets_decode[pos_inds.type(
                            torch.bool)],
                        avg_factor=bbox_targets.size(0),
                        reduction_override=reduction_override)
                else:
                    raise NotImplementedError
            else:
                losses['loss_bbox'] = bbox_pred[pos_inds].sum()
        return losses


@MODELS.register_module()
class Shared2FCBBoxHeadGauS(Shared2FCBBoxHead):
    def _predict_by_feat_single(
            self,
            roi: Tensor,
            cls_score: Tensor,
            bbox_pred: Tensor,
            img_meta: dict,
            rescale: bool = False,
            rcnn_test_cfg: Optional[ConfigDict] = None) -> InstanceData:
        """Transform a single image's features extracted from the head into
        bbox results.

        Args:
            roi (Tensor): Boxes to be transformed. Has shape (num_boxes, 5).
                last dimension 5 arrange as (batch_index, x1, y1, x2, y2).
            cls_score (Tensor): Box scores, has shape
                (num_boxes, num_classes + 1).
            bbox_pred (Tensor): Box energies / deltas.
                has shape (num_boxes, num_classes * 4).
            img_meta (dict): image information.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.
            rcnn_test_cfg (obj:`ConfigDict`): `test_cfg` of Bbox Head.
                Defaults to None

        Returns:
            :obj:`InstanceData`: Detection results of each image\
            Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        results = InstanceData()
        if roi.shape[0] == 0:
            return empty_instances([img_meta],
                                   roi.device,
                                   task_type='bbox',
                                   instance_results=[results],
                                   box_type=self.predict_box_type,
                                   use_box_type=False,
                                   num_classes=self.num_classes,
                                   score_per_cls=rcnn_test_cfg is None)[0]

        # some loss (Seesaw loss..) may have custom activation
        if self.custom_cls_channels:
            scores = self.loss_cls.get_activation(cls_score)
        else:
            scores = F.softmax(
                cls_score, dim=-1) if cls_score is not None else None

        img_shape = img_meta['img_shape']
        num_rois = roi.size(0)
        # bbox_pred would be None in some detector when with_reg is False,
        # e.g. Grid R-CNN.
        if bbox_pred is not None:
            num_classes = 1 if self.reg_class_agnostic else self.num_classes
            roi = roi.repeat_interleave(num_classes, dim=0)
            bbox_pred = bbox_pred.view(-1, self.bbox_coder.encode_size)
            bboxes = self.bbox_coder.decode(
                roi[..., 1:], bbox_pred, max_shape=img_shape)
        else:
            bboxes = roi[:, 1:].clone()
            if img_shape is not None and bboxes.size(-1) == 4:
                bboxes[:, [0, 2]].clamp_(min=0, max=img_shape[1])
                bboxes[:, [1, 3]].clamp_(min=0, max=img_shape[0])

        if rescale and bboxes.size(0) > 0:
            assert img_meta.get('scale_factor') is not None
            scale_factor = [1 / s for s in img_meta['scale_factor']]
            bboxes = scale_boxes(bboxes, scale_factor)

        # Get the inside tensor when `bboxes` is a box type
        bboxes = get_box_tensor(bboxes)
        box_dim = bboxes.size(-1)
        bboxes = bboxes.view(num_rois, -1)

        if rcnn_test_cfg is None:
            # This means that it is aug test.
            # It needs to return the raw results without nms.
            results.bboxes = bboxes
            results.scores = scores
        else:
            filtered_boxes, keep_scores, keep_labels = multiclass_preprocess(bboxes, scores,
                                                                             rcnn_test_cfg.score_thr,
                                                                             box_dim=box_dim)
            if filtered_boxes.numel() == 0:
                results.bboxes = filtered_boxes
                results.scores = keep_scores
                results.labels = keep_labels
                return results
            results = postprocess_rotated(filtered_boxes, keep_scores, keep_labels, rcnn_test_cfg.deepcopy())
            results.bboxes = get_box_tensor(results.bboxes)

            # det_bboxes, det_labels = multiclass_nms(
            #     bboxes,
            #     scores,
            #     rcnn_test_cfg.score_thr,
            #     rcnn_test_cfg.nms,
            #     rcnn_test_cfg.max_per_img,
            #     box_dim=box_dim)
            # results.bboxes = det_bboxes[:, :-1]
            # results.scores = det_bboxes[:, -1]
            # results.labels = det_labels
        return results