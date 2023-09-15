_base_ = ['../_base_/dota1_0/oriented-rcnn-le90_r50_fpn_1x_dota.py']

angle_version = 'le90'
model = dict(
    test_cfg=dict(
        rcnn=dict(
            nms_pre=2000,
            min_bbox_size=0,
            score_thr=0.05,
            nms=dict(type='nms_rotated', iou_threshold=0.1),
            max_per_img=2000
        )
    )
)