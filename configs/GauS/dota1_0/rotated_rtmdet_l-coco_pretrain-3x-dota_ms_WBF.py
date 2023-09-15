_base_ = '../../rotated_rtmdet/rotated_rtmdet_l-coco_pretrain-3x-dota_ms.py'


model = dict(
    test_cfg=dict(
        nms_pre=2000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms_rotated', iou_threshold=0.1),
        max_per_img=2000,
        # synth_cfg=dict(synth_thr=0.5, synth_method=2, alpha=1.0, beta=6.0) # GauS
        # synth_cfg=dict(synth_thr=0.5, synth_method=1, alpha=1.0, beta=1.0) # NMW
        # synth_cfg=dict(synth_thr=0.5, synth_method=1, alpha=1.0, beta=6.0) # Direct
        synth_cfg=dict(method=1, iou_thr=0.1) # WBF
        # synth_cfg=dict(method=2, iou_thr=0.1, sigma=0.5) # Soft-NMS
        # synth_cfg=dict(method=3, iou_thr=0.1)  # DIoU-NMS
    ))

work_dir = r'work_dirs/GauS/dota1_0/rtmdet_l/'

file = 'rotated_rtmdet_l-coco_pretrain-3x-dota_ms-06d248a2.pth'