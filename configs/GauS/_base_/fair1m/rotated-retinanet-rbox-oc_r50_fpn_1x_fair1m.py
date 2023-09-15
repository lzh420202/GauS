_base_ = 'rotated-retinanet-rbox-le135_r50_fpn_1x_fair1m.py'

angle_version = 'oc'

model = dict(
    bbox_head=dict(
        anchor_generator=dict(angle_version=angle_version),
        bbox_coder=dict(
            angle_version=angle_version,
            norm_factor=None,
            edge_swap=False,
            proj_xy=False)))

optim_wrapper = dict(optimizer=dict(lr=1e-2))

train_dataloader = dict(batch_size=2, num_workers=2)

work_dir = r'work_dirs/GauS/fair1m/rotated_retinanet/oc/'

# file = 'rotated-retinanet-rbox-oc_r50_fpn_1x_fair1m-e5c497f1.pth'

file = 'rotated-retinanet-rbox-oc_r50_fpn_1x_fair1m-0cb7d363.pth'