_base_ = 'rotated-retinanet-rbox-le135_r50_fpn_1x_dior.py'

angle_version = 'oc'

model = dict(
    bbox_head=dict(
        anchor_generator=dict(angle_version=angle_version),
        bbox_coder=dict(
            angle_version=angle_version,
            norm_factor=None,
            edge_swap=False,
            proj_xy=False)))

work_dir = r'work_dirs/GauS/dior/rotated_retinanet/oc/'

# file = 'rotated-retinanet-rbox-oc_r50_fpn_1x_dior-5f4951bb.pth'
file = 'rotated-retinanet-rbox-oc_r50_fpn_1x_dior-c751df5e.pth'