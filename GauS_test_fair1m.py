import copy
import os
import shutil
import glob
import os.path as osp
import tqdm
import json

from mmengine.config import Config
import numpy as np
import pynvml
import argparse
import pickle
from mmrotate.evaluation import eval_rbbox_map
from tools.GauS_tools import (print_color_str, get_latest_dir,
                              get_synth_name, get_file_sha256_name)


DATA_VAL_FOLDER = dict(DOTADataset='data/DOTA/val/labelTxt',
                       DOTAv15Dataset='data/DOTAv1_5/val/labelTxt',
                       DOTAv2Dataset='data/DOTAv2_0/val/labelTxt',
                       FAIR1MDataset='data/FAIR1M/validation/labelTxt')


def load_data_list(ann_folder, classes, ids=None):
    """Load annotations from an annotation file named as ``self.ann_file``
    Returns:
        List[dict]: A list of annotation.
    """  # noqa: E501
    cls_map = {c: i for i, c in enumerate(classes)}  # in mmdet v2.0 label is 0-based
    data_list = []
    txt_files = glob.glob(osp.join(ann_folder, '*.txt'))
    if ids:
        ann_ids = [os.path.splitext(os.path.basename(f))[0] for f in txt_files]
        idx = [ann_ids.index(id) for id in ids]
        txt_files = [txt_files[id] for id in idx]
    if len(txt_files) == 0:
        raise ValueError('There is no txt file in '
                         f'{ann_folder}')
    for txt_file in txt_files:
        img_id = osp.split(txt_file)[1][:-4]
        bboxes = []
        labels = []

        with open(txt_file) as f:
            s = [line.strip() for line in f.readlines() if len(line.strip()) > 0]
            s = s[2:]
            for si in s:
                bbox_info = si.split()
                bboxes.append([float(i) for i in bbox_info[:8]])
                cls_name = bbox_info[8]
                labels.append(cls_map[cls_name])
        data_list.append(dict(img_id=img_id,
                              bboxes=np.array(bboxes),
                              labels=np.array(labels),
                              labels_ignore=np.empty((0, )),
                              bboxes_ignore=np.empty((0, 8))
                              ))
    return data_list


def eval_val(result_path, ann_dirs):
    with open(result_path, 'rb') as f:
        ids, result, CLASS = pickle.load(f)
    data_infos = load_data_list(ann_dirs, CLASS, ids)
    ious = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    APs = dict()
    APs_all = dict()
    summary_str = ''
    for iou in tqdm.tqdm(ious):
        mean_ap, ap = eval_rbbox_map(result, data_infos, box_type='qbox', iou_thr=iou, dataset=CLASS, nproc=16, logger='silent')
        iou_key = f'AP{int(iou * 100)}'
        APs[iou_key] = mean_ap
        APs_all[iou_key] = {name: ap[i]['ap'].item() for i, name in enumerate(CLASS)}
        if iou_key == 'AP50':
            summary_str += ', '.join([f"{name}: {ap[i]['ap'].item()}" for i, name in enumerate(CLASS)])
    summary = {name: dict(AP50=APs_all['AP50'][name],
                          AP55=APs_all['AP55'][name],
                          AP60=APs_all['AP60'][name],
                          AP65=APs_all['AP65'][name],
                          AP70=APs_all['AP70'][name],
                          AP75=APs_all['AP75'][name],
                          AP80=APs_all['AP80'][name],
                          AP85=APs_all['AP85'][name],
                          AP90=APs_all['AP90'][name],
                          AP95=APs_all['AP95'][name],
                          mAP=sum([item[name] for item in APs_all.values()])/len(ious)) for i, name in enumerate(CLASS)}
    APs['mAP'] = sum(list(APs.values())) / len(ious)
    APs_all['AP50']['mAP'] = APs['AP50']
    APs_all['AP55']['mAP'] = APs['AP55']
    APs_all['AP60']['mAP'] = APs['AP60']
    APs_all['AP65']['mAP'] = APs['AP65']
    APs_all['AP70']['mAP'] = APs['AP70']
    APs_all['AP75']['mAP'] = APs['AP75']
    APs_all['AP80']['mAP'] = APs['AP80']
    APs_all['AP85']['mAP'] = APs['AP85']
    APs_all['AP90']['mAP'] = APs['AP90']
    APs_all['AP95']['mAP'] = APs['AP95']
    summary_str += f', AP50: {APs["AP50"]}, AP75: {APs["AP75"]}, mAP: {APs["mAP"]}.'
    with open(os.path.splitext(result_path)[0] + '.json', 'w') as f:
        result = dict(AP=APs, Summary=summary, ALL=APs_all)
        result['dota/summary'] = summary_str
        json.dump(result, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test a model')
    parser.add_argument('tasklist', help='train task list path')
    args = parser.parse_args()
    current_dir = ''
    with open(args.tasklist, 'r') as f:
        file_list = [file.strip() for file in f.readlines() if len(file.strip()) > 0]
    for file in file_list:
        if file.startswith('#'):
            print_color_str(f'Skip: {file[1:].strip()}', 'b')
            continue
        if not os.path.isfile(file):
            if file == 'merge_excel':
                if os.path.exists(current_dir) and os.path.isdir(current_dir):
                    print_color_str(f'Generate excel file, {current_dir}')
                    os.system(f'python3 get_excel.py {current_dir}')
                else:
                    print_color_str(f'Warring: dir is invalid, {current_dir}.\nSkip', 'y')
            else:
                print_color_str(f'Warring: invalid order, {file}.\nSkip', 'y')
            continue
        cfg = Config.fromfile(file)
        cfg_filename = os.path.splitext(os.path.basename(cfg.filename))[0]
        test_cfg = copy.deepcopy(cfg.model.test_cfg)
        if test_cfg.get('rcnn'):
            cfg_ = copy.deepcopy(test_cfg.rcnn)
        else:
            cfg_ = test_cfg
        synth_cfg = cfg_.get('synth_cfg', None)
        if synth_cfg:
            synth = get_synth_name(synth_cfg)
        else:
            print_color_str(f'Origin NMS', 'g')
            synth = 'origin_NMS'
        sha256_name = get_file_sha256_name(cfg.file)
        result_dir = os.path.join(cfg.work_dir, sha256_name)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        pth_path = os.path.join(cfg.work_dir, cfg.file)
        pynvml.nvmlInit()
        n_gpus = pynvml.nvmlDeviceGetCount()
        result_file = os.path.join(result_dir, f'{cfg_filename}_result_{synth}_{sha256_name}.pkl')
        commond = f'./tools/dist_test.sh {file} {pth_path} {n_gpus} ' \
                  f'--merge-out {result_file}'
        os.system(commond)
        latest_dir = get_latest_dir(cfg.work_dir)
        if latest_dir is not None:
            print_color_str(f'Delete test cache dir: {latest_dir}', 'g')
            shutil.rmtree(latest_dir)
        print_color_str('Evaluating', 'g')
        eval_val(result_file, DATA_VAL_FOLDER[cfg.dataset_type])
        print_color_str('Done', 'g')
        current_dir = copy.deepcopy(result_dir)