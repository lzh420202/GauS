import copy
import os
import shutil
import warnings

from mmengine.config import Config
import torch
import numpy as np
import pynvml
import argparse
import pickle
import zipfile
from datetime import datetime

from mmengine.structures import InstanceData
from mmdet.structures import DetDataSample
from mmrotate.post_processing import synth_rotated
from mmcv.ops.nms import batched_nms
from mmrotate.datasets.dota import DOTADataset
from mmrotate.structures.bbox.box_converters import rbox2qbox
import os.path as osp
from tqdm import tqdm
from tools.GauS_tools import (print_color_str, get_latest_dir,
                              get_synth_name, get_file_sha256_name)
from tools.tool import merge_results, CLASS_NAME
from mmengine.fileio import dump


def get_file_info(img_id):
    info = [i.strip('_') for i in img_id.split('__')]
    return info


def format_only(merge_file, outfile_prefix):
    with open(merge_file, 'rb') as f:
        id_list, dets_list, dataset_class = pickle.load(f)

    if osp.exists(outfile_prefix):
        warnings.warn(f'The outfile_prefix should be a non-exist path, '
                      f'but {outfile_prefix} is existing. ')
        shutil.rmtree(outfile_prefix)
    os.makedirs(outfile_prefix)

    files = [
        osp.join(outfile_prefix, 'Task1_' + cls + '.txt')
        for cls in dataset_class
    ]
    file_objs = [open(f, 'w') for f in files]
    print_color_str('Writing detections into text file...', 'b')
    for img_id, dets_per_cls in tqdm(zip(id_list, dets_list)):
        for f, dets in zip(file_objs, dets_per_cls):
            if dets.size == 0:
                continue
            th_dets = torch.from_numpy(dets)
            qboxes, scores = torch.split(th_dets, (8, 1), dim=-1)
            # qboxes = rbox2qbox(rboxes)

            for qbox, score in zip(qboxes, scores):
                txt_element = [img_id, str(round(float(score), 2))
                               ] + [f'{p:.2f}' for p in qbox]
                f.writelines(' '.join(txt_element) + '\n')

    for f in file_objs:
        f.close()
    print_color_str('Zip into one file', 'b')
    target_name = osp.split(outfile_prefix)[-1]
    zip_path = osp.join(outfile_prefix, target_name + '.zip')
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as t:
        for f in files:
            t.write(f, osp.split(f)[-1])
    print_color_str('Done', 'g')
    return zip_path


if __name__ == '__main__':
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    parser = argparse.ArgumentParser(description='Test a model')
    parser.add_argument('tasklist', help='train task list path')
    args = parser.parse_args()
    with open(args.tasklist, 'r') as f:
        file_list = [file.strip() for file in f.readlines() if len(file.strip()) > 0]
    for file in file_list:
        if file.startswith('#'):
            print_color_str(f'Skip: {file[1:].strip()}', 'b')
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
        result_dir = os.path.join(cfg.work_dir, sha256_name, f"{cfg_filename}_result_{synth}_{sha256_name}")
        if not os.path.exists(os.path.join(cfg.work_dir, sha256_name)):
            os.makedirs(os.path.join(cfg.work_dir, sha256_name))
        pth_path = os.path.join(cfg.work_dir, cfg.file)
        pynvml.nvmlInit()
        n_gpus = pynvml.nvmlDeviceGetCount()
        result_file = f'{result_dir}.pkl'
        # commond = f'./tools/dist_test.sh {file} {pth_path} {n_gpus} ' \
        #           f'--merge-out {result_file}'
        commond = f'./tools/dist_test.sh {file} {pth_path} {n_gpus} ' \
                  f'--out {result_file}'
        print_color_str(commond, 'g')
        os.system(commond)
        latest_dir = get_latest_dir(cfg.work_dir)
        if latest_dir is not None:
            print_color_str(f'Delete test cache dir: {latest_dir}', 'g')
            shutil.rmtree(latest_dir)
        with open(result_file, 'rb') as f:
            results = pickle.load(f)
        imageid, merge_result = merge_results(results, cfg.dataset_type, cfg_)
        torch.cuda.empty_cache()
        merge_result_file = result_file.replace('.pkl', '_merge.pkl')
        dump((imageid, merge_result, CLASS_NAME[cfg.dataset_type]), merge_result_file)
        print_color_str('Format only', 'g')
        zip_file = format_only(merge_result_file, result_dir)
        print_color_str('Clean up cache', 'b')
        print_color_str(f'Move {zip_file} to \"{os.path.join(cfg.work_dir, sha256_name)}\"', 'g')
        shutil.move(zip_file, os.path.join(cfg.work_dir, sha256_name))
        print_color_str(f'Delete cache dir: {result_dir}', 'g')
        shutil.rmtree(result_dir)
        print_color_str('All Done', 'g')