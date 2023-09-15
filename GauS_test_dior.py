import copy
import os
import shutil

from mmengine.config import Config
import pynvml
import argparse
from tools.GauS_tools import (print_color_str, get_latest_dir,
                              get_synth_name, get_file_sha256_name)


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
        cfg_filename = os.path.basename(cfg.filename).replace('.py', '')
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
        result_file = os.path.join(result_dir, f'{cfg_filename}_result_{synth}_{sha256_name}.json')
        commond = f'./tools/dist_test.sh {file} {pth_path} {n_gpus} ' \
                  f'--metric-out {result_file}'
        if cfg.get('with_tta', False):
            commond += f' --tta'
        print_color_str(commond, 'r')
        if cfg.get("with_tta", False):
            print_color_str('Use TTA: Enable', 'g')
        else:
            print_color_str('Use TTA: Disable', 'y')
        os.system(commond)
        latest_dir = get_latest_dir(cfg.work_dir)
        if latest_dir is not None:
            print_color_str(f'Delete test cache dir: {latest_dir}', 'g')
            shutil.rmtree(latest_dir)
        current_dir = copy.deepcopy(result_dir)