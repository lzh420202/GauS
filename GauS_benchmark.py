import os
import json

from mmengine.config import Config
import argparse
import copy
from tools.GauS_tools import (print_color_str, get_latest_dir, get_synth_name, get_file_sha256_name)
from default_synth_parameters import synth_cfg as default_synth_cfg


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test a model')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--max-iter', type=int, default=10000, help='num of max iter')
    parser.add_argument(
        '--log-interval', type=int, default=50, help='interval of logging')
    parser.add_argument(
        '--mode',
        choices=['cpu', 'gpu'],
        default='gpu',
        help='Which task do you want to go to benchmark')
    args = parser.parse_args()
    file = args.config

    ori_cfg = Config.fromfile(file)
    out_dir = os.path.join(ori_cfg.work_dir, get_file_sha256_name(ori_cfg.file) + '_FPS')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    results = dict()
    # default_synth_cfg = dict()
    default_synth_cfg['no_nms'] = dict(name='no_nms', mode='tensor')
    synth_cfg = copy.deepcopy(default_synth_cfg)
    for key, value in synth_cfg.items():
        cfg = copy.deepcopy(ori_cfg)
        cfg_filename = os.path.splitext(os.path.basename(cfg.filename))[0]
        test_cfg = copy.deepcopy(cfg.model.test_cfg)
        name = value.pop('name')
        mode = value.pop('mode', 'predict')
        if len(value) > 0:
            if test_cfg.get('rcnn'):
                cfg.model.test_cfg.rcnn['synth_cfg'] = value
            else:
                cfg.model.test_cfg['synth_cfg'] = value
            synth = get_synth_name(Config(value))
        else:
            if mode == 'tensor':
                print_color_str(f'NO NMS', 'g')
                synth = 'NO_NMS'
            else:
                print_color_str(f'Origin NMS', 'g')
                synth = 'origin_NMS'
        test_cfg_file = os.path.join(out_dir, f'{cfg_filename}_{name}.py')
        cfg.dump(test_cfg_file)
        print_color_str(f'{"".join(["+"]*20)}', 'm')
        print_color_str(f'Max iter: {args.max_iter}', 'c')
        print_color_str(f'Log interval: {args.log_interval}', 'c')
        pth_path = os.path.join(cfg.work_dir, cfg.file)
        out_file = os.path.join(out_dir, f'{synth}_FPS.txt')
        commond = f'python3 -m torch.distributed.launch ' \
                  f'--nproc_per_node=1 --master_port=29500 ' \
                  f'tools/analysis_tools/benchmark.py ' \
                  f'{test_cfg_file} --checkpoint {pth_path} --launcher pytorch ' \
                  f'--task rotated ' \
                  f'--mode {mode} ' \
                  f'--repeat-num 1 ' \
                  f'--max-iter {args.max_iter} ' \
                  f'--log-interval {args.log_interval} ' \
                  f'--out {out_file}'
        print_color_str(f'{commond}', 'g')
        os.system(commond)
        with open(out_file, 'r') as f:
            fps = [float(line) for line in f.readlines() if len(line.strip()) > 0][0]
        results[name] = dict(fps=fps, latency=1000.0 / fps)

    with open(os.path.join(out_dir, 'fps.json'), 'w') as f:
        json.dump(results, f, indent=4)