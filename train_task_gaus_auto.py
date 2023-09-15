import argparse
import os
from datetime import datetime

from mmengine.config import Config
import shutil
from default_synth_parameters import synth_cfg as default_synth_cfg
import copy


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config list file path')
    parser.add_argument('num_gpus', help='train config list file path')
    parser.add_argument('--outfile', default=None)
    args = parser.parse_args()

    return args


def get_latest_dir(target):
    dirs = [file for file in os.listdir(target) if os.path.isdir(os.path.join(target, file))]
    current = datetime.now()
    times = [(current - datetime.strptime(d, '%Y%m%d_%H%M%S')).total_seconds() for d in dirs]
    new_ = list(zip(dirs, times))
    new_.sort(key=lambda pair: pair[1])
    return os.path.join(target, new_[0][0])


def main():
    args = parse_args()
    path = args.config
    outfile = args.outfile
    time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    if outfile is None:
        outfile = os.path.join(os.getcwd(), time_str)
    log_f = open(outfile, 'w')
    log_f.write(f'Begin time {time_str}\n')
    with open(path, 'r') as f:
        file_list = [file.strip() for file in f.readlines() if len(file.strip()) > 0]
    cmds = []
    # test_cmds = []
    save_dirs = []
    device_num = args.num_gpus
    valid_files = []
    for file in file_list:
        if file.startswith('#'):
            print(file)
            continue
        try:
            split_file = [line.strip() for line in file.split(' ') if len(line.strip()) > 0]
            if len(split_file) == 1:
                file = split_file[0]
                options = ''
            elif len(split_file) >= 2:
                file = split_file[0]
                options = ' ' + ' '.join(split_file[1:])
            else:
                raise ValueError
            valid_files.append(file)
            cfg = Config.fromfile(file)
            cmds.append(f'./tools/dist_train.sh {file} {device_num}{options}')
            # test_cmds.append(f'./tools/dist_test.sh {file}')
            save_dirs.append(cfg.work_dir)
        except Exception as e:
            log_f.write(f'There are something wrong when parse {file}.\n\t{e}')
            print(f'There are something wrong when parse {file}.\n\t{e}')
    log_f.write(f'Valid config file: {len(cmds)}\n')
    for cmd in cmds:
        log_f.write(f'{cmd}\n')
    log_f.write('\n')
    result_list = ['20230603_200829', '20230603_231847',
                   '20230604_022025', '20230604_052423']
    for cmd, valid_file, save_dir, result_dir in zip(cmds, valid_files, save_dirs, result_list):
        try:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            # os.system(cmd)
            # result_dir = get_latest_dir(save_dir)
            result_dir = os.path.join(save_dir, result_dir)
            file_name = os.path.splitext(os.path.basename(valid_file))[0]
            with open(os.path.join(save_dir, "last_checkpoint"), 'r') as f:
                pth_path = [line.strip() for line in f.readlines() if len(line.strip()) > 0][-1]
            # shutil.copy(pth_path, result_dir)
            # new_pth_path = os.path.join(result_dir, os.path.basename(pth_path))
            # test_save_dir = os.path.join(result_dir, file_name)
            # if os.path.exists(test_save_dir):
            #     shutil.rmtree(test_save_dir)
            #     log_f.write(f'Warning!!! Remove path {test_save_dir}\n')
            test_list_file = os.path.join(result_dir, 'test_list.txt')
            with open(test_list_file, 'w') as f:
                cfg = Config.fromfile(valid_file)
                nms_file = os.path.join(result_dir, f'{file_name}_nms.py')
                cfg.work_dir = result_dir
                # cfg.file = pth_path
                cfg['file'] = os.path.basename(pth_path)
                cfg.dump(nms_file)
                f.write(f"{nms_file}\n")
                for key in default_synth_cfg.keys():
                    CFG = copy.deepcopy(cfg)
                    cfg_name = os.path.join(result_dir,
                                            f'{file_name}_{default_synth_cfg[key]["name"]}.py')
                    added_cfg = copy.deepcopy(default_synth_cfg[key])
                    _ = added_cfg.pop('name', None)

                    test_cfg = copy.deepcopy(CFG.model.test_cfg)
                    if test_cfg.get('rcnn'):
                        CFG.model.test_cfg.rcnn['synth_cfg'] = added_cfg
                    else:
                        CFG.model.test_cfg['synth_cfg'] = added_cfg
                    CFG.dump(cfg_name)
                    f.write(f"{cfg_name}\n")
            test_cmd = f'python3 GauS_test_dior.py {test_list_file}'
            os.system(test_cmd)
            collector_cmd = f'python3 get_excel.py {result_dir}'
            os.system(collector_cmd)

            # cfg = Config.fromfile(os.path.join(save_dir, f'{file_name}.py'))
            # cfg.test_evaluator.outfile_prefix = test_save_dir
            # cfg_file = os.path.join(result_dir, f'{file_name}.py')
            # cfg.dump(cfg_file)
            # test_cmd = f'./tools/dist_test.sh {cfg_file} {pth_path} {device_num} ' \
            #            f'--out {os.path.join(result_dir, "predictions.pkl")}'
            # os.system(test_cmd)
            # test_generate = get_latest_dir(save_dir)
            # shutil.rmtree(test_generate)
            # shutil.move(os.path.join(result_dir, file_name, f'{file_name}.zip'),
            #             os.path.join(result_dir, f'{file_name}.zip'))
            # shutil.rmtree(os.path.join(result_dir, file_name))

            log_f.write(f'{cmd}\tSucceed\n')

        except Exception as e:
            log_f.write(f'{cmd}\tFailed\nDetail: {e}\n\n')

    log_f.write('Done\n')
    log_f.write(f'End time {datetime.now().strftime("%Y%m%d_%H%M%S")}\n')
    log_f.close()

if __name__ == '__main__':
    main()