import argparse
import os
import sys
from datetime import datetime
import signal

from mmengine.config import Config
import logging
import shutil
import pynvml
from tools.GauS_tools import (print_color_str, get_latest_dir)
from tools.model_converters.publish_model import process_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config list file path')
    parser.add_argument('gpus', nargs='*', help='gpus id')
    parser.add_argument('--single', action='store_true', default=False, help='train config list file path')
    parser.add_argument('--multipy', action='store_true', default=False, help='train config list file path')
    parser.add_argument('--outfile', default=None)
    args = parser.parse_args()

    if args.multipy == args.single:
        raise ValueError
    if args.single and (len(args.gpus) > 1 or len(args.gpus) == 0):
        raise ValueError
    return args


def logEnd():
    logging.info(f'End time {datetime.now().strftime("%Y%m%d_%H%M%S")}')


class HandlerStopExp(Exception):
    def __init__(self, *args, **kwargs):
        self.message = f'Handle stop process.'
    # value = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default


def handler_signal(signal, frame):
    print_color_str('Handle stop process.\n', 'r')
    raise HandlerStopExp


def main():
    signal.signal(signal.SIGINT, handler_signal)
    args = parse_args()
    path = args.config
    if args.single:
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.gpus[0]}"
    else:
        pynvml.nvmlInit()
        n_gpus = pynvml.nvmlDeviceGetCount()
    outfile = args.outfile
    time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    date_str = time_str.split('_')[0]
    if outfile is None:
        task_file = os.path.splitext(os.path.basename(path))[0]
        outdir = os.path.join(os.getcwd(), 'log', date_str)
        os.makedirs(outdir, exist_ok=True)
        outfile = os.path.join(outdir, f'{task_file}_{time_str}')
    logging.basicConfig(filename=f'{outfile}.log', level=logging.INFO,
                        format='%(asctime)s %(message)s', datefmt='%Y%m%d-%H%M%S')
    logging.info(f'Begin time {time_str}')
    with open(path, 'r') as f:
        file_list = [file.strip() for file in f.readlines() if len(file.strip()) > 0]
    cmds = []
    save_dirs = []
    valid_files = []
    for file in file_list:
        if file.startswith('#'):
            print_color_str(f'Skip: {file[1:].strip()}', 'b')
            logging.info(f'Skip: {file[1:].strip()}')
            continue
        try:
            split_file = [line.strip() for line in file.split(' ') if len(line.strip()) > 0]
            if len(split_file) == 1:
                file = split_file[0]
                options = ''
            elif len(split_file) >= 2:
                file = split_file[0]
                options = ' '.join(split_file[1:])
            else:
                raise ValueError
            valid_files.append(file)
            cfg = Config.fromfile(file)
            if args.single:
                cmds.append(f'python3 ./tools/train.py {file} {options}')
            else:
                cmds.append(f'./tools/dist_train.sh {file} {n_gpus} {options}')
            save_dirs.append(cfg.work_dir)
        except HandlerStopExp as hse:
            logging.critical(f'{hse}')
            logEnd()
            sys.exit(0)
        except Exception as e:
            logging.error(f'There are something wrong when parse {file}.\n\t{e}')
            print_color_str(f'There are something wrong when parse {file}.\n\t{e}', 'm')
    logging.info(f'Valid config file: {len(cmds)}')
    for cmd in cmds:
        logging.info(f'\t{cmd}')

    for cmd, valid_file, save_dir in zip(cmds, valid_files, save_dirs):
        try:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            print_color_str(cmd, 'c')
            logging.info(f'Training: {valid_file}')
            os.system(cmd)
            logging.info(f'Done: {valid_file}')
            logging.info(f'{cmd}\tSucceed')
            result_dir = get_latest_dir(save_dir)
            file_name = os.path.splitext(os.path.basename(valid_file))[0]
            with open(os.path.join(save_dir, "last_checkpoint"), 'r') as f:
                pth_path = [line.strip() for line in f.readlines() if len(line.strip()) > 0][-1]
            shutil.copy(pth_path, result_dir)
            print_color_str(f'Copy {pth_path}\tTo\t{result_dir}', 'g')
            logging.info(f'Copy {pth_path}\tTo\t{result_dir}')
            test_save_dir = os.path.join(result_dir, file_name)
            pubulish_checkpoint = process_checkpoint(pth_path, f'{test_save_dir}.pth')
            print_color_str(f'Publish {pth_path}\tTo\t{pubulish_checkpoint}', 'g')
            logging.info(f'Publish {pth_path}\tTo\t{pubulish_checkpoint}')
            shutil.copy(os.path.join(save_dir, f'{file_name}.py'), result_dir)

            test_outfile_prefix = os.path.join(result_dir, file_name)
            if args.single:
                test_cmd = f'python3 ./tools/test.py {valid_file} {pubulish_checkpoint} ' \
                           f'--cfg-options test_evaluator.outfile_prefix={test_outfile_prefix}'
            else:
                test_cmd = f'./tools/dist_test.sh {valid_file} {pubulish_checkpoint} {n_gpus} ' \
                           f'--cfg-options test_evaluator.outfile_prefix={test_outfile_prefix}'
            print_color_str(test_cmd, 'g')
            logging.info(test_cmd)
            os.system(test_cmd)
            test_dir = get_latest_dir(save_dir)
            if test_dir is not None:
                shutil.rmtree(test_dir)
                print_color_str(f'Remove cache: {test_dir}', 'm')
                logging.info(f'Remove cache: {test_dir}')
            shutil.move(os.path.join(result_dir, file_name, f'{file_name}.zip'), result_dir)
            print_color_str(f"Move {os.path.join(result_dir, file_name, f'{file_name}.zip')} "
                            f"To {result_dir}", 'g')
            logging.info(f"Move {os.path.join(result_dir, file_name, f'{file_name}.zip')} To {result_dir}")
            shutil.rmtree(os.path.join(result_dir, file_name))
            print_color_str(f'Remove cache: {os.path.join(result_dir, file_name)}', 'm')
            logging.info(f'Remove cache: {os.path.join(result_dir, file_name)}')
            print_color_str(f'Test succeed! {file_name}')
            logging.info(f'Test succeed! {file_name}')

        except HandlerStopExp as hse:
            logging.critical(f'{hse}')
            logEnd()
            sys.exit(0)
        except Exception as e:
            logging.error(f'{cmd}\tFailed\n\tDetail: {e}')
    logEnd()
    # logging.info(f'End time {datetime.now().strftime("%Y%m%d_%H%M%S")}')

if __name__ == '__main__':
    main()