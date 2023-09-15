# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import torch
import torch.nn as nn
import time
from typing import List

from mmdet.utils.benchmark import (DataLoaderBenchmark, DatasetBenchmark,
                                   InferenceBenchmark, print_process_memory,
                                   print_log)
from mmengine import MMLogger
from mmengine.config import Config, DictAction
from mmengine.dist import init_dist
from mmengine.utils import mkdir_or_exist
from mmengine.device import get_max_cuda_memory
from mmengine.runner import load_checkpoint
from mmdet.registry import MODELS
from torch.nn.parallel import DistributedDataParallel
from mmcv.cnn import fuse_conv_bn

from mmrotate.utils import register_all_modules


class RotatedBenchmark(InferenceBenchmark):
    def __init__(self,
                 mode=None,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.mode = 'tensor' if mode is None else mode

    def _init_model(self, checkpoint: str, is_fuse_conv_bn: bool) -> nn.Module:
        """Initialize the model."""
        model = MODELS.build(self.cfg.model)
        # TODO need update
        # fp16_cfg = self.cfg.get('fp16', None)
        # if fp16_cfg is not None:
        #     wrap_fp16_model(model)

        load_checkpoint(model, checkpoint, map_location='cpu')
        if is_fuse_conv_bn:
            model = fuse_conv_bn(model)

        model = model.cuda()
        self.data_preprocessor = model.data_preprocessor
        if self.distributed:
            model = DistributedDataParallel(
                model,
                device_ids=[torch.cuda.current_device()],
                broadcast_buffers=False,
                find_unused_parameters=False)

        model.eval()
        return model

    def run_once(self) -> dict:
        """Executes the benchmark once."""
        pure_inf_time = 0
        fps = 0

        for i, data in enumerate(self.data_loader):

            if (i + 1) % self.log_interval == 0:
                print_log('==================================', self.logger)

            torch.cuda.synchronize()
            start_time = time.perf_counter()

            with torch.no_grad():
                # self.model(data, return_loss=False)
                # data['inputs'] = torch.stack(data['inputs'], dim=0)
                # data = self.model.data_preprocessor(data, False)
                data = self.data_preprocessor(data, False)
                # data['mode'] = 'predict'
                data['mode'] = self.mode
                self.model(**data)

            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start_time

            if i >= self.num_warmup:
                pure_inf_time += elapsed
                if (i + 1) % self.log_interval == 0:
                    fps = (i + 1 - self.num_warmup) / pure_inf_time
                    cuda_memory = get_max_cuda_memory()

                    print_log(
                        f'Done image [{i + 1:<3}/{self.max_iter}], '
                        f'fps: {fps:.1f} img/s, '
                        f'times per image: {1000 / fps:.1f} ms/img, '
                        f'cuda memory: {cuda_memory} MB', self.logger)
                    print_process_memory(self._process, self.logger)

            if (i + 1) == self.max_iter:
                fps = (i + 1 - self.num_warmup) / pure_inf_time
                break

        return {'fps': fps}

    def average_multiple_runs(self, results: List[dict]) -> dict:
        """Average the results of multiple runs."""
        print_log('============== Done ==================', self.logger)

        # fps_list_ = [round(result['fps'], 2) for result in results]
        fps_list_ = [result['fps'] for result in results]
        avg_fps_ = sum(fps_list_) / len(fps_list_)
        outputs = {'avg_fps': avg_fps_, 'fps_list': fps_list_}

        if len(fps_list_) > 1:
            times_pre_image_list_ = [
                round(1000 / result['fps'], 2) for result in results
            ]
            avg_times_pre_image_ = sum(times_pre_image_list_) / len(
                times_pre_image_list_)

            print_log(
                f'Overall fps: {fps_list_}[{avg_fps_:.2f}] img/s, '
                'times per image: '
                f'{times_pre_image_list_}[{avg_times_pre_image_:.2f}] '
                'ms/img', self.logger)
        else:
            print_log(
                f'Overall fps: {fps_list_[0]:.2f} img/s, '
                f'times per image: {1000 / fps_list_[0]:.2f} ms/img',
                self.logger)

        print_log(f'cuda memory: {get_max_cuda_memory()} MB', self.logger)
        print_process_memory(self._process, self.logger)

        return outputs


def parse_args():
    parser = argparse.ArgumentParser(description='MMDet benchmark')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('--checkpoint', help='checkpoint file')
    parser.add_argument(
        '--task',
        choices=['inference', 'dataloader', 'dataset', 'rotated'],
        default='dataloader',
        help='Which task do you want to go to benchmark')
    parser.add_argument(
        '--repeat-num',
        type=int,
        default=1,
        help='number of repeat times of measurement for averaging the results')
    parser.add_argument(
        '--mode',
        choices=['tensor', 'predict'],
        default='tensor',
        help='Which task do you want to go to benchmark')
    parser.add_argument(
        '--out',
        type=str,
        default=None,
        help='out results')
    parser.add_argument(
        '--max-iter', type=int, default=2000, help='num of max iter')
    parser.add_argument(
        '--log-interval', type=int, default=50, help='interval of logging')
    parser.add_argument(
        '--num-warmup', type=int, default=5, help='Number of warmup')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--dataset-type',
        choices=['train', 'val', 'test'],
        default='test',
        help='Benchmark dataset type. only supports train, val and test')
    parser.add_argument(
        '--work-dir',
        help='the directory to save the file containing '
        'benchmark metrics')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def inference_benchmark(args, cfg, distributed, logger):
    benchmark = InferenceBenchmark(
        cfg,
        args.checkpoint,
        distributed,
        args.fuse_conv_bn,
        args.max_iter,
        args.log_interval,
        args.num_warmup,
        logger=logger)
    return benchmark


def rotated_benchmark(args, cfg, distributed, logger):
    benchmark = RotatedBenchmark(
        args.mode,
        cfg,
        args.checkpoint,
        distributed,
        args.fuse_conv_bn,
        args.max_iter,
        args.log_interval,
        args.num_warmup,
        logger=logger)
    return benchmark


def dataloader_benchmark(args, cfg, distributed, logger):
    benchmark = DataLoaderBenchmark(
        cfg,
        distributed,
        args.dataset_type,
        args.max_iter,
        args.log_interval,
        args.num_warmup,
        logger=logger)
    return benchmark


def dataset_benchmark(args, cfg, distributed, logger):
    benchmark = DatasetBenchmark(
        cfg,
        args.dataset_type,
        args.max_iter,
        args.log_interval,
        args.num_warmup,
        logger=logger)
    return benchmark


def main():
    register_all_modules()

    args = parse_args()
    cfg = Config.fromfile(args.config)

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    distributed = False
    if args.launcher != 'none':
        init_dist(args.launcher, **cfg.get('env_cfg', {}).get('dist_cfg', {}))
        distributed = True

    log_file = None
    if args.work_dir:
        log_file = os.path.join(args.work_dir, 'benchmark.log')
        mkdir_or_exist(args.work_dir)

    logger = MMLogger.get_instance(
        'mmdet', log_file=log_file, log_level='INFO')

    benchmark = eval(f'{args.task}_benchmark')(args, cfg, distributed, logger)
    result = benchmark.run(args.repeat_num)
    if args.out is not None:
        with open(args.out, 'w') as f:
            f.write(f"{result['avg_fps']}")


if __name__ == '__main__':
    main()
