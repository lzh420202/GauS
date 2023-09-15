from mmengine.dataset import DefaultSampler
from typing import Iterator, Optional, Sized
from mmrotate.registry import DATA_SAMPLERS
import math
import torch


@DATA_SAMPLERS.register_module()
class SupplementSampler(DefaultSampler):
    def __init__(self, mode, rate,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.mode = mode
        self.rate = rate
        self.valid_num = self.dataset.valid_num
        self.pure_background_num = self.dataset.pure_background_num
        if mode.lower() in ['all', 'total']:
            self.pure_num = round(self.pure_background_num * rate)
        elif mode.lower() in ['valid', ]:
            self.pure_num = round(self.valid_num * rate)
        else:
            raise NotImplementedError
        # self.pure_num = min(self.pure_num, self.pure_background_num)
        self.all_num = self.valid_num + self.pure_num

        if self.round_up:
            self.num_samples = math.ceil(self.all_num / self.world_size)
            self.total_size = self.num_samples * self.world_size
        else:
            self.num_samples = math.ceil((self.all_num - self.rank) / self.world_size)
            self.total_size = self.all_num

    def __iter__(self):
        """Iterate the indices."""
        # deterministically shuffle based on epoch and seed
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            pure_seed = torch.Generator()
            pure_seed.manual_seed(self.seed + self.epoch * 2)
            random_valid = torch.tensor(self.dataset.valid_data_idx)[torch.randperm(self.valid_num, generator=pure_seed)]
            random_valid = random_valid[:self.pure_num]
            if self.pure_num > self.pure_background_num:
                repeat_n = math.ceil(self.pure_num / self.pure_background_num)
                pure_idx = torch.tensor(self.dataset.pure_data_idx).repeat(repeat_n)
            else:
                pure_idx = torch.tensor(self.dataset.pure_data_idx)
            random_pure = pure_idx[torch.randperm(self.pure_num, generator=pure_seed)]
            random_all = torch.cat([torch.tensor(self.dataset.valid_data_idx), random_pure])
            aux_all = torch.cat([torch.full_like(torch.tensor(self.dataset.valid_data_idx), -1), random_valid])
            indices = torch.randperm(self.all_num, generator=g)
            random_all = random_all[indices].tolist()
            random_aux_all = aux_all[indices].tolist()
        else:
            raise NotImplementedError
            # indices = torch.arange(len(self.dataset)).tolist()

        # add extra samples to make it evenly divisible
        if self.round_up:
            random_all = (
                random_all *
                int(self.total_size / len(random_all) + 1))[:self.total_size]
            random_aux_all = (
                random_aux_all *
                int(self.total_size / len(random_aux_all) + 1))[:self.total_size]
            # indices = (
            #     indices *
            #     int(self.total_size / len(indices) + 1))[:self.total_size]

        # subsample
        # indices = indices[self.rank:self.total_size:self.world_size]
        random_all = random_all[self.rank:self.total_size:self.world_size]
        random_aux_all = random_aux_all[self.rank:self.total_size:self.world_size]
        return zip(random_all, random_aux_all)

        # return iter(indices)