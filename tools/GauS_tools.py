import os
from datetime import datetime
from .tool import print_color_str
from mmengine.config import Config


def get_latest_dir(target, empty=True, size=100):
    dirs = [file for file in os.listdir(target) if os.path.isdir(os.path.join(target, file))]
    current = datetime.now()
    times = []
    valid_dris = []
    for d in dirs:
        try:
            t = (current - datetime.strptime(d, '%Y%m%d_%H%M%S')).total_seconds()
            if empty:
                dir_ = os.path.join(target, d)
                file_list = [file for file in os.listdir(dir_) if os.path.isfile(os.path.join(dir_, file)) and os.path.getsize(os.path.join(dir_, file)) > size * (2 << 19)]
                if len(file_list) > 0:
                    print_color_str(f'Skip: {dir_}', 'm')
                    for file in file_list:
                        path = os.path.join(dir_, file)
                        print_color_str(f'\t{file}, size: {os.path.getsize(path) / (2 << 19):.2f}M, Limit: {size}M', 'm')
                    continue
            times.append(t)
            valid_dris.append(d)
        except:
            print_color_str(f'Skip dir: {d}.', 'm')
            continue
    assert len(times) == len(valid_dris)
    if len(valid_dris) > 0:
        new_ = list(zip(valid_dris, times))
        new_.sort(key=lambda pair: pair[1])
        return os.path.join(target, new_[0][0])
    else:
        return None


def get_synth_name(synth_cfg):
    assert isinstance(synth_cfg, dict) or isinstance(synth_cfg, Config)
    synth_str = ''
    # synth_method: 1 -> directly weight boxes or NMW (alpha=1 and beta=1); 2 -> gaussian weight boxes.
    gauss_valid_keys = ['synth_thr', 'synth_method', 'alpha', 'beta']
    check_gauss = all([key in synth_cfg for key in gauss_valid_keys])
    # method: 1 -> WBF, 2 -> soft-NMS, 3-> DIoU-NMS
    others_valid_keys = ['method', 'iou_thr']
    check_others = all([key in synth_cfg for key in others_valid_keys])
    if check_gauss + check_others == 0:
        raise AttributeError('Incomplete key "' + '" ,"'.join(gauss_valid_keys + others_valid_keys) + '"' +
                             '\nCurrent key: "' + '" ,"'.join(synth_cfg) + '"')
    elif check_gauss + check_others == 2:
        raise AttributeError('Cannot use 2 types of parameters at the same time.')
    elif check_gauss + check_others == 1:
        print_color_str('Use cfg key: "' + '" ,"'.join(synth_cfg) + '"', 'c')
    if check_gauss:
        new_synth_cfg = synth_cfg.copy()
        method = new_synth_cfg.get('synth_method')
        alpha = new_synth_cfg.get('alpha')
        beta = new_synth_cfg.get('beta')
        para_str = f'\talpha = {alpha}, \tbeta = {beta}'
        print_str = f'{new_synth_cfg.synth_thr}_{new_synth_cfg.alpha}_{new_synth_cfg.beta}'
        if method == 1:
            if (alpha == 1) and (beta == 1):
                print_color_str(f'Synth method: NMW, {para_str}', 'g')
                synth_str += f'NMW_{print_str}'
            else:
                print_color_str(f'Synth method: Direct, {para_str}', 'g')
                synth_str += f'Direct_{print_str}'
        elif method == 2:
            print_color_str(f'Synth method: GauS, {para_str}', 'g')
            synth_str += f'GauS_{print_str}'
        else:
            raise NotImplementedError
    elif check_others:
        new_cfg = synth_cfg.copy()
        method = new_cfg.get('method')
        iou_thr = new_cfg.get('iou_thr')
        para_str = f'\tIoU threshold = {iou_thr}'
        print_str = f'{new_cfg.iou_thr}'
        if method == 1: # WBF
            print_color_str(f'Synth method: WBF, {para_str}', 'g')
            synth_str += f'WBF_{print_str}'
        elif method == 2: # soft-NMS add sigma
            sigma = new_cfg.get('sigma')
            para_str += f',\tsigma = {sigma}'
            synth_str += f'Soft_NMS_{print_str}_{sigma}'
            print_color_str(f'Synth method: Soft-NMS, {para_str}', 'g')
        elif method == 3: # DIoU-NMS
            synth_str += f'DIoU_NMS_{print_str}'
            print_color_str(f'Synth method: DIoU-NMS, {para_str}', 'g')
        else:
            raise NotImplementedError('Unsupport post-processing method.')
    else:
        raise NotImplementedError('Unsupport config file.')
    return synth_str


def get_file_sha256_name(name):
    return os.path.splitext(name)[0][-8:]