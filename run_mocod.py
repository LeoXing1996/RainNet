import os
from collections import defaultdict
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from imageio import mimsave
from skimage import io
from tqdm import tqdm

# from data.test_dataset import TestDataset
from data.mocod_dataset import TestDataset
from models.networks import RainNet
from models.normalize import RAIN
from util import util
from util.config import cfg as test_cfg


def load_network(cfg):
    net = RainNet(input_nc=cfg.input_nc,
                  output_nc=cfg.output_nc,
                  ngf=cfg.ngf,
                  norm_layer=RAIN,
                  use_dropout=not cfg.no_dropout)

    load_path = os.path.join(cfg.checkpoints_dir, cfg.name, 'net_G.pth')
    if not os.path.exists(load_path):
        raise FileExistsError(
            print('%s not exists. Please check the file' % (load_path)))
    print(f'loading the model from {load_path}')
    state_dict = torch.load(load_path)
    util.copy_state_dict(net.state_dict(), state_dict)
    # net.load_state_dict(state_dict)
    return net


def save_img(path, img):
    os.makedirs(os.path.split(path)[0], exist_ok=True)
    io.imsave(path, img)


def make_dataset(root):
    fg_root = root.format('objs')
    bg_root = root.format('bgs')

    obj_kind = os.listdir(fg_root)

    mask_fname, obj_fname = defaultdict(list), defaultdict(list)
    for kind in obj_kind:
        kind_root = os.path.join(fg_root, kind)
        for file in os.listdir(kind_root):
            if 'mask' in file and file.endswith('png'):
                mask_fname[kind].append(os.path.join(kind_root, file))
            if 'fg' in file and file.endswith('png'):
                obj_fname[kind].append(os.path.join(kind_root, file))

    bg_fname = []
    for root_, dir_, files in os.walk(bg_root):
        for file in files:
            if file and file.endswith('jpg'):
                bg_fname.append(os.path.join(root_, file))

    # build product of fg and bg
    bg_fname = [[fname] for fname in bg_fname]
    kind_pair = dict()
    for kind in obj_kind:
        fg_fname = [[mask, obj]
                    for mask, obj in zip(mask_fname[kind], obj_fname[kind])]
        # [([mask, obj], [bg]), ...]
        pair = list(product(fg_fname, bg_fname))

        kind_pair[kind] = dict(fg=[p[0][1] for p in pair],
                               mask=[p[0][0] for p in pair],
                               bg=[p[1][0] for p in pair])

    test_dataset_dict = dict()
    for kind in obj_kind:
        test_dataset = TestDataset(foreground_paths=kind_pair[kind]['fg'],
                                   mask_paths=kind_pair[kind]['mask'],
                                   background_paths=kind_pair[kind]['bg'])
        test_dataset_dict[kind] = test_dataset

    return test_dataset_dict


def recover(comp, meta):
    """This function provide a interface to recover the crop-resized image.
    NOTE: the inputs are shape as [ch, H, W]
    """
    # seems we only need 'copy region' and t
    assert ('crop_region' in meta) and ('mask_ori' in meta)

    crop_h_min, crop_w_min, crop_h_max, crop_w_max = meta['crop_region']

    crop_h = crop_h_max - crop_h_min
    crop_w = crop_w_max - crop_w_min

    assert comp.ndim in [3, 4]
    if comp.ndim == 3:
        comp = comp[None, ...]
    comp_resize = F.interpolate(comp, size=(crop_h, crop_w), mode='bicubic')[0]

    img_old = meta['real_ori']
    img_old[:, crop_h_min:crop_h_max, crop_w_min:crop_w_max] = comp_resize

    return img_old


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    rainnet = load_network(test_cfg)

    rainnet.cuda()

    root = 'dataset/MOCOD_{}'
    test_data_dict = make_dataset(root)
    repeat_times = 1  # adjust the foreground image by several times

    num_kind = len(list(test_data_dict.keys()))
    for idx, kind in enumerate(test_data_dict):
        test_dataset = test_data_dict[kind]

        kind_root = os.path.join('results', kind)
        os.makedirs(kind_root, exist_ok=True)

        for idx in tqdm(range(len(test_dataset)),
                        desc=f'{idx:>3d} / {num_kind} Kind: {kind}'):
            sample = test_dataset[idx]

            comp = sample['comp'].unsqueeze(0).to(device)
            mask = sample['mask'].unsqueeze(0).to(device)
            # if you want to adjust the background to be compatible with the
            # foreground, then add the following command
            # mask = 1 - mask
            bg = sample['real'].unsqueeze(0).to(device)

            pred = rainnet.processImage(comp, mask, bg)

            # repeat for N times
            for _ in range(repeat_times):
                pred = rainnet.processImage(comp, mask, bg)

            # tensor2image
            # pred_rgb = util.tensor2im(pred[0:1])
            pred_rgb = pred[0:1]
            # comp_rgb = util.tensor2im(comp[:1])
            # mask_rgb = util.tensor2im(mask[:1])
            # bg_rgb = util.tensor2im(bg[:1])

            # recover
            meta = sample['meta']
            pred_recover = recover(pred_rgb, meta)
            pred_recover = util.tensor2im(pred_recover[None, ...])
            # mask_ori = util.tensor2im(meta['mask_ori'][None, ...])
            bg_ori = util.tensor2im(meta['real_ori'][None, ...])
            comp_ori = util.tensor2im(meta['comp_paste'][None, ...])

            all_path = os.path.join(kind_root, f'{idx}_comp.png')

            single_path = os.path.join(kind_root, f'{idx}.png')
            paste_path = os.path.join(kind_root, f'{idx}_paste.png')

            save_img(all_path, np.hstack([comp_ori, bg_ori, pred_recover]))

            save_img(single_path, util.tensor2im(pred[0:1]))
            save_img(paste_path, util.tensor2im(comp))

            fullres = os.path.join(kind_root, f'{idx}_fullRes.png')
            fullres_paste_path = os.path.join(kind_root,
                                              f'{idx}_fullRes_paste.png')
            save_img(fullres, pred_recover)
            save_img(fullres_paste_path, comp_ori)


if __name__ == "__main__":
    main()
