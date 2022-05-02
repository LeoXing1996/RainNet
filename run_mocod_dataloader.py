import os
from collections import defaultdict
from itertools import product
import json

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
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
    return net


def save_img(path, img):
    os.makedirs(os.path.split(path)[0], exist_ok=True)
    Image.fromarray(img).save(path)


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

    try:
        crop_h_min, crop_w_min, crop_h_max, crop_w_max = meta['crop_region']

        crop_h = crop_h_max - crop_h_min
        crop_w = crop_w_max - crop_w_min

        assert comp.ndim in [3, 4]
        if comp.ndim == 3:
            comp = comp[None, ...]
        comp_resize = F.interpolate(comp,
                                    size=(crop_h, crop_w),
                                    mode='bicubic')[0]

        img_old = meta['real_ori']
        img_old[:, crop_h_min:crop_h_max, crop_w_min:crop_w_max] = comp_resize
    except Exception:
        import ipdb
        ipdb.set_trace()

    return img_old


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    rainnet = load_network(test_cfg)

    rainnet.cuda()

    root = 'dataset/MOCOD_{}'
    test_data_dict = make_dataset(root)
    repeat_times = 1  # adjust the foreground image by several times
    batch_size = 4
    save_full_res = False

    num_kind = len(list(test_data_dict.keys()))
    for idx, kind in enumerate(test_data_dict):
        test_dataset = test_data_dict[kind]

        loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4)
        kind_root = os.path.join('results', kind)
        os.makedirs(kind_root, exist_ok=True)

        counter = 0
        desc = f'{idx+1:>3d} / {num_kind} Kind: {kind}'
        for data_sample in tqdm(loader, desc=desc):

            comp = data_sample['comp'].to(device)
            mask = data_sample['mask'].to(device)
            bg = data_sample['real'].to(device)

            pred = rainnet.processImage(comp, mask, bg)

            # repeat for N times
            for _ in range(repeat_times):
                pred = rainnet.processImage(comp, mask, bg)

            # save and recover images
            meta = data_sample['meta']
            for idx in range(batch_size):

                pred_rgb = pred[idx]
                meta_idx = {k: v[idx] for k, v in meta.items()}

                pred_recover = recover(pred_rgb, meta_idx)
                pred_recover = util.tensor2im(pred_recover[None, ...])

                bg_ori = util.tensor2im(meta_idx['real_ori'][None, ...])
                comp_ori = util.tensor2im(meta_idx['comp_paste'][None, ...])

                all_path = os.path.join(kind_root, f'{counter}_comp.png')

                single_path = os.path.join(kind_root, f'{counter}.png')
                paste_path = os.path.join(kind_root, f'{counter}_paste.png')

                save_img(all_path, np.hstack([comp_ori, bg_ori, pred_recover]))

                save_img(single_path, util.tensor2im(pred_rgb[None, ...]))
                save_img(paste_path, util.tensor2im(comp[idx][None, ...]))

                file_meta = dict(img_path=meta_idx['img_path'],
                                 bg_path=meta_idx['bg_path'],
                                 bbox=meta_idx['crop_region'].tolist())

                with open(os.path.join(kind_root, f'{counter}_meta.json'),
                          'w') as file:
                    json.dump(file_meta, file)

                if save_full_res:
                    fullres = os.path.join(kind_root, f'{counter}_fullRes.png')
                    fullres_paste_path = os.path.join(
                        kind_root, f'{counter}_fullRes_paste.png')
                    save_img(fullres, pred_recover)
                    save_img(fullres_paste_path, comp_ori)

                counter += 1


if __name__ == "__main__":
    main()
