import os
from collections import defaultdict
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from imageio import mimsave
from PIL import Image
from skimage import io
from tqdm import tqdm

from data.mocod_dataset import TestDataset
from run_mocod import recover
from models.networks import RainNet
from models.normalize import RAIN
from util import util
from util.config import cfg as test_cfg


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


def save_img(img, name):
    img = ((img.permute(1, 2, 0) + 1) / 2 * 255.).numpy().astype(np.uint8)
    if img.shape[-1] == 1:
        img = img[..., 0]
    Image.fromarray(img).save(name)


def crop_bbox(data_sample):

    bg = data_sample['real']
    comp = data_sample['comp']
    mask = data_sample['mask']
    comp_paste = data_sample['meta']['comp_paste']

    save_img(bg, 'bg.png')
    save_img(comp, 'comp.png')
    save_img(mask, 'mask.png')
    save_img(comp_paste, 'comp_paste.png')

    comp_recover = recover(comp, data_sample['meta'])
    save_img(comp_recover, 'recover.png')

    print(data_sample.keys())
    print(data_sample['meta'].keys())


def main():
    root = 'dataset/MOCOD_{}'
    test_data_dict = make_dataset(root)

    test_data = test_data_dict[list(test_data_dict.keys())[0]]
    data_sample = test_data[0]
    crop_bbox(data_sample)


if __name__ == '__main__':
    main()
