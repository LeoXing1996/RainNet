from copy import deepcopy
import os.path
from glob import glob
from PIL import Image
import numpy as np

import random
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class TestDataset(Dataset):
    def __init__(self,
                 foreground_paths,
                 mask_paths,
                 background_paths=None,
                 load_size=256):
        '''
        foreground_paths: [folder, imagepath_list, image path]
        mask_paths: [folter, imagepath_list, image path]
        background_paths: [folter, imagepath_list, image path]
        '''
        self.foreg_paths, self.mask_paths, self.backg_paths = \
            foreground_paths, mask_paths, background_paths

        self._load_images_paths()
        self._load_sizez = load_size
        self.transform_image = transforms.Compose([
            # transforms.RandomCrop([load_size, load_size]),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
        self.transform_mask = transforms.Compose([
            # transforms.Resize([load_size, load_size]),
            transforms.ToTensor()
        ])

    def _load_images_paths(self, ):
        if isinstance(self.foreg_paths, list):
            pass
        elif os.path.isdir(self.foreg_paths):
            self.foreg_paths = glob(self.foreg_paths + '/*.jpg') + \
                               glob(self.foreg_paths + '/*.png') + \
                               glob(self.foreg_paths + '/*.bmp')
            self.mask_paths = glob(self.mask_paths + '/*.png')
            if self.backg_paths is not None:
                self.backg_paths = glob(self.mask_paths + '/*.png') + \
                                   glob(self.backg_paths + '/*.jpg') + \
                                   glob(self.backg_paths + '/*.bmp')
        elif os.path.isfile(self.foreg_paths) and (self.foreg_paths[-3:]
                                                   in ['jpg', 'png', 'bmp']):
            self.foreg_paths = [self.foreg_paths]
            self.mask_paths = [self.mask_paths]
            if self.backg_paths is not None:
                self.backg_paths = [self.backg_paths]
        else:
            print(f'please check the test path: {self.foreg_paths} '
                  f'{self.mask_paths} {self.backg_paths}')
            exit()
        self._sort()
        print(f'total {len(self.foreg_paths)} images')

    def _sort(self):
        self.foreg_paths.sort()
        self.mask_paths.sort()
        if self.backg_paths is not None:
            self.backg_paths.sort()

    def get_mask_center(self, mask):
        """This function calculate the center of the mask and crop a 256x256
        region

        Returns:
            dict: a dict of bbox and cent
        """
        # mask: [1, h, w]
        _, h, w = torch.nonzero(mask == 1, as_tuple=True)
        h_min, h_max = h.min(), h.max()
        w_min, w_max = w.min(), w.max()

        h_cent = (h_min + h_max) / 2
        w_cent = (w_min + w_max) / 2
        bbox = torch.LongTensor(
            [int(coord) for coord in [h_min, w_min, h_max, w_max]])
        cent = torch.LongTensor([int(h_cent), int(w_cent)])
        return dict(bbox=bbox, cent=cent)

    def crop_and_padding(self, sample):
        """In this function, we crop the mask region and resize that
        region to [load_size, load_size].
        And add a key called 'meta'.
        """

        assert 'mask' in sample and 'comp' in sample

        # ignore `crop_and_padding`
        if 'cent' not in sample:
            return sample
        # cent = sample['cent']
        bbox = sample['bbox']
        # h_cent, w_cent = cent

        bbox_h_min, bbox_w_min, bbox_h_max, bbox_w_max = bbox
        # h_min, h_max = h_cent - half_size, h_cent + half_size
        # w_min, w_max = w_cent - half_size, w_cent + half_size

        # get element
        mask = sample['mask']
        comp = sample['comp']
        h, w = mask.shape[1:]

        # half_size = self._load_sizez // 2

        # add random padding
        h_padding = random.uniform(15, 30)
        w_padding = random.uniform(15, 30)
        crop_h_min = int(max(0, bbox_h_min - h_padding))
        crop_h_max = int(min(h, bbox_h_max + h_padding))
        crop_w_min = int(max(0, bbox_w_min - w_padding))
        crop_w_max = int(min(w, bbox_w_max + w_padding))

        # crop
        mask_crop = mask[:, crop_h_min:crop_h_max, crop_w_min:crop_w_max]
        comp_crop = comp[:, crop_h_min:crop_h_max, crop_w_min:crop_w_max]

        # ipdb.set_trace()
        # resize
        mask_resize = F.interpolate(mask_crop[None, ...],
                                    size=(self._load_sizez, self._load_sizez),
                                    mode='bicubic')[0]
        comp_resize = F.interpolate(comp_crop[None, ...],
                                    size=(self._load_sizez, self._load_sizez),
                                    mode='bicubic')[0]

        sample['mask'] = mask_resize
        sample['comp'] = comp_resize

        # add meta
        meta = dict(crop_region=torch.LongTensor(
            [crop_h_min, crop_w_min, crop_h_max, crop_w_max]),
                    mask_ori=deepcopy(mask),
                    comp_ori=deepcopy(comp))

        # print(meta['crop_region'])

        # handle real
        if 'real' in sample:
            real = sample['real']
            real_crop = real[:, crop_h_min:crop_h_max, crop_w_min:crop_w_max]
            real_resize = F.interpolate(real_crop[None, ...],
                                        size=(self._load_sizez,
                                              self._load_sizez),
                                        mode='bicubic')[0]
            sample['real'] = real_resize
            meta['real_ori'] = deepcopy(real)

            # apply raw comp and replace sample and meta
            # ipdb.set_trace()
            comp_raw = comp_resize * mask_resize + real_resize * (1 -
                                                                  mask_resize)
            # meta['comp_resize'] = deepcopy(comp_resize)
            sample['comp'] = comp_raw

            # add comp raw
            comp_paste = deepcopy(real)
            comp_raw = comp_crop * mask_crop + real_crop * (1 - mask_crop)
            comp_paste[:, crop_h_min:crop_h_max,
                       crop_w_min:crop_w_max] = comp_raw
            meta['comp_paste'] = comp_paste

        sample['meta'] = meta
        return sample

    def __getitem__(self, index):
        comp = self.transform_image(
            Image.open(self.foreg_paths[index]).convert('RGB'))
        mask = self.transform_mask(
            Image.open(self.mask_paths[index]).convert('1'))

        if self.backg_paths is not None:
            real = self.transform_image(
                Image.open(self.backg_paths[index]).convert('RGB'))
            # comp = self._compose(comp, mask, real)
        else:
            real = self.transform_image(
                Image.open(self.foreg_paths[index]).convert('RGB'))

        bbox_cent = self.get_mask_center(mask)

        sample = {'comp': comp, 'mask': mask, 'real': real, **bbox_cent}

        # we calculate simple comp in this function
        sample = self.crop_and_padding(sample)

        # add info
        sample['meta'].update({
            'img_path': self.foreg_paths[index],
            'bg_path': self.backg_paths[index],
        })
        return sample

    def __len__(self):
        """Return the total number of images."""
        return len(self.foreg_paths)

    def _compose(self, foreground_img, foreground_mask, background_img):
        return foreground_img * foreground_mask + background_img * (
            1 - foreground_mask)

    def _compose_raw(self, foreground_img, foreground_mask, background_img):
        """This function apply paste at center
        """
        return
