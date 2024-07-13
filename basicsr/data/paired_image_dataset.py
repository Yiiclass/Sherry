from torch.utils import data as data
from torchvision.transforms.functional import normalize

from basicsr.data.data_util import (paired_paths_from_folder,
                                    paired_DP_paths_from_folder,
                                    paired_paths_from_lmdb,
                                    paired_paths_from_meta_info_file)
from basicsr.data.transforms import augment, paired_random_crop, paired_random_crop_DP, random_augmentation
from basicsr.utils import FileClient, imfrombytes, img2tensor, padding, padding_DP, imfrombytesDP

import random
import numpy as np
import torch
import cv2
from pdb import set_trace as stx

### mask2former
class Dataset_PairedImage(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and
    GT image pairs.

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If opt['io_backend'] == lmdb.
    2. 'meta_info_file': Use meta information file to generate paths.
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. 'folder': Scan folders to generate paths.
        The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            dataroot_seg (str): Data root path for seg.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            filename_tmpl (str): Template for each filename. Note that the
                template excludes the file extension. Default: '{}'.
            gt_size (int): Cropped patched size for gt patches.
            geometric_augs (bool): Use geometric augmentations.

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(Dataset_PairedImage, self).__init__()
        self.opt = opt
        # file client (io backend) 文件客户端
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        
        self.gt_folder, self.lq_folder, self.seg_folder = opt['dataroot_gt'], opt['dataroot_lq'], opt['dataroot_seg']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder, self.seg_folder]
            self.io_backend_opt['client_keys'] = ['lq', 'gt', 'seg']
            self.paths = paired_paths_from_lmdb(
                [self.lq_folder, self.gt_folder, self.seg_folder], ['lq', 'gt', 'seg'])
        elif 'meta_info_file' in self.opt and self.opt[
                'meta_info_file'] is not None:
            self.paths = paired_paths_from_meta_info_file(
                [self.lq_folder, self.gt_folder, self.seg_folder], ['lq', 'gt', 'seg'],
                self.opt['meta_info_file'], self.filename_tmpl)
        else:
            self.paths = paired_paths_from_folder(
                [self.lq_folder, self.gt_folder, self.seg_folder], ['lq', 'gt', 'seg'],
                self.filename_tmpl)

        if self.opt['phase'] == 'train':
            self.geometric_augs = opt['geometric_augs']

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']
        index = index % len(self.paths)
        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')
        try:
            img_gt = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("gt path {} not working".format(gt_path))

        lq_path = self.paths[index]['lq_path']
        img_bytes = self.file_client.get(lq_path, 'lq')
        try:
            img_lq = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("lq path {} not working".format(lq_path))
        
        seg_path = self.paths[index]['seg_path']
        np_seg = np.load(seg_path)
        np_seg = np.expand_dims(np_seg, axis=-1)
        np_seg = np_seg.astype(np.float32) / 255

        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # padding
            img_gt, img_lq, np_seg = padding(img_gt, img_lq, np_seg, gt_size)

            # random crop
            img_gt, img_lq, np_seg = paired_random_crop(img_gt, img_lq, np_seg, gt_size, scale,
                                                gt_path, seg_path)

            # flip, rotation augmentations
            if self.geometric_augs:
                img_gt, img_lq, np_seg = random_augmentation(img_gt, img_lq, np_seg)
            
        # BGR to RGB, HWC to CHW, numpy to tensor
        ### change
        ### ------------------------------------------------------------------------------
        # img_gt, img_lq, img_seg = img2tensor([img_gt, img_lq, img_seg],
        #                             bgr2rgb=True,
        #                             float32=True)
        ### ------------------------------------------------------------------------------
        img_gt, img_lq = img2tensor([img_gt, img_lq],
                                    colorspace=self.opt['colorspace'],
                                    float32=True)
        
        
        tensor_seg = torch.from_numpy(np_seg.transpose(2, 0, 1))
        ### ------------------------------------------------------------------------------
                
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)
        
        return {
            'lq': img_lq,
            'gt': img_gt,
            'seg': tensor_seg,
            'lq_path': lq_path,
            'gt_path': gt_path,
            'seg_path': seg_path
        }

    def __len__(self):
        return len(self.paths)
