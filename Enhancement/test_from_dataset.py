### Mask2former

from ast import arg
import numpy as np
import os
import argparse
from tqdm import tqdm
import cv2

import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import utils

from natsort import natsorted
from glob import glob
from skimage import img_as_ubyte
from pdb import set_trace as stx
from skimage import metrics

from basicsr.models import create_model
from basicsr.utils.options import dict2str, parse
from basicsr.utils import img2tensor, tensor2img


parser = argparse.ArgumentParser(
    description='Image Enhancement using Retinexformer')

parser.add_argument('--input_dir', default='./Enhancement/Datasets',
                    type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='./results/',
                    type=str, help='Directory for results')
parser.add_argument(
    '--opt', type=str, default='Options/RetinexFormer_SDSD_indoor.yml', help='Path to option YAML file.')
parser.add_argument('--weights', default='pretrained_weights/SDSD_indoor.pth',
                    type=str, help='Path to weights')
parser.add_argument('--dataset', default='SDSD_indoor', type=str,
                    help='Test Dataset') 
parser.add_argument('--gpu_id', type=str, default="0", help='GPU devices.')
parser.add_argument('--GT_mean', action='store_true', help='Use the mean of GT to rectify the output of the model')

args = parser.parse_args()

# 指定 gpu
gpu_list = ','.join(str(x) for x in args.gpu_id)
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
print('export CUDA_VISIBLE_DEVICES=' + gpu_list)

####### Load yaml #######
yaml_file = args.opt
weights = args.weights
print(f"dataset {args.dataset}")

import yaml

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

opt = parse(args.opt, is_train=False)
opt['dist'] = False


x = yaml.load(open(args.opt, mode='r'), Loader=Loader)
s = x['network_g'].pop('type')
##########################


model_restoration = create_model(opt).net_g

# 加载模型
checkpoint = torch.load(weights)

try:
    model_restoration.load_state_dict(checkpoint['params'])
except:
    new_checkpoint = {}
    for k in checkpoint['params']:
        new_checkpoint['module.' + k] = checkpoint['params'][k]
    model_restoration.load_state_dict(new_checkpoint)

print("===>Testing using weights: ", weights)
model_restoration.cuda()
model_restoration = nn.DataParallel(model_restoration)
model_restoration.eval()

# 生成输出结果的文件
factor = 4
dataset = args.dataset
config = os.path.basename(args.opt).split('.')[0]
checkpoint_name = os.path.basename(args.weights).split('.')[0]
result_dir = os.path.join(args.result_dir, dataset, config, checkpoint_name)
result_dir_input = os.path.join(args.result_dir, dataset, 'input')
result_dir_gt = os.path.join(args.result_dir, dataset, 'gt')
# stx()
os.makedirs(result_dir, exist_ok=True)

psnr = []
ssim = []
input_dir = opt['datasets']['val']['dataroot_lq']
target_dir = opt['datasets']['val']['dataroot_gt']
seg_dir = opt['datasets']['val']['dataroot_seg']
    
print(input_dir)
print(target_dir)
print(seg_dir)

input_paths = natsorted(
    glob(os.path.join(input_dir, '*.png')) + glob(os.path.join(input_dir, '*.bmp')))

target_paths = natsorted(glob(os.path.join(
    target_dir, '*.png')) + glob(os.path.join(target_dir, '*.bmp')))
    
seg_paths = natsorted(glob(os.path.join(
    seg_dir, '*.png')) + glob(os.path.join(seg_dir, '*.bmp')) + glob(os.path.join(seg_dir, '*.npy')))

with torch.inference_mode():
    for inp_path, tar_path, seg_path in tqdm(zip(input_paths, target_paths, seg_paths), total=len(target_paths)):

        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()

        img = np.float32(utils.load_img(inp_path)) / 255.
        target = np.float32(utils.load_img(tar_path)) / 255.
                
        np_seg = np.load(seg_path)
        np_seg = np.expand_dims(np_seg, axis=-1)
        seg = np_seg.astype(np.float32) / 255


        img = torch.from_numpy(img).permute(2, 0, 1)
        input_ = img.unsqueeze(0).cuda()

        seg = torch.from_numpy(seg).permute(2, 0, 1)
        seg_ = seg.unsqueeze(0).cuda()

                # Padding in case images are not multiples of 4
        h, w = input_.shape[2], input_.shape[3]
        H, W = ((h + factor) // factor) * \
            factor, ((w + factor) // factor) * factor
        padh = H - h if h % factor != 0 else 0
        padw = W - w if w % factor != 0 else 0
        input_ = F.pad(input_, (0, padw, 0, padh), 'reflect')

        h, w = seg_.shape[2], seg_.shape[3]
        H, W = ((h + factor) // factor) * \
            factor, ((w + factor) // factor) * factor
        padh = H - h if h % factor != 0 else 0
        padw = W - w if w % factor != 0 else 0
        seg_ = F.pad(seg_, (0, padw, 0, padh), 'reflect')
        
                
        restored = model_restoration(input_, seg_, seg_)

        # Unpad images to original dimensions
        restored = restored[:, :, :h, :w]

        restored = torch.clamp(restored, 0, 1).cpu(
        ).detach().permute(0, 2, 3, 1).squeeze(0).numpy()

        psnr.append(utils.PSNR(target, restored))
        ssim.append(utils.calculate_ssim(
            img_as_ubyte(target), img_as_ubyte(restored)))
        utils.save_img((os.path.join(result_dir, os.path.splitext(
            os.path.split(inp_path)[-1])[0] + '.bmp')), img_as_ubyte(restored))

            

psnr = np.mean(np.array(psnr))
ssim = np.mean(np.array(ssim))
print("PSNR: %f " % (psnr))
print("SSIM: %f " % (ssim))
