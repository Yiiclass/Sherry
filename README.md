# Object-Aware NIR-to-Visible Translation
We provide the Pytorch implementation of "Object-Aware NIR-to-Visible Translation"

## Abstract
While near-infrared (NIR) imaging technology is essential for assisted driving and safety monitoring systems, its monochromatic nature and detail limitations hinder its broader application, which prompts the development of NIR-to-visible translation tasks.
However, the performance of existing translation methods is limited by the neglected disparities between NIR and visible imaging and the lack of paired training data.
To address these challenges, we propose a novel object-aware framework for NIR-to-visible translation. Our approach decomposes the visible image recovery into object-independent luminance sources and object-specific reflective components, processing them separately to bridge the gap between NIR and visible imaging under various lighting conditions. Leveraging prior segmentation knowledge enhances our model's ability to identify and understand the separated object reflection. We also collect the Fully Aligned NIR-Visible Image Dataset, a large-scale dataset comprising fully matched pairs of NIR and visible images captured with a multi-sensor coaxial camera. Empirical evaluations demonstrate the superiority of our approach over existing methods, producing visually compelling results on mainstream datasets.


## Highlights
+ Observing the large variance for illumination on visible and NIR ranges, we demonstrate that decomposing the illumination and object reflectance to process them separately can effectively enhance NIR2VIS translation task. 

<img width="726" alt="image" src="https://github.com/Yiiclass/Sherry/assets/69071622/91c9cc8a-b93b-441d-8b8f-22b5dfaa4a62">

+ Incorporation of segmentation as an object-aware prior knowledge can facilitate the estimation of object reflectance.

+ We collect a Fully Aligned NIR-VIS Image Dataset (FANVID) containing fully paired data in dynamic scenes. The experimental results on mainstream NIR-VIS datasets indicate our method's superiority over leading methods and yield more visually appealing results.

<img width="518" alt="image" src="https://github.com/Yiiclass/Sherry/assets/69071622/cc3e6f01-55e5-4290-84b0-7423d464ef7e">


## Dataset
Download our FANVID dataset at [Onedirve](https://1drv.ms/f/c/e976acca7b9fcd1f/EiDybm6th_dCmf7v0HDM-hYBjuHcOsVkjCa2067pgzaUxQ?e=eVisVX) or [Baidu Netdisk](https://pan.baidu.com/s/1NKEli68HZ9gYQa0QRbD9mw?pwd=Cool), extraction code: Cool

The datasets used for evaluation are [EPFL](https://www.epfl.ch/labs/ivrl/research/downloads/rgb-nir-scene-dataset/) and [ICVL](https://icvl.cs.bgu.ac.il/hyperspectral/).


```
dataset
+-- 'Train'
|   +-- paired_NIR1
|   |   +-- Train_00001.bmp
|   |   +-- Train_00002.bmp
|   |   +-- Train_00003.bmp
|   |   +-- Train_00004.bmp
|   |   +-- ...
|   +-- paired_NIR2
|   |   +-- Train_00001.bmp
|   |   +-- Train_00002.bmp
|   |   +-- Train_00003.bmp
|   |   +-- Train_00004.bmp
|   |   +-- ...
|   +-- paired_RGB
|   |   +-- Train_00001.bmp
|   |   +-- Train_00002.bmp
|   |   +-- Train_00003.bmp
|   |   +-- Train_00004.bmp
|   |   +-- ...
|   +-- seg_mask2former_NIR1
|   |   +-- Train_00001.npy
|   |   +-- Train_00002.npy
|   |   +-- Train_00003.npy
|   |   +-- Train_00004.npy
|   |   +-- ...
|   +-- seg_mask2former_NIR2
|   |   +-- Train_00001.npy
|   |   +-- Train_00002.npy
|   |   +-- Train_00003.npy
|   |   +-- Train_00004.npy
|   |   +-- ...


+-- 'Test'
|   +-- paired_NIR1
|   |   +-- Test_00001.bmp
|   |   +-- Test_00002.bmp
|   |   +-- Test_00003.bmp
|   |   +-- Test_00004.bmp
|   |   +-- ...
|   +-- paired_NIR2
|   |   +-- Test_00001.bmp
|   |   +-- Test_00002.bmp
|   |   +-- Test_00003.bmp
|   |   +-- Test_00004.bmp
|   |   +-- ...
|   +-- paired_RGB
|   |   +-- Test_00001.bmp
|   |   +-- Test_00002.bmp
|   |   +-- Test_00003.bmp
|   |   +-- Test_00004.bmp
|   |   +-- ...
|   +-- seg_mask2former_NIR1
|   |   +-- Test_00001.npy
|   |   +-- Test_00002.npy
|   |   +-- Test_00003.npy
|   |   +-- Test_00004.npy
|   |   +-- ...
|   +-- seg_mask2former_NIR2
|   |   +-- Test_00001.npy
|   |   +-- Test_00002.npy
|   |   +-- Test_00003.npy
|   |   +-- Test_00004.npy
|   |   +-- ...

```


## Usage
+ Create conda environment and download our repository

```
conda create -n Sherry python=3.7
conda activate Sherry
conda install pytorch=1.11 torchvision cudatoolkit=11.3 -c pytorch
pip install matplotlib scikit-learn scikit-image opencv-python yacs joblib natsort h5py tqdm tensorboard
pip install einops gdown addict future lmdb numpy pyyaml requests scipy yapf lpips
```

+ Install BasicSR

```
python setup.py develop --no_cuda_ext
```


### Training
python3 basicsr/train.py --opt Options/NIR1.yml --gpu_id 0



### Testing
python3 Enhancement/test_from_dataset.py --opt Options/NIR1.yml --gpu_id 2 \
--weights weight_path --dataset dataset


## Results
### ### Quantitative comparison on the FANVID dataset

FANVID NIR1/NIR2, respectively, indicate using the 700-800nm band NIR1 or 820-1100nm band NIR2 images as input. All the methods have been retrained on both the NIR and RGB domains of our FANVID dataset, ensuring consistency in inputs and uniformity in settings.


| Method               | PSNR ↑ | SSIM ↑ | Delta-E ↓ | FID ↓   | PSNR ↑ | SSIM ↑ | Delta-E ↓ | FID ↓   |
|----------------------|--------|--------|-----------|---------|--------|--------|-----------|---------|
|                      |**FANVID NIR1**       |        |           |         |**FANVID NIR2**          |        |           |         |
| Retinexformer        | 24.61  | 0.86   | 6.55      | 39.72   | 22.46  | 0.79   | 8.64      | 51.01   |
| CT2                  | 17.41  | 0.68   | 20.82     | 52.75   | 14.80  | 0.51   | 29.45     | 61.43   |
| FastCUT              | 18.65  | 0.71   | 15.94     | 44.29   | 16.74  | 0.63   | 20.03     | 58.96   |
| pix2pix              | 20.10  | 0.70   | 12.42     | 54.34   | 18.00  | 0.60   | 15.90     | 66.01   |
| CycleGAN             | 18.63  | 0.71   | 16.33     | 45.72   | 16.35  | 0.61   | 21.58     | 52.02   |
| **NIRcolor**         | 15.71  | 0.56   | 27.38     | 47.70   | 14.19  | 0.46   | 32.39     | 60.60   |
| TLM                  | 20.65  | 0.75   | 11.23     | 49.79   | 18.76  | 0.66   | 14.47     | 63.25   |
| Ours                 | **25.57**  | **0.87**   | **5.78**      | **37.15**   | **23.37**  | **0.80**   | **7.61**      | **48.98**   |


### Quantitative comparison on the EPFL and ICVL datasets

All the methods have been retrained on both the NIR and RGB domains of the EPFL/ICVL datasets, ensuring consistency in inputs and uniformity in settings.


| Method               | PSNR ↑ | SSIM ↑ | Delta-E ↓ | FID ↓   | PSNR ↑ | SSIM ↑ | Delta-E ↓ | FID ↓   |
|----------------------|--------|--------|-----------|---------|--------|--------|-----------|---------|
|                      |**EPFL**       |        |           |         | **ICVL**         |        |           |         |
| Retinexformer        | 17.93  | 0.64   | 14.89     | 130.67  | 27.12  | 0.89   | 7.78      | 88.31   |
| CT2                  | 12.68  | 0.29   | 27.03     | 116.73  | 17.96  | 0.70   | 20.91     | 134.53  |
| FastCUT              | 10.30  | 0.10   | 33.77     | 255.39  | 18.98  | 0.65   | 18.70     | 169.97  |
| pix2pix              | 16.90  | 0.55   | 16.17     | 121.02  | 24.84  | 0.81   | 9.70      | 124.05  |
| CycleGAN             | 15.13  | 0.55   | 21.87     | 119.64  | 19.58  | 0.63   | 18.25     | 169.91  |
| NIRcolor             | 13.99  | 0.53   | 29.37     | 150.60  | 16.36  | 0.69   | 25.52     | 142.85  |
| TLM                  | 15.63  | 0.49   | 19.08     | 193.17  | 24.53  | 0.82   | 9.61      | 130.51  |
| Ours                 | **18.41** | **0.65** | **13.85** | **113.90** | **27.47** | **0.90** | **7.43** | **82.95** |

## Citation


## contact
If you have any problems, please feel free to contact me at yiiclass@qq.com
