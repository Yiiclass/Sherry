create env
conda create -n Sherry python=3.7
conda activate Sherry
conda install pytorch=1.11 torchvision cudatoolkit=11.3 -c pytorch
pip install matplotlib scikit-learn scikit-image opencv-python yacs joblib natsort h5py tqdm tensorboard
pip install einops gdown addict future lmdb numpy pyyaml requests scipy yapf lpips
python setup.py develop --no_cuda_ext
清华源
-i https://pypi.tuna.tsinghua.edu.cn/simple

train
python3 basicsr/train.py --opt Options/NIR1.yml --gpu_id 0

test
python3 Enhancement/test_from_dataset.py --opt Options/NIR1.yml --gpu_id 2 \
--weights weight_path --dataset dataset
