conda create -n dudu233 python=3.11

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip3 install torch torchvision torchaudio -f https://mirrors.aliyun.com/pytorch-wheels/cu124
pip3 install torch==2.6.0 torchvision torchaudio -f https://mirrors.aliyun.com/pytorch-wheels/cu118

pip install -r requirements.txt



cd siclib

pip install -U openmim
mim install mmcv-full

pip install natten==0.17.5
pip3 install natten==0.17.5+torch260cu124 -f https://shi-labs.com/natten/wheels/

cd kernels/selective_scan && pip install .

pip install -r requirements.txt

我们的环境 CU116

conda create -n dudu233 python=3.11

pip3 install torch==2.6.0 torchvision torchaudio -f https://mirrors.aliyun.com/pytorch-wheels/cu118

pip install -r requirements.txt


cd siclib

pip install -U openmim
mim install mmcv-full

pip install natten==0.17.5+torch260cu124 -f https://shi-labs.com/natten/wheels/

cd kernels/selective_scan && pip install .

pip install -r requirements.txt