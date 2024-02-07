# pcd_completion
Git clone of the repo. And run:
```
pip install -e .
```

Before running the package install the following libraries: 
```
pip install torch==1.10.2+cu113 torchvision==0.11.3+cu113 torchaudio==0.10.2 -f https://download.pytorch.org/whl/torch_stable.html 

pip install "git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib" 

pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl 

pip install vedo Ninja pyyaml easydict timm open3d
```
## USAGE:
The pretrained model weights for pcd completion are at this link **https://drive.google.com/drive/u/1/folders/1_PlETSxbObabV6OkVX1Tqea8wc4dWNi8**
```
from Completion_inf.PCD_completion import PCD_completion
pcd_in = PATH TO XYZ POINT CLOUD FILE
model_weights = PATH TO MODEL WEIGHTS
PCN = PCD_completion(model_weights)
pcd_out = PCN.comp_inf(pcd_in)
```
