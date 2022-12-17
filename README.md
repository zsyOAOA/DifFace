# DifFace: Blind Face Restoration with Diffused Error Contraction

[Zongsheng Yue](https://zsyoaoa.github.io/), [Chen Change Loy](https://www.mmlab-ntu.com/person/ccloy/) 

[Paper](https://arxiv.org/abs/2212.06512) | [![Hugging Face](https://img.shields.io/badge/Demo-Hugging%20Face-blue)](https://huggingface.co/spaces/OAOA/DifFace) | ![visitors](https://visitor-badge.laobi.icu/badge?page_id=zsyOAOA/DifFace) 

<img src="assets/DifFace_Framework.png" width="800px"/>

:star: If DifFace is helpful to your images or projects, please help star this repo. Thanks! :hugs: 

## Update
- **2022.12.13**: Create this repo.

## Applications
### :point_right: Old Photo Enhancement
[<img src="assets/Solvay_conference.png" width="805px"/>](https://imgsli.com/MTM5NTgw)

[<img src="assets/Hepburn.png" height="555px" width="400px"/>](https://imgsli.com/MTM5NTc5) [<img src="assets/oldimg_05.png" height="555px" width="400px"/>](https://imgsli.com/MTM5NTgy)

### :point_right: Face Restoration
<img src="testdata/cropped_faces/0368.png" height="200px" width="200px"/><img src="assets/0368.png" height="200px" width="200px"/> <img src="testdata/cropped_faces/0885.png" height="200px" width="200px"/><img src="assets/0885.png" height="200px" width="200px"/>
<img src="testdata/cropped_faces/0729.png" height="200px" width="200px"/><img src="assets/0729.png" height="200px" width="200px"/> <img src="testdata/cropped_faces/0934.png" height="200px" width="200px"/><img src="assets/0934.png" height="200px" width="200px"/>

## Requirements
A suitable [conda](https://conda.io/) environment named `DifFace` can be created and activated with:

```
conda env create -f environment.yaml
conda activate taming
```

## Inference
#### :boy: Face image restoration (cropped and aligned)
```
python inference_difface.py --aligned --in_path [image folder/image path] --out_path [result folder] --gpu_id [gpu index]
```
#### :couple: Whole image enhancement
```
python inference_difface.py --in_path [image folder/image path] --out_path [result folder] --gpu_id [gpu index]
```

## Training
#### :turtle: Prepare data
1. Download the [FFHQ](https://github.com/NVlabs/ffhq-dataset) dataset, and resize them into size 512x512.
```
python datapipe/prepare/face/big2small_face.py --face_dir [Face folder(1024x1024)] --save_dir [Saving folder] --pch_size 512 
```
2. Extract the image path into 'datapipe/files_txt/ffhq512.txt'
```
python datapipe/prepare/face/split_train_val.py --face_dir [Face folder(512x512)] --save_dir [Saving folder] 
```
3. Making the testing dataset
```
python datapipe/prepare/face/make_testing_data.py --files_txt datapipe/files_txt/ffhq512.txt --save_dir [Saving folder]  
```
#### :dolphin: Train diffusion model
```
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 --nnodes=1 main_diffusion.py --cfg_path configs/training/diffsuion_ffhq512.yaml --save_dir [Logging Folder]  
```
#### :whale: Train diffused estimator (SwinIR)
```
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 --nnodes=1 main_sr.py --cfg_path configs/training/swinir_ffhq512.yaml --save_dir [Logging Folder]  
```

## License

This project is licensed under <a rel="license" href="https://github.com/sczhou/CodeFormer/blob/master/LICENSE">NTU S-Lab License 1.0</a>. Redistribution and use should follow this license.

## Acknowledgement

This project is based on [Improved Diffusion Model](https://github.com/openai/improved-diffusion).  Some codes are brought from [BasicSR](https://github.com/XPixelGroup/BasicSR), [YOLOv5-face](https://github.com/deepcam-cn/yolov5-face), and [FaceXLib](https://github.com/xinntao/facexlib). We also adopt [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) to support background image enhancement. Thanks for their awesome works.

### Contact
If you have any questions, please feel free to contact me via `zsyzam@gmail.com`.
