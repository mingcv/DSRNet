# DSRNet: Single Image Reflection Separation via Component Synergy (ICCV 2023)

> :book: [[Arxiv](https://arxiv.org/abs/2308.10027)] [[Supp.](https://github.com/mingcv/DSRNet/files/12387445/full_arxiv_version_supp.pdf)] <br>
> [Qiming Hu](https://scholar.google.com.hk/citations?user=4zasPbwAAAAJ), [Xiaojie Guo](https://sites.google.com/view/xjguo/homepage) <br>
> College of Intelligence and Computing, Tianjin University<br>


### Network Architecture
![fig_arch](https://github.com/mingcv/DSRNet/assets/31566437/2a4bb4be-9d03-40eb-b585-f2d5f8a44f42)

### Data Preparation

#### Training dataset
* 7,643 images from the
  [Pascal VOC dataset](http://host.robots.ox.ac.uk/pascal/VOC/), center-cropped as 224 x 224 slices to synthesize training pairs.
* 90 real-world training pairs provided by [Zhang *et al.*](https://github.com/ceciliavision/perceptual-reflection-removal)

#### Testing dataset
* 45 real-world testing images from [CEILNet dataset](https://github.com/fqnchina/CEILNet).
* 20 real testing pairs provided by [Zhang *et al.*](https://github.com/ceciliavision/perceptual-reflection-removal)
* 454 real testing pairs from [SIR^2 dataset](https://sir2data.github.io/), containing three subsets (i.e., Objects (200), Postcard (199), Wild (55)). 

### Usage

#### Training ```
```python train_sirs.py --inet dsrnet_l --model dsrnet_model_sirs --dataset sirs_dataset --loss losses  --name dsrnet_l  --lambda_vgg 0.01 --lambda_rec 0.2 --if_align --seed 2018 --base_dir "[YOUR DATA DIR]"```
#### Testing 
```python test_sirs.py --inet dsrnet_l --model dsrnet_model_sirs --dataset sirs_dataset --name dsrnet_l_test --if_align --base_dir "[YOUR DATA DIR]" --resume --icnn_path ./checkpoints/ytmt_uct_sirs/ytmt_uct_sirs_68_077_00595364.pt```

#### Trained weights
[Google Drive](https://drive.google.com/file/d/1yOKFzhhFUdbKzU3eafYKFLN7AdHqW4_7/view?usp=sharing)

#### Visual comparison on real20 and SIR^2
![image](https://github.com/mingcv/DSRNet/assets/31566437/0d32ee2b-4c9e-46ad-834b-6b08fc6aadd5)


#### Impressive Restoration Quality of Reflection Layers
![image](https://github.com/mingcv/DSRNet/assets/31566437/e75e2abb-c413-4250-acd1-3f10e9d887b1)

