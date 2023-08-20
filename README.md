# DSRNet: Single Image Reflection Separation via Component Synergy (ICCV 2023)

> :book: [Arxiv] [[Supp.](https://github.com/mingcv/DSRNet/files/12387445/full_arxiv_version_supp.pdf)] <br>
> [Qiming Hu](https://scholar.google.com.hk/citations?user=4zasPbwAAAAJ), [Xiaojie Guo](https://sites.google.com/view/xjguo/homepage) <br>
> College of Intelligence and Computing, Tianjin University<br>


### Network Architecture
![fig_arch](https://github.com/mingcv/DSRNet/assets/31566437/2a4bb4be-9d03-40eb-b585-f2d5f8a44f42)

### Data Preparation

#### Training dataset
* 7,643 images from the
  [Pascal VOC dataset](http://host.robots.ox.ac.uk/pascal/VOC/), center-cropped as 224 x 224 slices to synthesize training pairs.
* 90 real-world training pairs provided by [Zhang *et al.*](https://github.com/ceciliavision/perceptual-reflection-removal)

#### Tesing dataset
* 45 real-world testing images from [CEILNet dataset](https://github.com/fqnchina/CEILNet).
* 20 real testing pairs provided by [Zhang *et al.*](https://github.com/ceciliavision/perceptual-reflection-removal)
* 454 real testing pairs from [SIR^2 dataset](https://sir2data.github.io/), containing three subsets (i.e., Objects (200), Postcard (199), Wild (55)). 

### Usage

#### Training 
* For stage 1: ```python train_sirs.py --inet ytmt_ucs --model ytmt_model_sirs --name ytmt_ucs_sirs --hyper --if_align```
* For stage 2: ```python train_twostage_sirs.py --inet ytmt_ucs --model twostage_ytmt_model --name ytmt_uct_sirs --hyper --if_align --resume --resume_epoch xx --checkpoints_dir xxx```

#### Testing 
```python test_sirs.py --inet ytmt_ucs_old --model twostage_ytmt_model --name ytmt_uct_sirs_test --hyper --if_align --resume --icnn_path ./checkpoints/ytmt_uct_sirs/ytmt_uct_sirs_68_077_00595364.pt```

*Note: "ytmt_ucs_old" is only for our provided checkpoint, and please change it as "ytmt_ucs" when you train our model by yourself, since it is a refactorized verison for a better view.*

#### Trained weights
[Google Drive](https://drive.google.com/file/d/1yOKFzhhFUdbKzU3eafYKFLN7AdHqW4_7/view?usp=sharing)

#### Visual comparison on real20 and SIR^2
![image](https://github.com/mingcv/DSRNet/assets/31566437/0d32ee2b-4c9e-46ad-834b-6b08fc6aadd5)


#### Impressive Restoration Quality of Reflection Layers
![image](https://github.com/mingcv/DSRNet/assets/31566437/e75e2abb-c413-4250-acd1-3f10e9d887b1)

