# DSRNet: Single Image Reflection Separation via Component Synergy (ICCV 2023)

> :book: [[ICCV](https://openaccess.thecvf.com/content/ICCV2023/papers/Hu_Single_Image_Reflection_Separation_via_Component_Synergy_ICCV_2023_paper.pdf)] [[Arxiv](https://arxiv.org/abs/2308.10027)] [[Supp.](https://github.com/mingcv/DSRNet/files/12387445/full_arxiv_version_supp.pdf)]  <br>
> [Qiming Hu](https://scholar.google.com.hk/citations?user=4zasPbwAAAAJ), [Xiaojie Guo](https://sites.google.com/view/xjguo/homepage) <br>
> College of Intelligence and Computing, Tianjin University<br>


### Network Architecture
![fig_arch](https://github.com/mingcv/DSRNet/assets/31566437/2a4bb4be-9d03-40eb-b585-f2d5f8a44f42)

### Environment Preparation (Python 3.9)
```pip install -r requirements.txt```
### Data Preparation


#### Training dataset
* 7,643 images from the
  [Pascal VOC dataset](http://host.robots.ox.ac.uk/pascal/VOC/), center-cropped as 224 x 224 slices to synthesize training pairs;
* 90 real-world training pairs provided by [Zhang *et al.*](https://github.com/ceciliavision/perceptual-reflection-removal;
* 200 real-world training pairs provided by [IBCLN](https://github.com/JHL-HUST/IBCLN) (In our training setting 2, &dagger; labeled in our paper).

#### Testing dataset
* 45 real-world testing images from [CEILNet dataset](https://github.com/fqnchina/CEILNet);
* 20 real testing pairs provided by [Zhang *et al.*](https://github.com/ceciliavision/perceptual-reflection-removal);
* 20 real testing pairs provided by [IBCLN](https://github.com/JHL-HUST/IBCLN);
* 454 real testing pairs from [SIR^2 dataset](https://sir2data.github.io/), containing three subsets (i.e., Objects (200), Postcard (199), Wild (55)). 

Download all in one by [Google Drive](https://drive.google.com/file/d/1hFZItZAzAt-LnfNj-2phBRwqplDUasQy/view?usp=sharing) or [百度云](https://pan.baidu.com/s/15zlk5o_-kx3ruKj4KfOvtA?pwd=1231).
### Usage

#### Training 
Setting I (w/o Nature): ```python train_sirs.py --inet dsrnet_l --model dsrnet_model_sirs --dataset sirs_dataset --loss losses  --name dsrnet_l  --lambda_vgg 0.01 --lambda_rec 0.2 --if_align --seed 2018 --base_dir "[YOUR DATA DIR]"```

Setting II (w/ Nature): ```python train_sirs_4000.py --inet dsrnet_l --model dsrnet_model_sirs --dataset sirs_dataset --loss losses  --name dsrnet_l_4000  --lambda_vgg 0.01 --lambda_rec 0.2 --if_align --seed 2018 --base_dir "[YOUR DATA DIR]"```
#### Evaluation 
Setting I (w/o Nature): ```python eval_sirs.py --inet dsrnet_l --model dsrnet_model_sirs --dataset sirs_dataset  --name dsrnet_l_test --if_align --resume --weight_path "./weights/dsrnet_l_epoch18.pt" --base_dir "[YOUR_DATA_DIR]"```

Setting II (w/ Nature): ```python eval_sirs_4000.py --inet dsrnet_l --model dsrnet_model_sirs --dataset sirs_dataset  --name dsrnet_l_4000_test --if_align --resume --weight_path "./weights/dsrnet_l_4000_epoch33.pt" --base_dir "[YOUR_DATA_DIR]"```

More commands can be found in [scripts.sh](https://github.com/mingcv/DSRNet/blob/main/scripts.sh).

#### Trained weights

Download the trained weights by [Google Drive](https://drive.google.com/drive/folders/1AIS9-EgBN3_q-TCq7W0j5OeWMgLO_de0?usp=sharing) or [百度云](https://pan.baidu.com/s/17jW9oBAfIZ03FKa3jc-qig?pwd=1231) and drop them into the "weights" dir.

![image](https://github.com/mingcv/DSRNet/assets/31566437/1bfbd2c6-ca80-40ac-9c9e-a30ba0d095c7)


#### Visual comparison on real20 and SIR^2
![image](https://github.com/mingcv/DSRNet/assets/31566437/0d32ee2b-4c9e-46ad-834b-6b08fc6aadd5)


#### Impressive Restoration Quality of Reflection Layers
![image](https://github.com/mingcv/DSRNet/assets/31566437/e75e2abb-c413-4250-acd1-3f10e9d887b1)

