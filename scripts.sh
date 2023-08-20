### To train the model
python train_sirs.py --inet dsrnet --model dsrnet_model_sirs --dataset sirs_dataset --loss losses  --name dsrnet  --lambda_vgg 0.01 --lambda_rec 0.2 --if_align --seed 2018 --base_dir "[YOUR_DATA_DIR]"
