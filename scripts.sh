### To train the model
python train_sirs.py --inet dsrnet --model dsrnet_model_sirs --dataset sirs_dataset --loss losses  --name dsrnet  --lambda_vgg 0.01 --lambda_rec 0.2 --if_align --seed 2018 --base_dir "[YOUR_DATA_DIR]"

python eval_sirs.py --inet dsrnet_s --model dsrnet_model_sirs --dataset sirs_dataset  --name dsrnet_s_test --hyper --if_align --resume --weight_path "./weights/dsrnet_s_epoch14.pt" --base_dir "[YOUR_DATA_DIR]"

python eval_sirs.py --inet dsrnet_l --model dsrnet_model_sirs --dataset sirs_dataset  --name dsrnet_l_test --hyper --if_align --resume --weight_path "/home/limj/huqiming/workspace/DSRNet/checkpoints/dsrnet_l_shared_b/weights/dsrnet_l_epoch18.pt" --base_dir "/home/huqiming/datasets/sirs"
