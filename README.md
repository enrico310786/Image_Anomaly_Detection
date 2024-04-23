# Image_Anomaly_Detection

python 3.10

prima i requirement

poi anomalib install


### Patchcore

python train_anomalib/train_patchcore_anomalib.py --dataset_root /home/randellini/Image_Anomaly_Detection/dataset/images_lego_256/two_up --name_normal_dir 90_DEG --name_wandb_experiment patchcore_twoup_v1 --name two_up

### EfficientAD

python train_anomalib/train_efficientAD_anomalib.py --dataset_root /home/randellini/Image_Anomaly_Detection/dataset/images_lego_256/one_up --name_normal_dir 90_DEG --name_wandb_experiment effAD_oneup_v1 --name one_up --max_epochs 100 --patience 10 
