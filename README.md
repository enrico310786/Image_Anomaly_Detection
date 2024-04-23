# Image_Anomaly_Detection

python 3.10

prima i requirement

poi anomalib install


### Patchcore

python train_anomalib/train_patchcore_anomalib.py --dataset_root /home/randellini/Image_Anomaly_Detection/dataset/images_lego_256/two_up --name_normal_dir 90_DEG --name_wandb_experiment patchcore_twoup_v1 --name two_up

### EfficientAD

python train_anomalib/train_efficientAD_anomalib.py --dataset_root /home/randellini/Image_Anomaly_Detection/dataset/images_lego_256/one_up --name_normal_dir 90_DEG --name_wandb_experiment effAD_oneup_v1 --name one_up --max_epochs 100 --patience 10 

python train_anomalib/train_efficientAD_anomalib.py --dataset_root /home/randellini/Image_Anomaly_Detection/dataset/images_lego_256/one_up --name_normal_dir 90_DEG --name_wandb_experiment effAD_oneup_v1 --name one_up --max_epochs 100 --patience 10 --data_augmentation true

### Test

 python infer_anomalib/test_model.py --path_torch_model /home/enrico/Dataset/images_anomaly/results/Patchcore/one_up/v0/weights/torch/model.pt --path_dataset /home/enrico/Dataset/images_anomaly/dataset_lego/images_lego_256/one_up --name one_up --dir_result /home/enrico/Dataset/images_anomaly/results/Patchcore/one_up/v0 