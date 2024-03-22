import argparse
import wandb
import train_test_autoencoder
from utils import load_config

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--path_config_file', type=str, help='Path of the config file to use')
    parser.add_argument('--name_proj_wandb', type=str, default="image_anomaly_detection", help='Name of the project for wandb')
    parser.add_argument('--id_train_wandb', type=str, help='Path of the config file to use')
    opt = parser.parse_args()

    # 1 - load config file
    path_config_file = opt.path_config_file
    name_proj_wandb = opt.name_proj_wandb
    id_train_wandb = opt.id_train_wandb
    print("path_config_file: {}".format(path_config_file))
    cfg = load_config(path_config_file)

    if id_train_wandb is None:
        wandb.init(
            # set the wandb project where this run will be logged
            project=name_proj_wandb,
            # track hyperparameters and run metadata
            config=cfg
        )
    else:
        wandb.init(
            # set the wandb project where this run will be logged
            project=name_proj_wandb,
            # track hyperparameters and run metadata
            config=cfg,
            id=id_train_wandb,
            resume='must'
        )

    print("wandb.run.id = ", wandb.run.id)

    # 2 - run train and test
    do_train = cfg["model"].get("do_train", 1.0) > 0.0
    do_test = cfg["model"].get("do_test", 1.0) > 0.0

    train_test_autoencoder.run_train_test_model(cfg=cfg,
                                                do_train=do_train,
                                                do_test=do_test)