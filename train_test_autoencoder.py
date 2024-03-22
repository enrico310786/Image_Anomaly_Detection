import os
import pandas as pd
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR
from tqdm import tqdm
import yaml
from data import create_loaders
import numpy as np
import matplotlib.pyplot as plt
import random
import wandb
import seaborn as sns
import gc
from torchinfo import summary
from model import find_last_checkpoint_file, Autoencoder


def train_batch(inputs, model, optimizer, criterion):
    model.train()
    target, outputs = model(inputs)
    loss = criterion(outputs, target)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return loss.item()


@torch.no_grad()
def val_loss(inputs, model, criterion):
    model.eval()
    target, outputs = model(inputs)
    val_loss = criterion(outputs, target)
    return val_loss.item()


def calculate_errors(device,
                     model,
                     dataloader,
                     df_distribution,
                     label2class):
    '''
    Calculate the error reconstruction of the embedding vectors
    '''

    labels_list = []
    reconstructed_embeddings_array = None
    embeddings_array = None
    model = model.eval()

    with torch.no_grad():
        # cycle on all batches
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = list(labels)
            embeddings, reconstructed_embeddings = model(inputs)

            if embeddings_array is None:
                embeddings_array = embeddings.detach().cpu().numpy()
            else:
                embeddings_array = np.vstack((embeddings_array, embeddings.detach().cpu().numpy()))

            if reconstructed_embeddings_array is None:
                reconstructed_embeddings_array = reconstructed_embeddings.detach().cpu().numpy()
            else:
                reconstructed_embeddings_array = np.vstack((reconstructed_embeddings_array, reconstructed_embeddings.detach().cpu().numpy()))

            labels_list.extend(labels)

        # transform to numpy array
        labels_list = np.array(labels_list)

        print('len(labels_list): ', len(labels_list))
        print('reconstructed_embeddings_array.shape: ', reconstructed_embeddings_array.shape)
        print('embeddings_array.shape: ', embeddings_array.shape)

        # iter over the array to find error and distances from the correspondig centroid. The centroid is found using the label of the class
        for label, emb, rec_emb in zip(labels_list, embeddings_array, reconstructed_embeddings_array):
            #error = (np.square(emb - rec_emb)).mean()
            error = (np.square(emb - rec_emb)).sum() # è la stessa cosa di criterion = nn.MSELoss(reduction='sum'). Fa la somma dei quadrati delle differenze

            df_distribution = df_distribution.append({'CLASS': label2class[label],
                                                      'RECONSTRUCTION_ERROR': error}, ignore_index=True)

        return df_distribution


def train_model(cfg,
                device,
                model,
                criterion,
                optimizer,
                lr_scheduler,
                train_loader,
                val_loader,
                best_epoch,
                num_epoch,
                best_val_epoch_loss,
                checkpoint_dir,
                epoch_start_unfreeze=None,
                layer_start_unfreeze=None,
                scheduler_type=None):

    train_losses = []
    val_losses = []

    print("Start training")
    freezed = True
    for epoch in range(best_epoch, num_epoch):

        if epoch_start_unfreeze is not None and epoch >= epoch_start_unfreeze and freezed:
            print("****************************************")
            print("Unfreeze the base model weights")
            if layer_start_unfreeze is not None:
                print("unfreeze the layers greater and equal to layer_start_unfreeze: ", layer_start_unfreeze)
                #in this case unfreeze only the layers greater and equal the unfreezing_block layer
                for i, properties in enumerate(model.named_parameters()):
                    if i >= layer_start_unfreeze:
                        #print("Unfreeze model layer: {} -  name: {}".format(i, properties[0]))
                        properties[1].requires_grad = True
            else:
                # in this case unfreeze all the layers of the model
                print("unfreeze all the layer of the model")
                for name, param in model.named_parameters():
                    param.requires_grad = True

            freezed = False
            print("*****************************************")
            print("Model layer info after unfreezing")

            print("Check layers properties")
            for i, properties in enumerate(model.named_parameters()):
                print("Model layer: {} -  name: {} - requires_grad: {} ".format(i, properties[0],
                                                                                properties[1].requires_grad))
            print("*****************************************")

            pytorch_total_params = sum(p.numel() for p in model.parameters())
            pytorch_total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print("pytorch_total_params: ", pytorch_total_params)
            print("pytorch_total_trainable_params: ", pytorch_total_trainable_params)

            print("*****************************************")

        # define empty lists for the values of the loss of train and validation obtained in the batch of the current epoch
        # then at the end I take the average and I get the final values of the whole era
        train_epoch_losses = []
        val_epoch_losses = []

        # cycle on all train batches of the current epoch by executing the train_batch function
        for inputs, _ in tqdm(train_loader, desc=f"epoch {str(epoch)} | train"):
            inputs = inputs.to(device)
            batch_loss = train_batch(inputs, model, optimizer, criterion)
            train_epoch_losses.append(batch_loss)
            torch.cuda.empty_cache()
        train_epoch_loss = np.array(train_epoch_losses).mean()

        # cycle on all batches of val of the current epoch by calculating the accuracy and the loss function
        for inputs, _ in tqdm(val_loader, desc=f"epoch {str(epoch)} | val"):
            inputs = inputs.to(device)
            validation_loss = val_loss(inputs, model, criterion)
            val_epoch_losses.append(validation_loss)
            torch.cuda.empty_cache()
        val_epoch_loss = np.mean(val_epoch_losses)

        wandb.log({'Learning Rate': optimizer.param_groups[0]['lr'],
                   'Train Loss': train_epoch_loss,
                   'Valid Loss': val_epoch_loss})

        print("Epoch: {} - LR:{} - Train Loss: {:.4f} - Val Loss: {:.4f}".format(int(epoch), optimizer.param_groups[0]['lr'], train_epoch_loss, val_epoch_loss))

        train_losses.append(train_epoch_loss)
        val_losses.append(val_epoch_loss)

        print("Plot learning curves")
        plot_learning_curves(epoch - best_epoch + 1, train_losses, val_losses, checkpoint_dir)

        if best_val_epoch_loss > val_epoch_loss:
            print("We have a new best model! Save the model")
            # update best_val_epoch_loss
            best_val_epoch_loss = val_epoch_loss
            save_obj = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'best_eval_loss': best_val_epoch_loss
            }
            print("Save best checkpoint at: {}".format(os.path.join(checkpoint_dir, 'best.pth')))
            torch.save(save_obj, os.path.join(checkpoint_dir, 'best.pth'),  _use_new_zipfile_serialization=False)
            print("Save latest checkpoint at: {}".format(os.path.join(checkpoint_dir, 'latest.pth')))
            torch.save(save_obj, os.path.join(checkpoint_dir, 'latest.pth'),  _use_new_zipfile_serialization=False)
        else:
            print("Save the current model")
            save_obj = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'best_eval_loss': best_val_epoch_loss
            }
            print("Save latest checkpoint at: {}".format(os.path.join(checkpoint_dir, 'latest.pth')))
            torch.save(save_obj, os.path.join(checkpoint_dir, 'latest.pth'),  _use_new_zipfile_serialization=False)

        if scheduler_type == "ReduceLROnPlateau":
            print("lr_scheduler.step(val_epoch_loss)")
            lr_scheduler.step(val_epoch_loss)
        else:
            print("lr_scheduler.step()")
            lr_scheduler.step()

        torch.cuda.empty_cache()
        gc.collect()
        print("---------------------------------------------------------")

    print("End training")
    return


def plot_learning_curves(epochs, train_losses, val_losses, path_save):
    '''
    Plot learning curves of the training model
    '''
    x_axis = range(0, epochs)

    plt.figure(figsize=(27,9))
    plt.suptitle('Learning curves ', fontsize=18)

    plt.subplot(121)
    plt.plot(x_axis, train_losses, label='Training Loss')
    plt.plot(x_axis, val_losses, label='Validation Loss')
    plt.legend()
    plt.title('Train and Validation Losses', fontsize=16)
    plt.xlabel('Epochs', fontsize=16)
    plt.ylabel('Loss', fontsize=16)

    plt.savefig(os.path.join(path_save, "learning_curves.png"))


def analyze_error_distribution(df_distribution, dir_save_results):

    print("-------------------------------------------------------------------")
    print("RECONTRUCTION_ERROR DISTRIBUTION GROUPED BY CLASS")
    print("")
    desc_grouped = df_distribution.groupby('CLASS')["RECONSTRUCTION_ERROR"].describe()
    print("RECONTRUCTION_ERROR distrbution")
    print(desc_grouped)
    print("-------------------------------------------------------------------")

    # boxplot
    plt.figure(figsize=(15, 15))
    sns.boxplot(data=df_distribution, x="CLASS", y="RECONSTRUCTION_ERROR")
    plt.xticks(rotation=30, ha='right', rotation_mode='anchor')
    plt.title('Reconstruction error grouped by classes', fontsize=12)
    plt.savefig(os.path.join(dir_save_results, "error_distribution_grouped_by_classes.png"))


def run_train_test_model(cfg, do_train, do_test):

    seed_everything(42)
    checkpoint = None
    best_epoch = 0
    best_val_epoch_loss = float('inf')

    dataset_cfg = cfg["dataset"]
    model_cfg = cfg["model"]

    dataset_path = dataset_cfg['dataset_path']
    path_dataset_train_csv = dataset_cfg['path_dataset_train_csv']
    path_dataset_val_csv = dataset_cfg['path_dataset_val_csv']
    path_dataset_test_csv = dataset_cfg['path_dataset_test_csv']
    saving_dir_experiments = model_cfg['saving_dir_experiments']
    saving_dir_model = model_cfg['saving_dir_model']
    num_epoch = model_cfg['num_epoch']
    epoch_start_unfreeze = model_cfg.get("epoch_start_unfreeze", None)
    layer_start_unfreeze = model_cfg.get("layer_start_unfreeze", None)
    batch_size = dataset_cfg['batch_size']
    scheduler_type = model_cfg['scheduler_type']
    learning_rate = model_cfg['learning_rate']
    lr_patience = model_cfg.get("lr_patience", None)
    milestones = model_cfg.get("milestones", None)
    loss_function = model_cfg.get("loss_function", None)

    scheduler_step_size = model_cfg.get("scheduler_step_size", None)
    if scheduler_step_size is not None:
        scheduler_step_size = int(scheduler_step_size)

    lr_factor = model_cfg.get("lr_factor", None)
    T_max = model_cfg.get("T_max", None)
    eta_min = model_cfg.get("eta_min", None)

    use_pretrained_scheduler = model_cfg.get("use_pretrained_scheduler", 1.0) > 0.0

    # load, filter and shuffle the dataset
    df_dataset_train = pd.read_csv(path_dataset_train_csv)
    df_dataset_val = pd.read_csv(path_dataset_val_csv)
    df_dataset_test = pd.read_csv(path_dataset_test_csv)

    df_dataset_train = df_dataset_train.sample(frac=1).reset_index(drop=True)

    print("df_dataset_train.shape: ", df_dataset_train.shape)
    print("df_dataset_val.shape: ", df_dataset_val.shape)
    print("df_dataset_test.shape: ", df_dataset_test.shape)

    # create the directories with the structure required by the project
    print("create the project structure")
    print("saving_dir_experiments: {}".format(saving_dir_experiments))
    saving_dir_model = os.path.join(saving_dir_experiments, saving_dir_model)
    print("saving_dir_model: {}".format(saving_dir_model))
    os.makedirs(saving_dir_experiments, exist_ok=True)
    os.makedirs(saving_dir_model, exist_ok=True)

    # save the config file
    yaml_config_path = os.path.join(saving_dir_model, "config.yaml")
    with open(yaml_config_path, 'w') as file:
        documents = yaml.dump(cfg, file)

    # create the dataloaders
    train_loader, val_loader, test_loader = create_loaders(df_dataset_train=df_dataset_train,
                                                              df_dataset_val=df_dataset_val,
                                                              df_dataset_test=df_dataset_test,
                                                              cfg=cfg,
                                                              dataset_path=dataset_path,
                                                              batch_size=batch_size)

    # set the device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device: {}".format(device))

    # create the model
    print("Create the model")
    model = Autoencoder(cfg["model"]).to(device)

    print("*****************************************")
    print("Model summary")
    summary(model=model,
            input_size=(1, 3, cfg["data"]["size"], cfg["data"]["size"]),
            col_names=["input_size", "output_size", "num_params", "trainable"],
            col_width=20,
            row_settings=["var_names"])
    print("*****************************************")
    print("*****************************************")
    print("Model layer info")
    for i, properties in enumerate(model.named_parameters()):
        print("Model layer: {} -  name: {} - requires_grad: {} ".format(i, properties[0],
                                                                        properties[1].requires_grad))
    print("")
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    pytorch_total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("pytorch_total_params: ", pytorch_total_params)
    print("pytorch_total_trainable_params: ", pytorch_total_trainable_params)
    print("*****************************************")
    checkpoint_dir = saving_dir_model

    if do_train:
        # look if exist a checkpoint
        path_last_checkpoint = find_last_checkpoint_file(checkpoint_dir)
        if path_last_checkpoint is not None:
            print("Load checkpoint from path: ", path_last_checkpoint)
            checkpoint = torch.load(path_last_checkpoint, map_location=torch.device(device))
            model.load_state_dict(checkpoint['model'])
            model = model.to(device)

        # Set optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # set the scheduler
        scheduler = None
        if scheduler_type == "ReduceLROnPlateau":
            scheduler = ReduceLROnPlateau(optimizer=optimizer,
                                          mode='max',
                                          patience=lr_patience,
                                          verbose=True,
                                          factor=lr_factor)
        elif scheduler_type == "StepLR":
            print("StepLR")
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer,
                                                        step_size=scheduler_step_size,
                                                        gamma=lr_factor)
        elif scheduler_type == "MultiStepLR":
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
                                                             milestones=milestones,
                                                             gamma=lr_factor)
        elif scheduler_type == "CosineAnnealingLR":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,
                                                                   T_max=T_max,
                                                                   eta_min=eta_min)
        elif scheduler_type == "CosineAnnealingWarmRestarts":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer,
                                                                             T_0=T_max,
                                                                             T_mult=1,
                                                                             eta_min=eta_min)

        # set the loss
        #criterion = nn.MSELoss()
        if loss_function is None:
            criterion = nn.MSELoss(reduction='sum')
        elif loss_function == "MSE":
            criterion = nn.MSELoss(reduction='sum')
        elif loss_function == "HUBERLOSS":
            criterion = nn.HuberLoss(reduction='sum')

        if checkpoint is not None:
            print('Load the optimizer from the last checkpoint')
            optimizer.load_state_dict(checkpoint['optimizer'])
            if use_pretrained_scheduler:
                print("use_pretrained_scheduler: TRUE")
                scheduler.load_state_dict(checkpoint["scheduler"])
            else:
                print("use_pretrained_scheduler: FALSE")

            print('Latest epoch of the checkpoint: {}'.format(checkpoint['epoch']))
            print('Setting the new starting epoch: {}'.format(checkpoint['epoch'] + 1))
            best_epoch = checkpoint['epoch'] + 1
            print('Setting best best_eval_loss from checkpoint: {}'.format(checkpoint['best_eval_loss']))
            best_val_epoch_loss = checkpoint['best_eval_loss']

        # run train model function
        train_model(cfg=cfg,
                    device=device,
                    model=model,
                    criterion=criterion,
                    optimizer=optimizer,
                    lr_scheduler=scheduler,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    best_epoch=best_epoch,
                    num_epoch=num_epoch,
                    best_val_epoch_loss=best_val_epoch_loss,
                    checkpoint_dir=checkpoint_dir,
                    epoch_start_unfreeze=epoch_start_unfreeze,
                    layer_start_unfreeze=layer_start_unfreeze,
                    scheduler_type=scheduler_type)
        print("-------------------------------------------------------------------")
        print("-------------------------------------------------------------------")

    if do_test:

        print("Execute Inference on Train, Val and Anomaly Dataset with best checkpoint")

        path_last_checkpoint = find_last_checkpoint_file(checkpoint_dir=checkpoint_dir, use_best_checkpoint=True)
        if path_last_checkpoint is not None:
            print("Upload the best checkpoint at the path: ", path_last_checkpoint)
            checkpoint = torch.load(path_last_checkpoint, map_location=torch.device(device))
            model.load_state_dict(checkpoint['model'])
            model = model.to(device)


        # go through the lines of the dataset
        class2label = {}
        for index, row in df_dataset_train.iterrows():
            class_name = row["CLASS"]
            label = row["LABEL"]

            if class_name not in class2label:
                class2label[class_name] = label
        #sort the value of the label
        class2label = dict(sorted(class2label.items(), key=lambda item: item[1]))
        label2class = {k: v for (v, k) in class2label.items()}
        print("class2label: ", class2label)
        print("label2class: ", label2class)

        print("-------------------------------------------------------------------")
        print("-------------------------------------------------------------------")

        # create dataset for distribution
        df_distribution = pd.DataFrame(columns=['CLASS', 'RECONSTRUCTION_ERROR'])

        print("Inference on test dataset")
        df_distribution = calculate_errors(device=device,
                                           model=model,
                                           dataloader=test_loader,
                                           df_distribution=df_distribution,
                                           label2class=label2class)

        torch.cuda.empty_cache()
        gc.collect()
        print("-------------------------------------------------------------------")
        print("-------------------------------------------------------------------")
        analyze_error_distribution(df_distribution, checkpoint_dir)

        print("Save the errors and dist distribution dataset at: ", os.path.join(checkpoint_dir, "reconstruction_error_dataset.csv"))
        df_distribution.to_csv(os.path.join(checkpoint_dir, "reconstruction_error_dataset.csv"), index=False)

        print("End test")

    wandb.finish()


def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True