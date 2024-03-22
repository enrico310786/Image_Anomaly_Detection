import torch
import os
from torchvision import transforms
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import cv2
import albumentations as A


class Dataset(torch.utils.data.Dataset):
    def __init__(self, df_dataset, cfg, dataset_path, is_train=False) -> None:
        super().__init__()

        self.df_dataset = df_dataset
        self.dataset_path = dataset_path
        self.cfg = cfg
        self.do_resize = cfg["data"].get("do_resize", 1.0) > 0.0

        if cfg["data"]["normalization"] == "pm1":
            print("Apply pm1 normalization")
            self.normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        elif cfg["data"]["normalization"] == "imagenet":
            print("Apply imagenet normalization")
            self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        elif cfg["data"]["normalization"] == "None":
            print("No normalization to apply")
            self.normalize = None

        self.is_train = is_train
        self.transformation = A.Compose([A.RandomBrightnessContrast(brightness_limit=0.06, contrast_limit=0.3, p=0.6),
                                         A.ShiftScaleRotate(shift_limit=0.03, scale_limit=0.05, rotate_limit=2, p=0.6),
                                         A.RandomGamma(p=0.6),
                                         A.Perspective(scale=(0.012, 0.012), p=0.6)
                                         ])

    def __len__(self):
        return len(self.df_dataset)

    def __getitem__(self, idx):
        img_path = os.path.join(self.dataset_path, self.df_dataset.iloc[idx]["PATH"])
        label = self.df_dataset.iloc[idx]["LABEL"]

        if not os.path.exists(img_path):
            print("The image '{}' does not exist ".format(img_path))

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.is_train:
            img = self.transformation(image=img)['image']

        if self.do_resize:
            img = cv2.resize(img, (self.cfg["data"]["size"], self.cfg["data"]["size"]))
        img = transforms.ToTensor()(img)
        if self.normalize is not None:
            img = self.normalize(img)

        return img, label


def create_loaders(df_dataset_train, df_dataset_val, df_dataset_test, cfg, dataset_path, batch_size):

    # 1 - istanzio la classe dataset di train, val e test
    classification_dataset_train = Dataset(df_dataset=df_dataset_train,
                                           cfg=cfg,
                                           dataset_path=dataset_path,
                                           is_train=True)
    classification_dataset_val = Dataset(df_dataset=df_dataset_val,
                                         cfg=cfg,
                                         dataset_path=dataset_path)
    classification_dataset_test = None
    if df_dataset_test is not None:
        classification_dataset_test = Dataset(df_dataset=df_dataset_test,
                                              cfg=cfg,
                                              dataset_path=dataset_path)


    # 2 - istanzio i dataloader
    classification_dataloader_train = DataLoader(dataset=classification_dataset_train,
                                                 batch_size=batch_size,
                                                 shuffle=True,
                                                 num_workers=0,
                                                 drop_last=False)

    classification_dataloader_val = DataLoader(dataset=classification_dataset_val,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=0,
                                               drop_last=False)

    classification_dataloader_test = None
    if classification_dataset_test is not None:
        classification_dataloader_test = DataLoader(dataset=classification_dataset_test,
                                                    batch_size=batch_size,
                                                    shuffle=True,
                                                    num_workers=0,
                                                    drop_last=False)

    return classification_dataloader_train, classification_dataloader_val, classification_dataloader_test