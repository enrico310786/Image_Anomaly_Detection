import torch.nn as nn
from torchvision import models
from torchvision.models import EfficientNet_B0_Weights, EfficientNet_B1_Weights, EfficientNet_B2_Weights, \
    EfficientNet_B3_Weights, EfficientNet_B4_Weights, EfficientNet_B5_Weights, EfficientNet_B6_Weights, \
    EfficientNet_B7_Weights, ResNet18_Weights, ResNet34_Weights, ResNet50_Weights, ResNet101_Weights, ResNet152_Weights, \
    EfficientNet_V2_M_Weights, EfficientNet_V2_S_Weights, EfficientNet_V2_L_Weights
import os

class Encoder(nn.Module):
    def __init__(self,  init_dim, num_autoencoder_layers, dim_autoencoder_layers, dropout):
        super(Encoder, self).__init__()
        self.init_dim = init_dim
        self.num_autoencoder_layers = num_autoencoder_layers
        self.dim_autoencoder_layers = dim_autoencoder_layers
        self.layer_1 = nn.Linear(self.init_dim, dim_autoencoder_layers[0])
        self.layer_2 = nn.Linear(dim_autoencoder_layers[0], dim_autoencoder_layers[1])
        self.layer_3 = None
        if num_autoencoder_layers == 3:
            self.layer_3 = nn.Linear(dim_autoencoder_layers[1], dim_autoencoder_layers[2])
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.batchNorm1 = nn.BatchNorm1d(init_dim)

    def forward(self, x):
        x = self.batchNorm1(x)
        x = self.layer_1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer_2(x)
        x = self.relu(x)
        if self.num_autoencoder_layers == 3:
            x = self.dropout(x)
            x = self.layer_3(x)
            x = self.relu(x)
        return x


class Decoder(nn.Module):
    def __init__(self,  init_dim, num_autoencoder_layers, dim_autoencoder_layers, dropout):
        super(Decoder, self).__init__()
        self.init_dim = init_dim
        self.num_autoencoder_layers = num_autoencoder_layers
        self.dim_autoencoder_layers = dim_autoencoder_layers
        self.layer_3 = None
        if num_autoencoder_layers == 3:
            self.layer_3 = nn.Linear(dim_autoencoder_layers[2], dim_autoencoder_layers[1])
        self.layer_2 = nn.Linear(dim_autoencoder_layers[1], dim_autoencoder_layers[0])
        self.layer_1 = nn.Linear(dim_autoencoder_layers[0], self.init_dim)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        if self.num_autoencoder_layers == 3:
            x = self.layer_3(x)
            x = self.relu(x)
            x = self.dropout(x)
        x = self.layer_2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer_1(x)
        return x


class Autoencoder(nn.Module):

    def __init__(self, model_config):
        super().__init__()

        self.dropout = model_config["dropout"]
        self.init_dim = model_config['init_dim']
        self.freeze_layers = model_config.get("freeze_layers", 1.0) > 0.0
        dim_autoencoder_layers = model_config['dim_autoencoder_layers'].split(",")
        self.dim_autoencoder_layers = [int(i) for i in dim_autoencoder_layers]
        self.num_autoencoder_layers = len(self.dim_autoencoder_layers)
        self.image_model = self.get_model(model_config["image_model"])
        self.encoder = Encoder(self.init_dim, self.num_autoencoder_layers, self.dim_autoencoder_layers, self.dropout)
        self.decoder = Decoder(self.init_dim, self.num_autoencoder_layers, self.dim_autoencoder_layers, self.dropout)

    def forward(self, x):
        embedding = self.image_model(x)
        rec_embedding = self.decoder(self.encoder(embedding))
        return embedding, rec_embedding

    def freeze_layers_base_model(self, model):
        for name, param in model.named_parameters():
            param.requires_grad = False

    def get_model(self, name_pretrained_model):
        print("name_pretrained_model: ", name_pretrained_model)
        base_model = None

        if name_pretrained_model == 'resnet18':
            base_model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
            base_model.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
            base_model.fc = nn.Flatten() # 512
            if self.freeze_layers:
                print("Freeze layers of pretrained model")
                self.freeze_layers_base_model(base_model)

        elif name_pretrained_model == 'resnet34':
            base_model = models.resnet34(weights=ResNet34_Weights.DEFAULT)
            base_model.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
            base_model.fc = nn.Flatten() # 512
            if self.freeze_layers:
                print("Freeze layers of pretrained model")
                self.freeze_layers_base_model(base_model)

        elif name_pretrained_model == 'resnet50':
            base_model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
            base_model.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
            base_model.fc = nn.Flatten()  # 2048
            if self.freeze_layers:
                print("Freeze layers of pretrained model")
                self.freeze_layers_base_model(base_model)

        elif name_pretrained_model == 'resnet101':
            base_model = models.resnet101(weights=ResNet101_Weights.DEFAULT)
            base_model.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
            base_model.fc = nn.Flatten()  # 2048
            if self.freeze_layers:
                print("Freeze layers of pretrained model")
                self.freeze_layers_base_model(base_model)

        elif name_pretrained_model == 'resnet152':
            base_model = models.resnet152(weights=ResNet152_Weights.DEFAULT)
            base_model.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
            base_model.fc = nn.Flatten()  # 2048
            if self.freeze_layers:
                print("Freeze layers of pretrained model")
                self.freeze_layers_base_model(base_model)

        elif name_pretrained_model == 'efficientnet_b0':
            base_model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
            base_model.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
            base_model.classifier = nn.Flatten()  # 1280
            if self.freeze_layers:
                print("Freeze layers of pretrained model")
                self.freeze_layers_base_model(base_model)

        elif name_pretrained_model == 'efficientnet_b1':
            base_model = models.efficientnet_b1(weights=EfficientNet_B1_Weights.DEFAULT)
            base_model.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
            base_model.classifier = nn.Flatten()  # 1280
            if self.freeze_layers:
                print("Freeze layers of pretrained model")
                self.freeze_layers_base_model(base_model)

        elif name_pretrained_model == 'efficientnet_b2':
            base_model = models.efficientnet_b2(weights=EfficientNet_B2_Weights.DEFAULT)
            base_model.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
            base_model.classifier = nn.Flatten()  # 1408
            if self.freeze_layers:
                print("Freeze layers of pretrained model")
                self.freeze_layers_base_model(base_model)

        elif name_pretrained_model == 'efficientnet_b3':
            base_model = models.efficientnet_b3(weights=EfficientNet_B3_Weights.DEFAULT)
            base_model.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
            base_model.classifier = nn.Flatten()  # 1536
            if self.freeze_layers:
                print("Freeze layers of pretrained model")
                self.freeze_layers_base_model(base_model)

        elif name_pretrained_model == 'efficientnet_b4':
            base_model = models.efficientnet_b4(weights=EfficientNet_B4_Weights.DEFAULT)
            base_model.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
            base_model.classifier = nn.Flatten()  # 1792
            if self.freeze_layers:
                print("Freeze layers of pretrained model")
                self.freeze_layers_base_model(base_model)

        elif name_pretrained_model == 'efficientnet_b5':
            base_model = models.efficientnet_b5(weights=EfficientNet_B5_Weights.DEFAULT)
            base_model.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
            base_model.classifier = nn.Flatten()  # 2048
            if self.freeze_layers:
                print("Freeze layers of pretrained model")
                self.freeze_layers_base_model(base_model)

        elif name_pretrained_model == 'efficientnet_b6':
            base_model = models.efficientnet_b6(weights=EfficientNet_B6_Weights.DEFAULT)
            base_model.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
            base_model.classifier = nn.Flatten()  # 2304
            if self.freeze_layers:
                print("Freeze layers of pretrained model")
                self.freeze_layers_base_model(base_model)

        elif name_pretrained_model == 'efficientnet_b7':
            base_model = models.efficientnet_b7(weights=EfficientNet_B7_Weights.DEFAULT)
            base_model.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
            base_model.classifier = nn.Flatten()  # 2560
            if self.freeze_layers:
                print("Freeze layers of pretrained model")
                self.freeze_layers_base_model(base_model)

        elif name_pretrained_model == 'efficientnet_v2_m':
            base_model = models.efficientnet_v2_m(weights=EfficientNet_V2_M_Weights.DEFAULT)
            base_model.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
            base_model.classifier = nn.Flatten()  # 1280
            if self.freeze_layers:
                print("Freeze layers of pretrained model")
                self.freeze_layers_base_model(base_model)

        elif name_pretrained_model == 'efficientnet_v2_s':
            base_model = models.efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)
            base_model.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
            base_model.classifier = nn.Flatten()  # 1280
            if self.freeze_layers:
                print("Freeze layers of pretrained model")
                self.freeze_layers_base_model(base_model)

        elif name_pretrained_model == 'efficientnet_v2_l':
            base_model = models.efficientnet_v2_l(weights=EfficientNet_V2_L_Weights.DEFAULT)
            base_model.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
            base_model.classifier = nn.Flatten()  # 1280
            if self.freeze_layers:
                print("Freeze layers of pretrained model")
                self.freeze_layers_base_model(base_model)

        return base_model


def find_last_checkpoint_file(checkpoint_dir, use_best_checkpoint=False):
    '''
    Cerco nella directory checkpoint_dir il file .pth.
    Se use_best_checkpoint = True prendo il best checkpoint
    Se use_best_checkpoint = False prendo quello con l'epoca maggiore tra i checkpoint ordinari
    :param checkpoint_dir:
    :param use_best_checkpoint:
    :return:
    '''
    print("Cerco il file .pth in checkpoint_dir: {} ".format(checkpoint_dir))
    list_file_paths = []

    for file in os.listdir(checkpoint_dir):
        if file.endswith(".pth"):
            path_file = os.path.join(checkpoint_dir, file)
            list_file_paths.append(path_file)
            print("Find: {}".format(path_file))

    print("Number of files .pth: {}".format(int(len(list_file_paths))))
    path_checkpoint = None

    if len(list_file_paths) > 0:

        if use_best_checkpoint:
            if os.path.isfile(os.path.join(checkpoint_dir, 'best.pth')):
                path_checkpoint = os.path.join(checkpoint_dir, 'best.pth')
        else:
            if os.path.isfile(os.path.join(checkpoint_dir, 'latest.pth')):
                path_checkpoint = os.path.join(checkpoint_dir, 'latest.pth')

    return path_checkpoint