import time
import os
import torch
from torch import cuda
import torch.nn as nn
import torchvision
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def get_maskrcnn_model(num_classes: int, pretrained_weights: str):
    """
    Load maskrcnn pretrained model
    :param num_classes:
    :param pretrained_weights:
    :return:
    """
    print("Loading MaskRcnn pretrained model")
    start_time = time.time()

    # load model
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=False )

    # Load pretrained state dict
    pretrained = torch.load(pretrained_weights)
    model.load_state_dict(pretrained)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256

    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)
    # Get model to use GPU
    model = apply_gpu(model)

    # for name, param in model.named_parameters():
    #     print(name, param.requires_grad)
    return model


def load_maskrcnn(model_path:str, device:str, score_threshold:float = 0.5):
    """Load a PyTorcmodel_pathh model checkpoint
`    """

    # Load in checkpoint
    start_time = time.time()
    model_name = os.path.basename(model_path)

    # Select Device
    if device == "cpu":
        train_on_gpu, multi_gpu = False, False
    else:
        train_on_gpu, multi_gpu = check_gpu_status()

    # Load Model
    if train_on_gpu:
        checkpoint = torch.load(model_path)
    else:
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

    model_state_dict = checkpoint["state_dict"]
    class_to_idx = checkpoint['class_to_idx']
    num_classes = len(class_to_idx) + 1
    print(f"Loading {model_name} Model on cuda {train_on_gpu}")


    # Load model module
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(
                                                                pretrained=False,
                                                                pretrained_backbone=False,
                                                                num_classes= num_classes,
                                                                box_score_thresh = score_threshold
                                                               )

    # disable training for 1st layer of resnet
    for param in model.backbone.body.layer1.parameters():
        param.requires_grad = False

    # Load Model dict
    model.load_state_dict(model_state_dict)

    # Move to gpu
    if multi_gpu:
        model = nn.DataParallel(model)
    if train_on_gpu:
        model = model.to('cuda')

    # Model basics
    model.class_to_idx = checkpoint['class_to_idx']
    model.idx_to_class = checkpoint['idx_to_class']
    model.epochs = checkpoint['epochs']

    # # Optimizer
    # model.optimizer = checkpoint['optimizer']
    # model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    model_load_time = time.time() - start_time
    print(f"{model_name} Load Time: %.2f" % model_load_time)
    print(f"Classes: {class_to_idx}")
    print(f"Trained Epoch: {model.epochs}")
    print(f"Score threshold: {score_threshold}")

    # for name, param in model.named_parameters():
    #     print(name, param.requires_grad)
    return model


def save_final_model(model, path: str):
    """Save a PyTorch model checkpoint

    Params
    --------
        model (PyTorch model): model to save
        path (str): location to save model. Must start with `model_name-` and end in '.pth'

    Returns
    --------
        None, save the `model` to `path`

    """

    _, multi_gpu = check_gpu_status()

    # Basic details
    checkpoint = {
        'class_to_idx': model.class_to_idx,
        'idx_to_class': model.idx_to_class,
        'epochs': model.epochs,
    }

    # Extract the the state dictionary
    if multi_gpu:
        checkpoint['state_dict'] = model.module.state_dict()
    else:
        checkpoint['state_dict'] = model.state_dict()

    # Add the optimizer
    checkpoint['optimizer'] = model.optimizer
    checkpoint['optimizer_state_dict'] = model.optimizer.state_dict()

    # Save the data to the path
    torch.save(checkpoint, path)


def check_gpu_status():
    # Whether to train on a gpu
    train_on_gpu = cuda.is_available()

    # Number of gpus
    multi_gpu = False
    if train_on_gpu:
        gpu_count = cuda.device_count()
        # print(f'{gpu_count} gpus detected.')
        if gpu_count > 1:
            multi_gpu = True

    return train_on_gpu, multi_gpu


def apply_gpu(model):
    train_on_gpu, multi_gpu = check_gpu_status()
    # Move to gpu and parallelize
    if train_on_gpu:
        model = model.to('cuda')

    if multi_gpu:
        model = nn.DataParallel(model)
    return model
