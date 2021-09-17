from torchvision import models
import torch
from torch import cuda
import torch.nn as nn
from pathlib import Path
import os


def check_gpu_status():
    # Whether to train on a gpu
    train_on_gpu = cuda.is_available()
    # print(f'Train on gpu: {train_on_gpu}')

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


def get_resnet50_model(n_classes: int, freeze_extractor: bool = False):
    # Load base model
    model = models.resnet50(pretrained=False)

    # Load state dict
    path = os.path.join(str(Path(__file__).parent.resolve()), 'pretrained/resnet50-19c8e357.pth')
    pretrained = torch.load(path)
    model.load_state_dict(pretrained)

    # Freeze parameters
    if freeze_extractor:
        for param in model.parameters():
            param.requires_grad = False
    else:
        for param in model.parameters():
            param.requires_grad = True

    # Add classification layers
    n_inputs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(n_inputs, 256), nn.ReLU(), nn.Dropout(0.2),
        # nn.Linear(256, 64),  nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(256, n_classes)
        # , nn.Sigmoid()
    )
    # Get model to use GPU
    model = apply_gpu(model)

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

    # Extract the final classifier and the state dictionary
    if multi_gpu:
        checkpoint['fc'] = model.module.fc
        checkpoint['state_dict'] = model.module.state_dict()
    else:
        checkpoint['fc'] = model.fc
        checkpoint['state_dict'] = model.state_dict()

    # Add the optimizer
    checkpoint['optimizer'] = model.optimizer
    checkpoint['optimizer_state_dict'] = model.optimizer.state_dict()

    # Save the data to the path
    torch.save(checkpoint, path)


def load_trained_resnet50(path: str, freeze_extractor: bool = False):
    """Load a PyTorch model checkpoint
`    """

    train_on_gpu, multi_gpu = check_gpu_status()

    # Load in checkpoint
    checkpoint = torch.load(path)

    model = models.resnet50(pretrained=False)
    # Make sure to set parameters as not trainable
    if freeze_extractor:
        for param in model.parameters():
            param.requires_grad = False

    # Get the final classifier and the state dictionary
    model.fc = checkpoint['fc']

    # Load in the state dict
    model.load_state_dict(checkpoint['state_dict'])

    # Move to gpu
    if multi_gpu:
        model = nn.DataParallel(model)

    if train_on_gpu:
        model = model.to('cuda')

    # Model basics
    model.class_to_idx = checkpoint['class_to_idx']
    model.idx_to_class = checkpoint['idx_to_class']
    model.epochs = checkpoint['epochs']

    # Optimizer
    optimizer = checkpoint['optimizer']
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return model, optimizer
