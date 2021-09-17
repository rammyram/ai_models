import os
import numpy as np
from engine import train_one_epoch, evaluate
import albumentations as A
import sys
from model import save_final_model, get_maskrcnn_model, load_maskrcnn
from data import DatasetFromCOCO, collate_fn
from helpers import Logger
import shutil
import torch


def get_transform(aug_type: str) -> A.Compose:
    """
    Compose transforms defined in the config file
    :param aug_type:
    :return:
    """
    aug = []
    if aug_type == "train":
        aug = cfg.train_aug
    elif aug_type == "test":
        aug = cfg.test_aug

    transforms = A.Compose(aug, bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))
    return transforms


def train():

    # Set log
    sys.stdout = Logger(cfg.log_path)

    # Using DatasetFromCOCO from loading data
    dataset_train = DatasetFromCOCO(images_dir=cfg.images_dir, json_file=cfg.json_file_train,
                                    transforms=get_transform(aug_type=cfg.train_aug))
    dataset_val = DatasetFromCOCO(images_dir=cfg.images_dir, json_file=cfg.json_file_val,
                                  transforms=get_transform(cfg.test_aug))

    # define training and validation data loaders
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers,
        collate_fn=collate_fn)

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers,
        collate_fn=collate_fn)

    # Loading model
    if cfg.model_checkpoint is None:
        if cfg.pre_trained_weights:
            model = get_maskrcnn_model(cfg.num_classes, cfg.pre_trained_weights)
        else:
            raise Exception("Provide either model_checkpoint or pre_trained_weights paths in config")
    else:
        model = load_maskrcnn(cfg.model_checkpoint, cfg.device)

    # set model params
    if not hasattr(model, "idx_to_class"):
        model.idx_to_class = dataset_train.idx_to_class
    if not hasattr(model, "class_to_idx"):
        model.class_to_idx = {class_name: idx for idx, class_name in model.idx_to_class.items()}
    if not hasattr(model, "epochs"):
        model.epochs = 0

    # construct an optimizer
    if not hasattr(model, "optimizer"):
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=cfg.lr,
                                    momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    else:
        print("Loading optimizer from checkpoint")
        optimizer = model.optimizer

    # Move model to the right device
    model.to(cfg.device)

    # Learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=cfg.scheduler_step_size,
                                                   gamma=cfg.scheduler_gamma)

    # Start training
    best_map = -np.inf
    no_improvement = 0
    for epoch in range(model.epochs + 1, cfg.num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader_train, cfg.device, epoch, print_freq=10)

        # update the learning rate
        lr_scheduler.step()

        # evaluate on the test dataset
        coco_evaluator, current_stats = evaluate(model, data_loader_val, device=cfg.device)
        print("=" * 10)
        print(f"MAP at epoch [{epoch}/{cfg.num_epochs}] : {current_stats} ")

        # set model checkpoint params
        model.epochs += 1
        model.optimizer = optimizer

        # Save best model
        current_map = sum(current_stats.values()) / len(current_stats)
        if best_map < current_map:
            no_improvement = 0
            best_map = current_map
            print(f"Model improved at epoch : [{epoch}/{cfg.num_epochs}] , saving best model...")
            save_final_model(model, cfg.best_model_path)
        else:
            no_improvement += 1
            save_final_model(model, cfg.last_model_path)
            if no_improvement >= cfg.early_stop_limit:
                print(f"Early stopping, no improvement for {no_improvement} epochs")
                break
            else:
                print(f"Model has not improved for {no_improvement} epochs")

            print("=" * 10)

    print("Done -------------------")


if __name__ == "__main__":
    import default_config as cfg

    # Create project folder
    os.makedirs(cfg.project, exist_ok=True)

    # Copy config to project folder
    shutil.copy("default_config.py", f"{cfg.project}")

    # Train
    train()
