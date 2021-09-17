import os
from engine import evaluate
import albumentations as A
import sys
from model import load_frcnn
from data import DatasetFromCOCO, collate_fn
import torch
from helpers import Logger


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

    transforms = A.Compose(aug)
    return transforms


def eval():

    # Set log
    sys.stdout = Logger(cfg.eval_log_path)

    # Using DatasetFromCOCO from loading data
    dataset_eval = DatasetFromCOCO(images_dir=cfg.images_dir, json_file=cfg.json_eval_file,
                                    transforms=get_transform(aug_type=cfg.eval_aug))

    # define training and validation data loaders
    data_loader_eval = torch.utils.data.DataLoader(
        dataset_eval, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers,
        collate_fn=collate_fn)

    # Loading model
    if cfg.eval_weight is None:
        raise Exception("Provide model path to perform evaluation")
    else:
        model = load_frcnn(cfg.eval_weight,cfg.device)

    # eval
    torch.set_grad_enabled(False)
    model.eval()
    evaluate(model, data_loader_eval, device=cfg.device)

    print("Done -------------------")


if __name__ == "__main__":
    # import project configuration
    import default_config as cfg

    # Create project directory if not exist
    os.makedirs(cfg.project, exist_ok=True)

    # Run evaluation
    eval()
