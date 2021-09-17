import os
from data import create_dataset, get_image_albumentations
from model import get_resnet50_model, save_final_model, load_trained_resnet50
from multilabel import create_loss_and_optimizer, model_train


def train(config):

    # Location of data
    images_root = config['images_root']
    path = config["weight"]
    save_model_folder = config['save_model_folder']
    if not os.path.exists(save_model_folder):
        os.makedirs(save_model_folder)
    log_dir = os.path.join(save_model_folder, config['log_name'])
    save_model_dir = os.path.join(save_model_folder, config['save_model_dir'])
    checkpoint_dir = os.path.join(save_model_folder, config['checkpoint_dir'])

    batch_size = config['batch_size']

    image_transforms = get_image_albumentations()

    data, data_loaders = create_dataset(images_path=images_root, batch_size=batch_size,
                                        image_transforms=image_transforms, val_split=0.2)

    n_classes = len(data['train'].classes)

    if path is None :
        # Load imageNet pretrained model
        model = get_resnet50_model(n_classes)
    else:
        # To load trained model use:
        model, optimizer = load_trained_resnet50(path)

    # Add class info to model
    model.class_to_idx = data['train'].class_to_idx
    model.idx_to_class = {
        idx: class_
        for class_, idx in model.class_to_idx.items()
    }

    criterion, optimizer, lr_scheduler = \
        create_loss_and_optimizer(model, lr=config['learning_rate'],
                                  lr_step=config['lr_step'],
                                  lr_decrease=config['lr_decrease'])

    # Set model to train mode
    model.train()

    model, history = model_train(
        model,
        criterion,
        optimizer,
        lr_scheduler,
        data_loaders['train'],
        data_loaders['val'],
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir,
        early_stop_epoch_limit=config['early_stop_epoch_limit'],
        n_epochs=config['n_epochs'],
        weight_decay=config['weight_decay'],
        print_every=1)


    with open('history.txt', 'w') as f:
        for item in history:
            f.write(f"{item}\n")
    save_final_model(model, save_model_dir)


if __name__ == '__main__':
    basic_config = {
        "weight" : None,
        'images_root': '/home/sohoa1/rammy/datasets/2021/object_detection/fire_extinguisher/resnet_data_7_6_2021/fake_plus_original_train',
        'save_model_folder': '/home/sohoa1/rammy/main_projects/resnet50/resnet_data_7_6_2021_v2',
        'save_model_dir': 'final_model.pt',
        'checkpoint_dir': 'checkpoint.pt',
        'batch_size': 16,
        'learning_rate': 1e-4,
        'n_epochs': 200,
        'early_stop_epoch_limit': 15,
        'log_name': 'log.csv',
        'lr_decrease': 0.1,
        'lr_step': 12,
        'weight_decay': 0.3,
    }

    train(basic_config)


