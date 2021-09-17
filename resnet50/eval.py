import os
from model import load_trained_resnet50
from data import get_image_augmentation, create_valid_dataset
from multilabel import valid_mass_predict
import json

import torch
import torch.nn as  nn
import numpy as np
from torch import cuda
import pandas as pd
import  torchvision.transforms.functional as F
import matplotlib.pyplot as plt
import seaborn as sb
import cv2
import random


def run_valid_predict(config):

    # Location of data
    images_root = config['images_root']

    save_result_folder = config['save_result_folder']
    if not os.path.exists(save_result_folder):
        os.makedirs(save_result_folder)

    log_dir = os.path.join(save_result_folder, config['predict_result_file_name'])
    idx_to_class_dir = os.path.join(save_result_folder, config['idx_to_class_name'])

    batch_size = config['batch_size']

    # Load trained model
    model, _ = load_trained_resnet50(config['trained_model_dir'])

    idx_to_class = model.idx_to_class
    with open(idx_to_class_dir, 'w') as fp:
        json.dump(idx_to_class, fp)

    # Get test set
    image_transforms = get_image_augmentation()
    image_transforms = get_image_augmentation()
    data, data_loaders = create_valid_dataset(images_path=images_root, batch_size=batch_size,
                                              image_transforms=image_transforms,
                                              class_to_idx=model.class_to_idx)

    # Predict
    valid_mass_predict(model, test_loader=data_loaders['val'], log_dir=log_dir)


def visualize_model_heatmap(config):
    # Location of data
    train_on_gpu = torch.cuda.is_available()
    images_root = config['images_root']
    save_result_folder = config['save_result_folder']
    if not os.path.exists(save_result_folder):
        os.makedirs(save_result_folder)
    heatmap_dir = os.path.join(save_result_folder, "heatmaps")
    os.makedirs(heatmap_dir, exist_ok=True)

    # Load trained model
    model, _ = load_trained_resnet50(config['trained_model_dir'])
    model.eval()
    torch.set_grad_enabled(False)
    class_to_idx = model.class_to_idx

    # Get test set
    for class_name in os.listdir( images_root ):

        class_idx = class_to_idx[class_name]
        class_path = os.path.join( images_root, class_name )
        heatmap_dir_class = os.path.join(heatmap_dir, class_name)
        os.makedirs( heatmap_dir_class, exist_ok=True )

        for i, image_name in enumerate(os.listdir(class_path)):
            if class_name != "outside" or image_name!="mpa2_21_170.jpg": continue
            image_path = os.path.join( class_path, image_name )
            image      = cv2.imread( image_path )
            image      = cv2.cvtColor( image, cv2.COLOR_BGR2RGB )
            image      = cv2.resize( image, (224,224) )

            # inference on entire image
            input_image = F.to_tensor(input_image)
            input_image = F.normalize(input_image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            if train_on_gpu: input_image = input_image.unsqueeze(0).cuda()
            predictions = model(input_image)

            # get heatmap
            heatmap, image = heatmap_per_image(model, image, class_idx, occ_size=55, occ_stride=55)

            # set plot
            fig, ax = plt.subplots(1, 3, figsize=(14,6) )

            # Base image
            ax[0].set_xticks(np.arange(0,225,56))
            ax[0].set_yticks(np.arange(0,225,56))
            ax[0].imshow(image)
            ax[0].grid()

            # heatmap plotting
            df_cfm = pd.DataFrame(heatmap.cpu().numpy())
            sb.heatmap(df_cfm, vmax=1, vmin=0, ax = ax[1], xticklabels=False, yticklabels=False,  annot=True )

            # Save plot
            plt.title(class_name)
            plt.tight_layout()
            # plt.show()
            plt.savefig(os.path.join(heatmap_dir_class, image_name))
            plt.close('all')

            # get heatmaps for max 10 images per class
            if i == 0: break


def heatmap_per_image(model, image, label, occ_size=None, occ_stride=None, occ_pixel=None):
    # Check if gpu is available
    train_on_gpu = cuda.is_available()

    # get the width and height of the image
    height, width = image.shape[0], image.shape[1]

    # Set occ_size and occ_stride w.r.t image size
    if occ_size is None:
        occ_size = width // 3
    if occ_stride is None:
        occ_stride = occ_size//2

    # setting the output image width and height
    output_height = int(np.ceil((height - occ_size) / occ_stride))
    output_width = int(np.ceil((width - occ_size) / occ_stride))

    # create a white image of sizes we defined
    heatmap = torch.zeros((output_height, output_width))

    # iterate all the pixels in each column
    for h in range(0, height):
        for w in range(0, width):

            # set occlusion pixel
            if occ_pixel is None:
                occ_pixel = random.randint(0,255)

            h_start = h * occ_stride
            w_start = w * occ_stride
            h_end = min(height, h_start + occ_size)
            w_end = min(width, w_start + occ_size)
            if (w_end) >= width or (h_end) >= height:
                continue

            # replacing all the pixel information in the image with occ_pixel(grey) in the specified location
            input_image = np.copy(image)
            input_image[h_start:h_end, w_start:w_end, :] = occ_pixel

            # convert input to tensor and normalize
            input_image = F.to_tensor(input_image)
            input_image = F.normalize(input_image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            if train_on_gpu: input_image = input_image.unsqueeze(0).cuda()

            # run inference on modified image
            output = model(input_image)
            output = nn.functional.softmax(output, dim=1)
            prob   = output.tolist()[0][ label ]

            # setting the heatmap location to probability value
            heatmap[h, w] = prob

    return heatmap, image


if __name__ == "__main__":
    basic_config_valid = {
        'trained_model_dir': '/home/sohoa1/rammy/main_projects/resnet50/resnet_data_7_6_2021_v2/checkpoint.pt',
        'images_root': '/home/sohoa1/rammy/datasets/2021/object_detection/fire_extinguisher/resnet_data_7_6_2021/fake_plus_original_test',
        'save_result_folder': '/home/sohoa1/rammy/main_projects/resnet50/resnet_data_7_6_2021_v2/results',
        'batch_size': 16,
        'predict_result_file_name': 'result.csv',
        'idx_to_class_name': 'id_to_class.json',
    }

    run_valid_predict(basic_config_valid)
    visualize_model_heatmap(basic_config_valid)