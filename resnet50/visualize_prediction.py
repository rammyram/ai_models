import os
import json
import traceback
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import torch
from torch import cuda
import torch.nn as nn
import torchvision.transforms.functional as F
from model import load_trained_resnet50
import seaborn as sb
import random

def np_sigmoid():
    return lambda x: 100/(1 + np.exp(-x))


def get_show_classes_and_percentage(predict, idx_to_class, bool_show_classes):
    if np.sum(bool_show_classes) == 0:
        return [], []
    idx_show_classes = np.argwhere(bool_show_classes).flatten()
    show_classes = predict[idx_show_classes]
    idx_sort_show_classes = idx_show_classes[np.argsort(-show_classes)]
    sort_show_classes = [idx_to_class[str(i)] for i in idx_sort_show_classes]
    sort_show_classes_value = -np.sort(-show_classes)
    sort_show_classes_value = np_sigmoid()(sort_show_classes_value).tolist()

    return sort_show_classes, sort_show_classes_value


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
                occ_pixel = random.randint(0, 255)

            h_start = h * occ_stride
            w_start = w * occ_stride
            h_end = min(height, h_start + occ_size)
            w_end = min(width, w_start + occ_size)

            if (w_end) >= width or (h_end) >= height:
                continue
            input_image = np.copy( image )

            # replacing all the pixel information in the image with occ_pixel(grey) in the specified location
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


def visualize(config):
    result_folder = config['result_folder']
    predict_result_dir = os.path.join(result_folder, config['predict_result_file_name'])
    idx_to_class_dir = os.path.join(result_folder, config['idx_to_class_name'])
    save_result_folder = os.path.join(result_folder, config['save_result_image_folder_name'])
    if not os.path.exists(save_result_folder):
        os.makedirs(save_result_folder)
    print(f'Visualize result from {predict_result_dir}')

    # get idx_to_class
    with open(idx_to_class_dir, 'r') as fp:
        idx_to_class = json.load(fp)

    count_0 = 0
    count_3 = 0
    threshold = config['threshold']

    # Read test result
    predict_results = pd.read_csv(predict_result_dir, index_col=False).to_dict('records')

    # Loading model : required for heatmap calculation
    model, _ = load_trained_resnet50(config['trained_model_dir'])
    model.eval()
    torch.set_grad_enabled(False)

    # Visualize bad images and save to file
    for result in predict_results:
        try:
            image_path = result['image_path']
            image_group = os.path.basename(os.path.dirname(image_path))
            if not result['predict_base_0']:
                count_0 += 1

            # Only get wrong images based on predict_base_3
            if not result['predict_base_3']:
                count_3 += 1
                if count_3 % 100 == 0:
                    print(f'count_3: {count_3}')

                # Cast str -> list
                predict = np.array(json.loads(result['predict']))
                target = np.array(json.loads(result['target']))

                # Get file name
                file_name, _ = os.path.splitext(os.path.split(result['image_path'])[-1])

                # True positive + False negative
                bool_truth_classes = target == 1
                bool_good_truth_classes = (predict > -threshold) & bool_truth_classes
                sort_truth_class, sort_truth_class_value =\
                    get_show_classes_and_percentage(predict, idx_to_class, bool_good_truth_classes)

                # False positive
                bool_not_classes = ~bool_truth_classes
                bool_bad_not_class = (predict > -threshold) & bool_not_classes
                sort_bad_not_class, sort_bad_not_class_value =\
                    get_show_classes_and_percentage(predict, idx_to_class, bool_bad_not_class)

                sort_show_class = sort_truth_class + sort_bad_not_class
                sort_show_class_value = sort_truth_class_value + sort_bad_not_class_value

                # calc heatmap: todo: heatmap considers single true label per image
                target_cls_idx = np.argmax(target)
                target_class_name = idx_to_class[str(target_cls_idx)]
                base_image = cv2.imread(image_path)
                base_image = cv2.cvtColor(base_image, cv2.COLOR_BGR2RGB)
                base_image = cv2.resize(base_image, (224, 224))
                heatmap, image = heatmap_per_image(model, base_image, target_cls_idx, occ_size=55, occ_stride=55)

                # Draw image with graph one the right
                fig, ax = plt.subplots(1, 3, figsize=(20, 8),
                                       gridspec_kw={'width_ratios': [2, 2,1]})
                # Base image
                ax[0].set_xticks(np.arange(0, 225, 56))
                ax[0].set_yticks(np.arange(0, 225, 56))
                ax[0].imshow(base_image)
                ax[0].grid()

                # Heat Map
                df_cfm = pd.DataFrame(heatmap.cpu().numpy())
                sb.heatmap(df_cfm, vmax=1, vmin=0, ax=ax[1], xticklabels=False, yticklabels=False, annot=True)
                ax[1].set_title(target_class_name)

                # Bar chart
                bar_colors = ['g'] * len(sort_truth_class) + ['r'] * len(sort_bad_not_class)
                bars = ax[2].bar(sort_show_class, sort_show_class_value,
                                 width=0.4, color=bar_colors)
                ax[2].tick_params(axis='x', labelrotation=90)
                for bar in bars:
                    yval = bar.get_height()
                    plt.text(bar.get_x(), yval + .5, str(np.round(yval, 2)) + " %")

                ax[2].set_ylim([0, 105])
                ax[2].set(ylabel="%")

                plt.title(image_group)
                plt.tight_layout()

                # Save image
                # plt.show()
                plt.savefig(os.path.join(save_result_folder, file_name+'.png'))
                plt.close('all')

                # print(result['image_path'])
                # if count_3 >= 5:
                #     break

        except KeyError:
            traceback.print_exc()
            continue
    print(f'Final:')
    print(f'Count_0: {count_0}')
    print(f'Count_3: {count_3}')
    # print(predict_results[0])
    # print(type(predict_results[0]['predict_base_0']))
    # font = cv2.FONT_HERSHEY_SIMPLEX
    # org = (20, 30)
    # fontScale = 0.6
    # color = (255, 0, 0)
    # thickness = 2
    # img = cv2.putText(img, "".join(class_names), org, font,
    #                   fontScale, color, thickness, cv2.LINE_AA)
    # cv2.imwrite(result_image_path, img)


if __name__ == '__main__':
    basic_config = {
        'trained_model_dir': '/home/sohoa1/rammy/main_projects/resnet50/mov/final_model.pt',
        'result_folder': '/home/sohoa1/rammy/main_projects/resnet50/mov',
        'predict_result_file_name': 'result.csv',
        'idx_to_class_name': 'id_to_class.json',
        'save_result_image_folder_name': 'visualize_new',
        'threshold': 0,
    }
    visualize(basic_config)
