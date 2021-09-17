import tqdm
import os, shutil
import json
import csv
from data_mov import create_dataset
from model import load_trained_resnet50
import torch
from torch import cuda
import numpy as np
from timeit import default_timer as timer
from torchvision import  transforms
import pathlib

def mass_predict(model, test_loader, log_dir):
    # Check if gpu is available
    train_on_gpu = cuda.is_available()

    overall_start = timer()
    response_percent = []
    response_label_idx = []

    # Batches loop
    model.eval()
    # Don't need to keep track of gradients
    with torch.no_grad():
        for image_arr in tqdm.tqdm(test_loader):
            if train_on_gpu:
                image_arr = image_arr.cuda()

            output = model(image_arr)
            # Get result for response
            copy_output = output.clone()
            copy_output = torch.nn.Sigmoid()(copy_output)
            temp_response_percent, temp_response_label_idx = torch.max(copy_output, dim=1)

            # yield temp_response_percent.cpu().detach().tolist(), temp_response_label_idx.cpu().detach().tolist()
            response_percent.extend(temp_response_percent.cpu().detach().tolist())
            response_label_idx.extend(temp_response_label_idx.cpu().detach().tolist())

    total_time = timer() - overall_start
    return response_percent, response_label_idx, total_time

def resNet50_predict(config):
    # Input images
    input_images = config["input_images"]
    batch_size = config['batch_size']
    score_threshold = 0.8

    # Location to save results
    save_result_folder = config['save_result_folder']
    if not os.path.exists(save_result_folder):
        os.makedirs(save_result_folder)
    else:
        raise Exception("Save result folder already exist")

    # log file save
    log_dir = os.path.join(save_result_folder, config['predict_result_file_name'])

    # Load trained model
    model, _ = load_trained_resnet50(config['trained_model_dir'])

    # class dir where results are saved
    class_names = list(model.class_to_idx.keys())
    for cname in class_names:
        os.makedirs(os.path.join(save_result_folder,cname))

    # Below threshold class dir
    # below_threshold_dir_path = os.path.join(save_result_folder,
    #                                         f"images_below_score_threshold_{ int(score_threshold*100) }")
    # os.makedirs( below_threshold_dir_path )
    # class_names = list(model.class_to_idx.keys())
    # for cname in class_names:
    #     os.makedirs(os.path.join(below_threshold_dir_path,  cname))

    # Get test set
    image_transforms = transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    data, data_loader =create_dataset(input_images, image_transforms,batch_size = batch_size)

    # Predict
    response_percent, response_label_idx, total_time = mass_predict(model, test_loader=data_loader, log_dir=log_dir)

    # Save the image to result folder:
    for score,idx, source_img_path in zip(response_percent,response_label_idx, input_images):
        cname = model.idx_to_class[idx]
        image_name = os.path.basename(source_img_path)
        dst_image_path = os.path.join(save_result_folder, cname,
                                      f"{int(score * 100)}_{image_name}")
        shutil.copyfile(source_img_path, dst_image_path)

        # if score> score_threshold:
        #     dst_image_path = os.path.join(save_result_folder, cname, image_name)
        #     shutil.copyfile(source_img_path, dst_image_path)
        # else:
        #     dst_image_path = os.path.join(below_threshold_dir_path, cname,
        #                                   f"{int(score * 100)}_{image_name}")
        #     shutil.copyfile(source_img_path,dst_image_path)

    # print(response_percent, response_label_idx, total_time)


if __name__ == '__main__':
    image_folder = '/home/sohoa1/rammy/datasets/2021/object_detection/fire_extinguisher/frcnn/visualize_results_all_images/extracted_boxes/meter'
    basic_config = {
        "input_images": [ os.path.join(image_folder,f) for f in  os.listdir(image_folder)],
        'trained_model_dir': '/home/sohoa1/rammy/main_projects/resnet50/resnet_data_7_6_2021_v2/checkpoint.pt',
        'save_result_folder': '/home/sohoa1/rammy/datasets/2021/object_detection/fire_extinguisher/frcnn/resnet_results',
        'batch_size': 5,
        'predict_result_file_name': 'result.csv',
    }

    resNet50_predict(basic_config)



