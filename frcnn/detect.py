import os, shutil
import time
import torch
import csv
from torchvision.transforms import functional as F
import tqdm
from torch.utils.data import DataLoader
from mask_rcnn.helpers import visualize_detections
import matplotlib.pyplot as plt
from helpers import objdict
from data import LoadImagesDataset, collate_fn
from model import load_frcnn


def frcnn_predict(model, device: str, data_loader)-> list :
    """
    :param model: <torch loaded model>
    :param device: <str>
    :param data_loader: <torch dataloader>
    :yield: list of detections per image

                                        each detection<dict> has keys boxes<tensor>,
                                                                 labels<tensor>,
                                                                 scores<tensor>,
                                                                 image_name<str>,
                                                                 image<numpy original image>
                                   }
    """
    # Select Device
    if device != "cpu":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    model = model.to(device)
    model.eval()

    # perform inference
    with torch.no_grad():
        for index,(images, file_names) in enumerate(data_loader):

            # Convert to tensor and move to device
            images_original = images.copy()
            images = list(F.to_tensor(image).to(device) for image in images)

            # infer : outputs is list of dict per image, dict has keys boxes,labels,scores
            outputs = model(images)
            outputs = [{k: v.cpu() for k, v in t.items()} for t in outputs]

            # set image name
            for i,name in enumerate(file_names):
                outputs[i]["image_name"] = name
                outputs[i]["image"] = images_original[i]

            yield outputs


def inference(cfg):

    # initialize var
    start_time = time.time()
    weights, device, score_thres, output, source, batch_size, num_workers, \
    enable_visualize, enable_log, save_boxes =  cfg.weights, cfg.device, cfg.score_thres, \
                                                    cfg.output, cfg.source, cfg.batch_size, cfg.num_workers, \
                                                    cfg.enable_visualize, cfg.enable_log, cfg.save_boxes
    images_with_detections = []
    images_with_no_detections = []


    # set output directories
    if enable_log or save_boxes or enable_visualize:
        assert isinstance(output, str), "Output must be provided  visualize is enabled"
        os.makedirs(output, exist_ok=True)

        if save_boxes:
            extracted_boxes_dir = os.path.join(output, "extracted_boxes")
            if os.path.exists(extracted_boxes_dir): shutil.rmtree(extracted_boxes_dir)
            os.makedirs(extracted_boxes_dir)
        if enable_visualize:
            visualize_dir = os.path.join(output, "visualize")
            if os.path.exists(visualize_dir): shutil.rmtree(visualize_dir)
            os.makedirs(visualize_dir, exist_ok=False)
        if enable_log:
            log_file = os.path.join(output, "log.csv")
            log_fieldnames = ["filename", "score", "box", "cls"]
            with open(log_file, 'w', newline='') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=log_fieldnames)
                writer.writeheader()

        no_detections_dir = os.path.join(output, "no_detection_images")
        if os.path.exists(no_detections_dir): shutil.rmtree(no_detections_dir)
        os.makedirs(no_detections_dir, exist_ok=False)

    # load model
    model = load_frcnn(weights, device, score_threshold=score_thres)
    if hasattr(model, "idx_to_class"):
        idx_to_class = model.idx_to_class
    else:
        idx_to_class = None
    class_name = lambda idx: idx_to_class[idx] if idx_to_class else idx

    # Prediction code accepts list of images so change source to list if dir is provided
    if source and isinstance(source, str) and os.path.isdir(source):
        source = [os.path.join(source, f) for f in os.listdir(source) if os.path.isfile(os.path.join(source, f))]


    # create dataset and data loader
    dataset = LoadImagesDataset(source)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True,
                             collate_fn=collate_fn)

    # create inference generator
    mrcnn_predictor = frcnn_predict(model, device, data_loader)

    # do inference per batch
    for detections_per_batch in tqdm.tqdm(mrcnn_predictor, desc=f"Batch inferencing: ", unit="Batch", total=len(data_loader)):
        for detections_per_image in detections_per_batch:

            image_name = detections_per_image["image_name"]
            image = detections_per_image["image"]
            boxes = detections_per_image["boxes"].tolist()
            scores = detections_per_image["scores"].tolist()
            labels = detections_per_image["labels"].tolist()

            atleast_one_detection = 1 if len(boxes) else 0

            if atleast_one_detection:
                images_with_detections.append(image_name)
                scores = [int(score * 100) for score in scores ]

                if save_boxes:
                    # create per class directories to save extracts
                    [os.makedirs(os.path.join(extracted_boxes_dir, class_name(label)), exist_ok=True)
                     for label in labels]
                    extracts = [ image[ int(ymin):int(ymax), int(xmin):int(xmax),: ] for xmin,ymin,xmax,ymax in boxes ]
                    for extract, label, score in zip(extracts, labels, scores):
                        new_image_name = f"{score}_{image_name}"
                        plt.imsave(os.path.join(extracted_boxes_dir, class_name(label), new_image_name), extract)

                if enable_log:

                    with open(log_file, 'a', newline='') as csv_file:
                        writer = csv.DictWriter(csv_file, fieldnames=log_fieldnames)
                        [writer.writerow({
                            "filename": image_name,
                            "score": round(score,2 ),
                            "box": [int(x) for x in box],
                            "cls": label
                        }) for score,box,label in zip( scores, boxes,labels )  ]
                        images_with_detections.append(image_name)

                if enable_visualize:
                    result_path = os.path.join(visualize_dir, new_image_name)
                    visualize_detections(image, scores, boxes, labels, result_path)

            else:
                plt.imsave(os.path.join(no_detections_dir, image_name), image)
                images_with_no_detections.append(image_name)

    # print stats
    print(f"Time taken:\t {round(time.time() - start_time, 2)} \n"
          f"Total images:\t{len(source)} \n"          
          f"Detections:\t{len(images_with_detections)} \n"
          f"Image names with no detections:\t\n{images_with_no_detections} \n"
          )


if __name__ == "__main__":

    cfg = {

        "weights": "/home/sohoa1/rammy/main_projects/faster_rcnn/fire_extinguisher/fire_extinguisher_best.pt",

        # images directory
        "source": "/home/sohoa1/rammy/datasets/2021/object_detection/fire_extinguisher/frcnn/all_images",

        # directory to save results
        "output": "/home/sohoa1/rammy/datasets/2021/object_detection/fire_extinguisher/frcnn/visualize_results_all_images",

        # enable saving log files of detections
        "enable_log": False,

        # save images with detection drawn on image
        "enable_visualize": True,

        # Exract each detection and save under class folder
        "save_boxes": True,

        "device": "cuda",
        "batch_size": 5,
        "num_workers": 0,
        "score_thres": 0.5,
    }

    cfg = objdict(cfg)
    inference(cfg)