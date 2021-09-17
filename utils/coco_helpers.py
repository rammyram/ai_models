import yaml
import shutil
import json
import numpy as np
import os
import cv2
import random
import tqdm
from pycocotools.coco import COCO
from statistics import mean
import colorsys
import sys
from pathlib import Path
import sklearn
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

font_prop = fm.FontProperties(fname=r"draw_japanese\arial-unicode-ms.ttf")


class Logger:
    """
    Save all print messages as logs to a file
    """

    def __init__(self, filename):
        self.console = sys.stdout
        self.filename = filename

    def write(self, message):
        self.console.write(message)
        with open(self.filename, "a+") as f:
            f.write(message)

    def flush(self):
        self.console.flush()
        # self.file.flush()


def check_exist(path):
    """
    raise exception if path doesn't exist
    :param path: str
    :return:
    """
    assert os.path.exists(path), f"{path} does not exist"


def yolo_data_split(src_annotations: str,
                    source_img_dir: str,
                    dest_annotations_dir: str,
                    split_ratios=[0.7, 0.2, 0.1]):
    """
    Create directory format accepted by yolo to perform training
    Split darknet annotations to train, test, val
    :param src_annotations:
    :param source_img_dir:
    :param dest_annotations_dir:
    :param split_ratios:
    :return:
    """
    if os.path.exists(dest_annotations_dir):
        shutil.rmtree(dest_annotations_dir)
    os.makedirs(dest_annotations_dir)

    split_ratios = [sum(split_ratios[:i + 1])
                    for i in range(len(split_ratios))]
    print("split_ratios: ", split_ratios)

    images_dir = os.path.join(dest_annotations_dir, "images")
    os.makedirs(images_dir)

    labels_dir = os.path.join(dest_annotations_dir, "labels")
    os.makedirs(labels_dir)

    os.makedirs(os.path.join(images_dir, "train"))
    os.makedirs(os.path.join(images_dir, "test"))
    os.makedirs(os.path.join(images_dir, "val"))

    os.makedirs(os.path.join(labels_dir, "train"))
    os.makedirs(os.path.join(labels_dir, "test"))
    os.makedirs(os.path.join(labels_dir, "val"))

    try:

        train_images = 0
        test_images = 0
        val_images = 0

        for file_name in tqdm.tqdm(os.listdir(src_annotations)):
            type = None
            image_name = [image_name for image_name in os.listdir(source_img_dir)
                          if Path(image_name).stem == Path(file_name).stem]

            if len(image_name):
                image_name = image_name[0]
                image_path = os.path.join(source_img_dir, image_name)
            else:
                print(f"Image not found, Skipping file {file_name}")
                continue

            temp = random.random()
            if temp <= split_ratios[0]:
                train_images += 1
                type = "train"
            else:
                if temp > split_ratios[0] and temp <= split_ratios[1]:
                    test_images += 1
                    type = "test"
                else:
                    val_images += 1
                    type = "val"

            # set paths
            dest_label_dir = os.path.join(labels_dir, type)
            dest_images_dir = os.path.join(images_dir, type)

            # Copy from source to darknet folders
            shutil.copyfile(
                os.path.join(src_annotations, file_name),
                os.path.join(dest_label_dir, file_name)
            )
            shutil.copyfile(
                os.path.join(image_path),
                os.path.join(dest_images_dir, image_name)
            )

            with open(os.path.join(dest_annotations_dir, f"{type}.txt"), "a+") as writer:
                writer.write(f"./images/{type}/{image_name}" + "\n")
    finally:
        pass

    print("Done")
    print(f"Total Train Images: {train_images}")
    print(f"Total Val Images: {val_images}")
    print(f"Total Test Images: {test_images}")


def create_yaml_file(dest_yaml_file: str, classes: list, dest_annotations_dir: str):
    """
    Create YAML data file used for training yolo
    :param dest_yaml_file:
    :param classes:
    :param dest_annotations_dir:
    :return:
    """
    check_exist(dest_annotations_dir)
    train_labels = os.path.join(dest_annotations_dir, "train.txt")
    test_labels = os.path.join(dest_annotations_dir, "test.txt")
    val_labels = os.path.join(dest_annotations_dir, "val.txt")

    check_exist(train_labels)
    check_exist(test_labels)
    check_exist(val_labels)

    if os.path.exists(dest_yaml_file):
        os.remove(dest_yaml_file)
    data = {"train": train_labels,
            "test": test_labels,
            "val": val_labels,
            "nc": len(classes),
            "names": classes
            }

    with open(dest_yaml_file, 'w') as file:
        yaml.dump(data, file)
    print(f"Yaml file created at {dest_yaml_file}")


def get_all_files(src_folder: str):
    """
    Get list of all file paths at all levels
    :param src_folder:
    :return: list of file paths
    """
    files = []

    def lookup(src_folder):
        for file in os.listdir(src_folder):
            full_path = os.path.join(src_folder, file)
            if os.path.isdir(full_path):
                lookup(full_path)
            else:
                files.append(full_path)

    if isinstance(src_folder, str):
        lookup(src_folder)
    elif isinstance(src_folder, (list, tuple)):
        for i in src_folder:
            lookup(i)

    return sorted(files)


class CocoHelpers:

    @staticmethod
    def stats(file_path: str):
        """
        Prints statistics of coco data
        :param file_path:
        :return:
        """
        coco = COCO(file_path)
        cat_ids = coco.getCatIds()
        img_ids = coco.getImgIds()
        print(f"File: {os.path.basename(file_path)}\n"
              f"Categories: {coco.loadCats(cat_ids)}\n"
              f"Total images: {len(img_ids)}\n"
              )

    @staticmethod
    def save_json(json_file: str, data):
        with open(json_file, "w+") as f:
            json.dump(data, f)

    @staticmethod
    def save_figure(fig, save_dir: str, image_name: str):
        assert os.path.exists(
            save_dir), f"Save figure directory doesnt not exist: {save_dir}"

        plt.savefig(os.path.join(save_dir, image_name),
                    # bbox='tight',
                    edgecolor=fig.get_edgecolor(),
                    facecolor=fig.get_facecolor(),
                    dpi=150
                    )
        plt.close()

    @classmethod
    def explore_data(cls, coco, file_name: str = "", save_dir: str = None, graph: int = 0):
        """
        Data exploration on box annotations in the provided coco file, find graphs below
        0. ALL graphs
        1. Distribution of total pixels in images (0 to max and mean to max)
        2. Distribution of image aspect ratios
        3. Distribution of Box aspect ratios
        4. Avg box area vs Categories
        5. Box Count vs Categories
        :param coco:
        :param file_name:
        :param save_dir:
        :param graph : graph id
        :return:
        """

        # Load coco annotations
        catIds = sorted(coco.getCatIds())
        imgIds = coco.getImgIds()
        annIds = coco.getAnnIds()
        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))
        fig.suptitle(f'File: {file_name} & '
                     f'Images: {len(imgIds)}',
                     fontsize=20, color="blue")

        # helper functions
        def set_title_labels(row, col, title, xlabel, ylabel):
            ax[row][col].set_title(f"{title}", fontsize=16, fontproperties=font_prop, fontweight="bold")
            ax[row][col].set_xlabel(xlabel, fontproperties=font_prop, fontsize=13)
            ax[row][col].set_ylabel(ylabel, fontproperties=font_prop, fontsize=13)
            # Set the tick labels font
            [label.set_fontproperties(font_prop) for label in
             (ax[row][col].get_xticklabels() + ax[row][col].get_yticklabels())]

        def save_figure(fig):
            plt.tight_layout()
            if save_dir:
                cls.save_figure(fig, save_dir, f"{file_name}_graphs.jpg")
            else:
                plt.show()

        img_pix = []
        img_ar = []
        bins = int(len(imgIds) / 5)
        for info in coco.loadImgs(ids=imgIds):
            img_pix.append(info['width'] * info['height'])
            img_ar.append(info['width'] / info['height'])

        # # Count vs Total Pixels Per Image
        # info = {
        #     "row": 0,
        #     "col": 0,
        #     "title": "Distribution of total pixels in images from 0 to max",
        #     "xlabel": "Total Pixels",
        #     "ylabel": "Count"
        # }
        # set_title_labels(**info)
        # ax[info["row"]][info["col"]].hist(img_pix, bins=bins, range=(0, max(img_pix)))
        #
        # info = {
        #     "row": 0,
        #     "col": 1,
        #     "title": "Distribution of total pixels in images from mean to max",
        #     "xlabel": "Total Pixels",
        #     "ylabel": "Count"
        # }
        # set_title_labels(**info)
        # ax[info["row"]][info["col"]].hist(img_pix, bins=bins, range=(mean(img_pix), max(img_pix)))

        # Count vs image aspect ratio(w/h): Gives info if most of the data consists of wide images or narrow or balanced
        info = {
            "row": 0,
            "col": 0,
            "title": "Distribution of image aspect ratios",
            "xlabel": "Image Aspect Ratio (w/h)",
            "ylabel": "Count"
        }
        set_title_labels(**info)
        img_ar = [x - 1 for x in img_ar]
        ax[info["row"]][info["col"]].hist(img_ar, bins=np.arange(min(img_ar), max(img_ar), 0.5))

        # Bounding Box Aspect Ration: Gives info if boxes are of narrow
        box_ar = []
        for info in coco.loadAnns(ids=annIds):
            width, height = info["bbox"][-2], info["bbox"][-1]
            box_ar.append(width / height)
        info = {
            "row": 0,
            "col": 1,
            "title": "Distribution of Box aspect ratios",
            "xlabel": "Box Aspect Ratio",
            "ylabel": "Count"
        }
        set_title_labels(**info)
        ax[info["row"]][info["col"]].hist(box_ar, bins=np.arange(min(img_ar), max(img_ar), 0.5))

        # Avg Box Area vs Category: Helps us to consider the best anchor sizes for object detection and filter outlier boxes
        area_per_cat = {dict["name"]: 0 for dict in coco.loadCats(catIds)}
        count_per_cat = dict(area_per_cat)
        for info in coco.loadAnns(ids=annIds):
            cat_name = coco.loadCats(ids=[info["category_id"]])[0]["name"]
            area = info["area"]

            temp = area_per_cat.get(cat_name)
            avg_area = area if temp is None else (area + temp) / 2.0
            area_per_cat.update({
                cat_name: avg_area
            })

            count_per_cat.update({
                cat_name: count_per_cat.get(cat_name) + 1
            })

        categories = list(count_per_cat.keys())
        box_count = list(count_per_cat.values())
        info = {
            "row": 1,
            "col": 0,
            "title": "Box Count vs Categories",
            "xlabel": "Category",
            "ylabel": "Box Count"
        }
        ax[info["row"]][info["col"]].bar(categories, box_count)
        set_title_labels(**info)

        categories = list(area_per_cat.keys())
        avg_areas = list(area_per_cat.values())
        info = {
            "row": 1,
            "col": 1,
            "title": "Avg box area vs Categories",
            "xlabel": "Category",
            "ylabel": "Avg box area"
        }
        ax[info["row"]][info["col"]].bar(categories, avg_areas)
        set_title_labels(**info)

        print(f"File: {file_name}\n"
              f"Categories: {coco.loadCats(catIds)}\n"
              f"Total images: {len(imgIds)}\n"
              )

        # print(f"File: {os.path.basename(file_path)}\n"
        #       f'Total Images: {len(imgIds)}\n'
        #       f"Total Boxes: {sum(box_count)}\n"
        #       f"Categories: {coco.loadCats(catIds)}\n"
        #       f"Box count per category: {count_per_cat}"
        #       )

        # Save all graphs :
        save_figure(plt.gcf())

    @staticmethod
    def draw_box(image_arr: np.array,
                 xmin: int, ymin: int, xmax: int, ymax: int,
                 label: str = None, score: float = None,
                 text_color: tuple = (255, 255, 255),
                 box_color: tuple = (255, 255, 0)):
        """
        draw box, label, score on the image provided
        :param image_arr: numpy array of shape (height, width, channels)
        :param xmin, ymin, xmax, ymax
        :param label:
        :param score:
        :param text_color:
        :param bbox_color:
        :return: image_arr
        """

        image_h, image_w = image_arr.shape[0], image_arr.shape[1]
        box_thick = round(0.002 * (image_h + image_w) / 2) + 1
        if box_thick < 1:
            box_thick = 1
        fontScale = 0.75 * box_thick

        # Draw box
        cv2.rectangle(image_arr, (xmin, ymin), (xmax, ymax),
                      box_color, box_thick * 2)

        # Format score & label
        if label or score:
            if score:
                score = str(score) if score > 1 else str(int(score * 100))
            else:
                score = ""

            label = str(label).strip() if label else ""
            label = f"{label} {score}"

            # get text size
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                                                  fontScale, thickness=box_thick)
            # put filled text rectangle
            # cv2.rectangle(image_arr, (xmin, ymin), (xmin + text_width, ymin - text_height - baseline), bbox_color,
            #               thickness=cv2.FILLED)

            # put text above rectangle
            # (xmin + bbox_thick, ymin + bbox_thick + text_height)
            cv2.putText(image_arr, label, (xmin, ymin - 2), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                        fontScale, text_color, box_thick, lineType=cv2.LINE_AA)

        return image_arr

    @classmethod
    def visualize_boxes(cls, coco_json: str, images_dir: str, save_dir_path: str, save_draw_images: bool = False,
                        save_boxes: bool = False):
        """
        1. Draw labelled objects on images and save them
        2. Extract labelled objects and save them in per class directory

        Note: expected box annotations are in format: xmin, ymin, width, height
        :param coco: loaded coco object
        :param imgs_dir: source images dir
        :param save_dir : save results
        :param save_boxes_dir : save objects in an image in respective class folder
        """
        try:
            image_name = None
            coco = COCO(coco_json)
            img_ids = coco.getImgIds()
            cat_ids = coco.getCatIds()
            random.shuffle(img_ids)
            idx_to_cat_name = {cat["id"]: cat["name"]
                               for cat in coco.loadCats(cat_ids)}

            # create save dir
            os.makedirs(save_dir_path, exist_ok=True)

            # create visualize dir if save_draw_images is true
            if save_draw_images:
                save_draw_images_dir = os.path.join(save_dir_path, "visualize")
                os.makedirs(save_draw_images_dir, exist_ok=True)

                # if save boxes enabled create directories per class
            if save_boxes:
                save_boxes_dir = os.path.join(save_dir_path, "extracted_boxes")
                os.makedirs(save_boxes_dir, exist_ok=True)
                for category_details in coco.loadCats(cat_ids):
                    cat_save_dir = os.path.join(save_boxes_dir,
                                                f"{category_details['id']}_{category_details['name']}")
                    os.makedirs(cat_save_dir, exist_ok=True)

            # create set of colors for classes
            num_classes = len(cat_ids)
            hsv_tuples = [(1.0 * x / num_classes, 1., 1.)
                          for x in range(num_classes)]
            rand_colors = list(
                map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
            rand_colors = list(
                map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), rand_colors))
            random.seed(0)
            random.shuffle(rand_colors)
            random.seed(None)

            # loop through images and draw boxes
            for id in tqdm.tqdm(img_ids):
                image_name = coco.loadImgs(ids=[id])[0]["file_name"]
                image_path = image_name if images_dir is None else os.path.join(
                    images_dir, image_name)

                if not os.path.exists(image_path):
                    continue
                image_arr = cv2.imread(image_path)
                image_arr = cv2.cvtColor(image_arr, cv2.COLOR_BGR2RGB)
                annotations_per_img = coco.loadAnns(coco.getAnnIds(imgIds=id))
                title = []

                for i, annotation in enumerate(annotations_per_img):
                    category_id = annotation["category_id"]
                    # ,{round( annotation.get('score',0),1 )}"
                    category_name = f"{coco.loadCats(ids=category_id)[0]['name']}"
                    bbox = annotation["bbox"]  # xmin,ymin,width,height
                    bbox_color = rand_colors[cat_ids.index(category_id)]

                    box = [int(x) for x in bbox]
                    xmin, ymin, width, height = box
                    xmax, ymax = xmin + width, ymin + height

                    if save_boxes:
                        box_arr = image_arr[ymin:ymin +
                                                 height, xmin:xmin + width, :]
                        image_name_split = os.path.splitext(image_name)
                        cat_save_dir = os.path.join(
                            save_boxes_dir, f"{category_id}_{category_name}")
                        box_save_path = os.path.join(cat_save_dir,
                                                     image_name_split[0] + f"_{i}.jpg")  # {image_name_split[1]}
                        plt.imsave(box_save_path, box_arr)

                    if save_draw_images:
                        image_arr = cls.draw_box(image_arr, xmin, ymin, xmax, ymax, label=category_name,
                                                 box_color=bbox_color)
                        title.append(category_name)

                if save_draw_images:
                    plt.imsave(os.path.join(save_draw_images_dir, image_name),
                               image_arr)

                # elif visualize:
                #     plt.imshow(image_arr)
                #     plt.show()
                #     time.sleep(2)
        except Exception as e:
            print(image_name)
            print(e)

    # @classmethod
    # def visualize_coco(cls, coco_json: str, images_dir: str = None, save_dir: str = None):
    #     '''
    #     :param coco: loaded coco object
    #     :param imgs_dir: source images dir
    #     :param save_dir : save results
    #
    #     Note: expected box annotations are in format: xmin, ymin, width, height
    #     '''
    #     coco = COCO(coco_json)
    #     img_ids = coco.getImgIds()
    #     random.shuffle(img_ids)
    #
    #     for id in tqdm.tqdm(img_ids):
    #         image_name = coco.loadImgs(ids=[id])[0]["file_name"]
    #         image_path = image_name if images_dir is None else os.path.join(images_dir, image_name)
    #
    #         anns = coco.loadAnns(coco.getAnnIds(imgIds=[id]))
    #         categories = ", ".join([str(a["category_id"]) for a in anns])
    #
    #         I = plt.imread(image_path)
    #         plt.imshow(I)
    #         plt.axis('off')
    #         title = f"{os.path.basename(image_path)} : {categories}"
    #         plt.title(f"classes: {categories}")
    #         coco.showAnns(anns, draw_bbox=True)
    #
    #         if save_dir:
    #             fig = plt.gcf()
    #             cls.save_figure(fig, save_dir, image_name)
    #         else:
    #             plt.show()
    #             time.sleep(2)

    @staticmethod
    def img_info(file_path: str):
        """
        :param file_path: image path
        :return: filename, width, height, channels
        """
        filename = os.path.basename(file_path)
        img_arr = cv2.imread(file_path)  # Image.open(file_path).convert('L')

        height, width = img_arr.shape[0], img_arr.shape[1]
        return filename, width, height, 0

    @staticmethod
    def darknet_box_to_coco(file_name: str, img_width: int, img_height: int, delimiter: str = " "):
        """
        :param file_name: darkenet annotation file
        :param img_width:
        :param img_height:
        :return: <list of cls_ids >, < list of boxes in format (xmin, ymin, box_width, box_height)>
        """
        cls_ids, boxes = [], []
        with open(file_name) as f:
            lines = f.readlines()
            for line in lines:
                cls_id, *box = line.strip().split(delimiter)
                box = [float(x) for x in box]

                for x in box:
                    if x < 0 or x > 1:
                        print("Invalid darknet box format", file_name)

                box_width = round(box[2] * img_width, 2)
                box_height = round(box[3] * img_height, 2)
                box_x_ctr = box[0] * img_width
                box_y_ctr = box[1] * img_height
                xmin = round(box_x_ctr - box_width / 2., 2)
                ymin = round(box_y_ctr - box_height / 2., 2)

                coco_box = [int(xmin), int(ymin), int(
                    box_width), int(box_height)]
                cls_ids.append(int(cls_id))
                boxes.append(coco_box)
        return cls_ids, boxes

    @classmethod
    def convert_darknet_to_coco(cls, images: list, annotations: list, idx_to_cat: dict = None,
                                validation_split: float = 0.2):
        """
        Convert darknet annotations to COCO annotations

        :param images: list of image files paths
        :param annotations: list of darknet annotations files paths
        :param idx_to_cat: dictionary of { id: category_name,... }
        :param validation_split: ratio of train:validation split
        """

        # initialize var
        img_idx = 0
        annot_idx = 0
        cat_idx = 0
        categories = []
        train_annot = {"info": {"year": 2021, "version": "1.0"},
                       "images": [], "annotations": [], "categories": []}
        val_annot = {"info": {"year": 2021, "version": "1.0"},
                     "images": [], "annotations": [], "categories": []}
        images_with_no_annotations = []

        for image_path in tqdm.tqdm(images):
            image_name = os.path.basename(image_path)
            annot_path = [annot_path for annot_path in annotations if
                          Path(annot_path).stem.strip() == Path(image_path).stem.strip()]
            is_valid = random.random() <= validation_split

            if not annot_path:
                images_with_no_annotations.append(image_path)
                # print("images_with_no_annotations: ", image_name)
                continue
            else:
                annot_path = annot_path[0]

            # get image details :
            filename, width, height, _ = cls.img_info(image_path)
            img_idx += 1
            image_details = {"id": img_idx, "width": width, "height": height, "file_name": image_name,
                             "license": 0, "date_captured": ""}

            # Convert darkent format box to coco
            cat_ids, boxes = cls.darknet_box_to_coco(annot_path, width, height)

            # Coco cat_ids start from 1, as Id 0 is considered as background
            cat_ids = [x + 1 for x in cat_ids]

            # Skip images with categories outside of idx_to_cat
            if not set(cat_ids).issubset(set(idx_to_cat.keys())):
                print("outside cat_idx: ", image_name)
                continue

            # add image details to annotations
            if is_valid:
                val_annot["images"].append(image_details)
            else:
                train_annot["images"].append(image_details)

            # Add annotations
            for cat_id, box in zip(cat_ids, boxes):
                cat_name = idx_to_cat.get(cat_id)
                is_add_cat = not len(
                    [cat["id"] for cat in categories if cat["id"] == cat_id])
                if is_add_cat:
                    categories.append({
                        "name": cat_name,
                        "id": cat_id
                    })

                annot_idx += 1
                xmin, ymin, box_width, box_height = box
                xmax, ymax = xmin + box_width, ymin + box_height
                annotation = {"id": annot_idx,
                              "image_id": img_idx,
                              "segmentation": [[xmin, ymin,
                                                xmin, ymax,
                                                xmax, ymax,
                                                xmax, ymin
                                                ]],
                              "category_id": int(cat_id),
                              "bbox": box,
                              "area": box_width * box_height,
                              "iscrowd": 0, }

                if is_valid:
                    val_annot["annotations"].append(annotation)
                else:
                    train_annot["annotations"].append(annotation)

        train_annot["categories"] = categories
        val_annot["categories"] = categories

        if len(images_with_no_annotations):
            print(
                "Images with no annotations: " + '\n'.join(images_with_no_annotations))
        return train_annot, val_annot

    @classmethod
    def convert_coco_to_darknet(cls, coco: COCO, darknet_destination: str, cat_to_idx: dict = None):
        """
        Function helps in converting coco labels to darknet labels

        :param coco: coco object
        :param darknet_destination: dir to save darknet annotations
        :param cat_to_idx: dictionary of { category_name : id .... }
        :return:
        """
        os.makedirs(darknet_destination, exist_ok=True)

        imgIds = coco.getImgIds()

        for img_id in imgIds:
            img_annotations = coco.getAnnIds(imgIds=img_id)
            img_details = coco.loadImgs(img_id)[0]
            img_name, img_width, img_height = img_details[
                                                  "file_name"], img_details["width"], img_details["height"]

            # Skip if annotations not exist for the image:
            if not len(img_annotations):
                continue

            for annot_id in img_annotations:
                img_annot = coco.loadAnns(annot_id)[0]
                cat_id, box = img_annot["category_id"], img_annot["bbox"]
                category_name = coco.loadCats(cat_id)[0]["name"]

                if box is None or len(box) == 0 or cat_id <= 0:
                    continue

                # darknet categories start from idx 0 whereas coco start from 1 , so minus 1
                if cat_to_idx is not None:
                    if category_name in list(cat_to_idx.keys()):
                        cat_id = cat_to_idx[category_name]
                    else:
                        continue
                else:
                    cat_id -= 1

                with open(os.path.join(darknet_destination, Path(img_name).stem + ".txt"), 'a+') as f:
                    bw = box[2]
                    bh = box[3]
                    cx = box[0] + bw / 2.0
                    cy = box[1] + bh / 2.0

                    norm_cx = cx / float(img_width)
                    norm_cy = cy / float(img_height)
                    norm_bw = bw / float(img_width)
                    norm_bh = bh / float(img_height)

                    box = [norm_cx, norm_cy, norm_bw, norm_bh]
                    box = [str(round(x, 4)) for x in box]
                    box = [str(cat_id), *box]
                    f.write(" ".join(box) + "\n")

    @classmethod
    def create_coco_annotations(cls, img_dir: str, idx_to_cat: dict = None, save_json_path: str = None):
        annotations_dict = {"images": [], "categories": [], "annotations": []}
        assert os.path.exists(
            img_dir), f"Image directory does not exist: {img_dir}"

        for img_name in tqdm.tqdm(os.listdir(img_dir), desc="Reading images"):
            img_path = os.path.join(img_dir, img_name)
            filename, width, height, _ = cls.img_info(img_path)
            image_details = {"id": len(annotations_dict["images"]), "width": width, "height": height,
                             "file_name": img_name}
            annotations_dict["images"].append(image_details)

        if idx_to_cat is not None:
            annotations_dict["categories"].extend(
                [{"id": id, "name": cat_name}
                 for id, cat_name in idx_to_cat.items()]
            )

        if save_json_path is not None and save_json_path.lower().endswith(".json"):
            cls.save_json(save_json_path, annotations_dict)

        return annotations_dict

    @staticmethod
    def convert_json_to_coco(json_file: str, images_dir: str, result_coco_file: str, cat_to_idx: dict):
        """
        Convert the via tool JSON annotations to coco annotations

        :param json_file: via tool json format
        :param images_dir: directory path to images
        :param result_coco_file: path to save coco file
        :return:
        """
        final = {"info": {"year": 2021, "version": "1.0"},
                 "images": [], "annotations": [], "categories": []}
        categories = []
        annotations = []
        images = []
        json_data = json.load(open(json_file, "r"))

        for k, img_data in tqdm.tqdm(json_data.items()):
            file_name = img_data["filename"]
            file_path = os.path.join(images_dir, file_name)
            _, width, height, channels = CocoHelpers.img_info(file_path)

            regions = img_data["regions"]
            for region_data in regions:
                # if not( "shape_attributes" in region_data.keys() and "region_attributes" in region_data.keys() ): continue
                shape_attributes = region_data["shape_attributes"]
                region_attributes = region_data["region_attributes"]

                all_points_x = shape_attributes["all_points_x"]
                all_points_y = shape_attributes["all_points_y"]
                [(super_category_name, category_name)] = region_attributes.items()
                category_name = str(category_name).strip()
                category_id = int(cat_to_idx[category_name])

                segmentation = []
                [segmentation.extend([x, y])
                 for x, y in zip(all_points_x, all_points_y)]

                # Add image details to coco
                if len(images) == 0 or file_name not in [x["file_name"] for x in images]:
                    image_details = {"id": len(images) + 1, "width": width, "height": height, "file_name": file_name,
                                     "license": 0, "date_captured": ""}
                    images.append(
                        image_details
                    )

                # Add annotation to coco
                xmin, ymin, xmax, ymax = min(all_points_x), min(
                    all_points_y), max(all_points_x), max(all_points_y)
                box_width, box_height = xmax - xmin, ymax - ymin
                box = [xmin, ymin, box_width, box_height]
                image_id = [x["id"]
                            for x in images if x["file_name"] == file_name][0]
                annotation_details = {"id": len(annotations) + 1,
                                      "image_id": image_id,
                                      "segmentation": [segmentation],
                                      "category_id": int(category_id),
                                      "bbox": box,
                                      "area": box_width * box_height,
                                      "iscrowd": 0, }
                annotations.append(annotation_details)

                # Add categories to coco
                if len(categories) == 0 or category_id not in [x["id"] for x in categories]:
                    category_details = {
                        "name": category_name, "id": category_id}
                    categories.append(
                        category_details
                    )
        final["images"], final["annotations"], final["categories"] = images, annotations, categories
        with open(result_coco_file, "w+") as f:
            json.dump(final, f)

    @staticmethod
    def combine_coco_files(coco_files: list, save_path: str):
        """
        1. Merge the different coco files to one single coco file.
        Note : all coco files should have same categories

        :param coco_files: list of coco file paths
        :param save_path: path to save final coco file
        :return:
        """
        final = {"info": {"year": 2021, "version": "1.0"},
                 "images": [], "annotations": [], "categories": []}
        current_img_id = 0
        current_annot_id = 0
        for file_path in coco_files:
            coco = COCO(file_path)
            imgIds = coco.getImgIds()

            for imgid in imgIds:

                image_details = coco.loadImgs(ids=[imgid])
                image_name = image_details[0]["file_name"]
                annotIds = coco.getAnnIds(imgIds=[imgid])

                # if found same image names in files, skip them
                final_image_names = [f["file_name"] for f in final["images"]]
                if image_name in final_image_names:
                    print(
                        f"Duplicate image name found -> Skipping {image_name} in {file_path}")
                    continue

                # check if annotations exist for the image
                if len(annotIds):
                    img_annotations = coco.loadAnns(ids=annotIds)

                    current_img_id += 1
                    image_details[0]["id"] = current_img_id
                    for i in range(len(img_annotations)):
                        current_annot_id += 1
                        img_annotations[i]["id"] = current_annot_id
                        img_annotations[i]["image_id"] = current_img_id

                    final['images'].extend(image_details)
                    final["annotations"].extend(img_annotations)
                else:
                    print("No annotations for ", image_name)

            final["categories"] = [
                cat for cat in coco.loadCats(ids=coco.getCatIds())]

        with open(save_path, "w+") as f:
            json.dump(final, f)

    @staticmethod
    def split_coco_files(coco_file: str, split_ratios: list, split_save_paths: list):
        """
        Split the given coco_file into multiple files based on split ratios given

        :param coco_file: coco file path
        :param split_ratios: split the coco according to given ratios
        :param split_save_paths: list of coco file paths to be saved
        :return:
        """
        assert len(split_ratios) == len(
            split_save_paths), "len of split rations should be equal to total save paths"
        assert round(sum(split_ratios)) == 1, "sum of splits should equal to 1"

        coco = COCO(coco_file)
        src_img_ids = coco.getImgIds()
        random.shuffle(src_img_ids)

        split_ratios = [sum(split_ratios[:i + 1])
                        for i in range(len(split_ratios))]
        split_img_ids = np.split(
            src_img_ids, [int(len(src_img_ids) * x) for x in split_ratios])

        for split_index, imgIds in enumerate(split_img_ids[:-1]):
            imgIds = list(imgIds)
            current_img_id = 0
            current_annot_id = 0
            final = {"info": {"year": 2021, "version": "1.0"},
                     "images": [], "annotations": [], "categories": []}
            for imgid in imgIds:

                image_details = coco.loadImgs(ids=[imgid])
                image_name = image_details[0]["file_name"]
                annotIds = coco.getAnnIds(imgIds=[imgid])

                # if found same image names in files, skip them
                final_image_names = [f["file_name"] for f in final["images"]]
                if image_name in final_image_names:
                    print(
                        f"Duplicate image name found -> Skipping {image_name}")
                    continue

                # check if annotations exist for the image
                if len(annotIds):
                    img_annotations = coco.loadAnns(ids=annotIds)

                    current_img_id += 1
                    image_details[0]["id"] = current_img_id
                    for i in range(len(img_annotations)):
                        current_annot_id += 1
                        img_annotations[i]["id"] = current_annot_id
                        img_annotations[i]["image_id"] = current_img_id

                    final['images'].extend(image_details)
                    final["annotations"].extend(img_annotations)
                else:
                    print("No annotations for ", image_name)

            final["categories"] = [
                cat for cat in coco.loadCats(ids=coco.getCatIds())]
            with open(split_save_paths[split_index], "w+") as f:
                json.dump(final, f)

    @staticmethod
    def save_coco_mask_to_image(coco_file: str, image_dir: str, mask_save_dir: str = None):
        """
        Creates mask images from coco mask annotations

        :param coco_file: path to coco file
        :param image_dir: images directory
        :param mask_save_dir: path to save mask images
        :return:
        """
        coco = COCO(coco_file)
        imgids = coco.getImgIds()
        cat_ids = coco.getCatIds()

        for imgid in tqdm.tqdm(imgids):
            img = coco.loadImgs(ids=imgid)[0]
            anns_ids = coco.getAnnIds(
                imgIds=imgid, catIds=cat_ids, iscrowd=None)
            anns = coco.loadAnns(anns_ids)
            anns_img = np.zeros((img['height'], img['width']))
            for ann in anns:
                coco_mask = coco.annToMask(ann)
                anns_img = np.maximum(anns_img, coco_mask * ann['category_id'])
                # objids = np.unique(anns_img)
                # print(imgid, img["file_name"], objids)

            if mask_save_dir:
                plt.imsave(
                    os.path.join(mask_save_dir, img["file_name"]),
                    anns_img,
                    cmap="gray"
                )
            else:
                plt.imshow(anns_img)
                plt.show()
                img = plt.imread(os.path.join(image_dir, img["file_name"]))
                plt.imshow(img)
                plt.show()

    @staticmethod
    def validate_coco_height_width(coco_file: str, img_dir: str, fix: bool = False):
        """
        Make sure height and width of coco annotations and original images match

        :param coco_file: path to coco file
        :param images_dir: path to images directory
        :return:
        """
        annotations_dict = {"images": [], "categories": [], "annotations": []}
        assert os.path.exists(
            img_dir), f"Image directory does not exist: {img_dir}"
        assert os.path.exists(
            coco_file), f"Coco file does not exist: {coco_file}"

        coco = COCO(coco_file)
        all_img_details = coco.loadImgs(ids=coco.getImgIds())
        annot_ids = coco.getAnnIds(imgIds=coco.getImgIds())
        annotations = coco.loadAnns(ids=annot_ids)
        annotations_dict["annotations"] = annotations
        annotations_dict["categories"] = coco.loadCats(ids=coco.getCatIds())

        for img_details in tqdm.tqdm(all_img_details, desc="Reading coco"):
            img_name = img_details["file_name"]
            coco_height, coco_width = img_details["height"], img_details["width"]

            img_path = os.path.join(img_dir, img_name)
            filename, width, height, _ = CocoHelpers.img_info(img_path)

            if coco_height != height or coco_width != width:
                print(f"Invalid image dimensions in coco found {img_name}")
                img_details["height"], img_details["width"] = height, width

            if fix:
                annotations_dict["images"].append(
                    img_details
                )

        if fix:
            CocoHelpers.save_json(coco_file, annotations_dict)

        return annotations_dict


if __name__ == "__main__":
    pass

    # Fix coco height width
    # CocoHelpers.validate_coco_height_width(
    #     coco_file="/home/sohoa1/rammy/datasets/analog_meter/maskrcnn/annotations_important/new_plus_old_coco_train.json",
    #     img_dir="/home/sohoa1/rammy/datasets/analog_meter/maskrcnn/new_plus_old_images",
    #     fix= False
    # )

    # Convert JSON file from VIA to coco
    # json_file = "/home/sohoa1/rammy/datasets/2021/object_detection/fire_extinguisher/mask_rcnn/mask_annotations_fire_json.json"
    # images_dir = "/home/sohoa1/rammy/datasets/2021/object_detection/fire_extinguisher/mask_rcnn/mask_labelled_images"
    # result_coco_file = "/home/sohoa1/rammy/datasets/2021/object_detection/fire_extinguisher/mask_rcnn/mask_annotations_fire_coco.json"
    # classes = [ "arrow", "red", "green", "yellow" ]
    # cat_to_idx = { class_name:i+1 for i, class_name in enumerate(classes) }
    # CocoHelpers.convert_json_to_coco(json_file,images_dir, result_coco_file, cat_to_idx)

    # Save coco mask annotations to mask images
    # CocoHelpers.save_coco_mask_to_image(
    #     coco_file="/home/sohoa1/rammy/datasets/digital_meter/yolact/mask_annotations_yolact/final/val_2021.json",
    #     image_dir="/home/sohoa1/rammy/datasets/digital_meter/yolact/mask_annotations_yolact/final/validation/images",
    #     mask_save_dir="/home/sohoa1/rammy/datasets/digital_meter/yolact/mask_annotations_yolact/final/validation/masks"
    # )

    # Combine COCO
    # CocoHelpers.combine_coco_files(
    #     [
    #      "/home/sohoa1/rammy/datasets/2021/object_detection/accu_battery/coco_json_files_all_classes/train_annotations.json",
    #     "/home/sohoa1/rammy/datasets/2021/object_detection/accu_battery/coco_json_files_all_classes/validation_annotations.json"
    #     ],
    #     save_path="/home/sohoa1/rammy/datasets/2021/object_detection/accu_battery/coco_json_files_all_classes/accu_battery_coco.json"
    # )

    # Split coco
    # CocoHelpers.split_coco_files(
    #     coco_file="/home/sohoa1/rammy/datasets/2021/object_detection/fire_extinguisher/frcnn/all_images_fire_meter_coco.json",
    #     split_ratios= [0.7,0.2,0.1],
    #     split_save_paths= [
    #         "/home/sohoa1/rammy/datasets/2021/object_detection/fire_extinguisher/frcnn/all_images_fire_meter_coco_train1.json",
    #         "/home/sohoa1/rammy/datasets/2021/object_detection/fire_extinguisher/frcnn/all_images_fire_meter_coco_val1.json",
    #         "/home/sohoa1/rammy/datasets/2021/object_detection/fire_extinguisher/frcnn/all_images_fire_meter_coco_test1.json"
    #     ]
    # )

    # Darknet to JSON
    # images = get_all_files(
    #     [r"E:\FPT\projects\panama\data\dataset\images\valid"])
    # annots = get_all_files(
    #     [r"E:\FPT\projects\panama\data\dataset\labels\valid"])
    # classes = ["折戸", "開戸", "引戸"]
    # idx_to_cat = {i + 1: name for i, name in enumerate(classes)}
    # train_annot, val_annot = CocoHelpers.convert_darknet_to_coco(
    #     images, annots, idx_to_cat, validation_split=0)
    # json_save_dir = r"E:\FPT\models\projects\pana\data"
    # train_annotations = os.path.join(json_save_dir, "valid_3doors_coco.json")
    # CocoHelpers.save_json(train_annotations, train_annot)

    # Explore data
    coco_annotations = r"E:\FPT\models\projects\pana\data\test_3doors_coco.json"
    CocoHelpers.explore_data(
        COCO(coco_annotations),
        graph=0,
        file_name=Path(coco_annotations).stem,
        save_dir=Path(coco_annotations).parent
    )

    # Visualize
    # train_annotations = "/home/sohoa1/rammy/datasets/2021/object_detection/smoke_detector/smoke_with_7dc_coco.json"
    # images_dir = "/home/sohoa1/rammy/datasets/2021/object_detection/smoke_detector/original_images"
    # CocoHelpers.visualize_boxes( train_annotations,
    #                             images_dir,
    #                             save_dir_path= "/home/sohoa1/rammy/datasets/2021/object_detection/smoke_detector/visualize",
    #                             save_boxes=True,
    #                             save_draw_images=True
    #                            )

    # Stats
    # CocoHelpers.stats(train_annotations)

    # Visualize
    # image_dir = "/home/sohoa1/rammy/main_projects/experiment_projects/Unbiased_teacher_frcnn/unbiased_new/unlabel_data"
    # annots = "/home/sohoa1/rammy/main_projects/experiment_projects/Unbiased_teacher_frcnn/unbiased_new/results/label_creation/label_generate.json"
    # CocoHelpers.visualize_boxes(
    #     annots,#"/home/sohoa1/rammy/main_projects/experiment_projects/Unbiased_teacher_frcnn/unbiased_new/results/label_creation/best_inferences.json",
    #     image_dir,#"/home/sohoa1/rammy/datasets/digital_meter/yolo/coco/cleaned_data/train_no_rotation",
    #     save_dir= "/home/sohoa1/rammy/main_projects/experiment_projects/Unbiased_teacher_frcnn/unbiased_new/results/visualize_unlabel_data_inference"
    # )
    # pass

    # Test coco to darknet
    # train_annotations = "/home/sohoa1/rammy/datasets/2021/object_detection/accu_battery/yolo_darknet.json"
    # darknet_destination = "/home/sohoa1/rammy/datasets/2021/object_detection/accu_battery/darknet_labels"
    # classes = ["p1+p6", "p2+p3", "p4"]
    # cat_to_idx = { name:idx for idx,name in enumerate(classes) }
    # f = COCO(train_annotations)
    # CocoHelpers.convert_coco_to_darknet(
    # f,
    # darknet_destination=darknet_destination,
    # cat_to_idx = cat_to_idx
    # )

    # # Create Darkent format files for training with yolo
    # src_annotations = darknet_destination
    # source_img_dir = "/home/sohoa1/rammy/datasets/2021/object_detection/accu_battery/images"
    # dest_annotations_dir = "/home/sohoa1/rammy/datasets/2021/object_detection/accu_battery/yolo_darknet_form_3_classes"
    # yolo_data_split(src_annotations,
    #                 source_img_dir,
    #                 dest_annotations_dir)

    # Create YAML file for yolo training
    # classes = ["0","1","2","3","4","5","6","7","8","9",".","r3","r4","r7"]
    # dest_yaml_file = "/home/sohoa1/rammy/main_projects/yolov5/data/accu_batter_3.yaml"
    # dest_annotations_dir = dest_annotations_dir
    # create_yaml_file(dest_yaml_file,
    #                  classes,
    #                  dest_annotations_dir)
