import cv2
import os
import  numpy as np
import torch
import torchvision.transforms.functional as F
import skimage
import json
from pycocotools.coco import COCO
import random
import urllib
from PIL import Image
import io
from torch.utils.data import Dataset


class DatasetFromCOCO(Dataset):
    """
    Loading dataset from coco file format
    """
    def __init__(self, images_dir: str, json_file: str, transforms = None ):
        self.images_dir = images_dir
        self.coco_file  = json_file
        self.transforms = transforms
        self.coco = COCO(self.coco_file)
        self.idx_to_class = {  x["id"]:x["name"] for x in self.coco.loadCats( self.coco.getCatIds() ) }

        # load all image files
        self.img_ids = self.coco.getImgIds()
        random.shuffle(self.img_ids)

    def __getitem__(self, idx):
        try:
            # load image details
            img_id      = self.img_ids[idx]
            img_details = self.coco.loadImgs(ids=[img_id])[0]
            img_name    = img_details["file_name"]
            img_path    = os.path.join(self.images_dir, img_name)

            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Visualize image before transform
            # print("img_name", img_name)
            # plt.imshow(img)
            # plt.show()

            # Read Annotations
            annot_ids = self.coco.getAnnIds(imgIds=[img_id])
            annotations = self.coco.loadAnns(ids=annot_ids)
            boxes, labels, iscrowd = list(
                zip(*[(annot["bbox"], annot['category_id'], annot['iscrowd']) for annot in annotations]))

            # convert from xmin,ymin,w,h to xmin,ymin,xmax,ymax
            boxes = [[box[0], box[1], box[0] + box[2], box[1] + box[3]] for box in
                     boxes]

            # apply transforms
            if self.transforms is not None:
                transformed = self.transforms(image=img, bboxes=boxes, class_labels=labels)
                img = transformed['image']
                boxes = transformed['bboxes']
                labels = transformed['class_labels']

            # convert everything into a torch.Tensor
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            area = torch.as_tensor([(x[2] - x[0]) * (x[3] - x[1]) for x in boxes], dtype=torch.float32)
            img = F.to_tensor(img)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            image_id = torch.tensor(img_id)
            iscrowd = torch.tensor(iscrowd, dtype=torch.int64)

            target = {"boxes": boxes, "labels": labels, "image_id": image_id, "iscrowd": iscrowd, "area": area}

            return img, target

        except Exception as e:
            print(e)
            print(img_name)

    def __len__(self):
        return len(self.img_ids)


class LoadImagesDataset:
    def __init__(self, image_paths: list):
        """
        Create a dataset from the images provided
        :param image_paths: list of fpt paths or image_paths
        """
        assert isinstance(image_paths, list), "image_paths is expected to be a list of image paths"
        self.files = self.__filter(image_paths)
        self.num_files = len(self.files)
        assert self.num_files > 0, 'No images found in %s ' % image_paths

    def __filter(self,image_paths):
        img_extensions = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')
        files = [ f for f in image_paths if str(f).endswith(img_extensions) ]
        return files

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns: image_arr (numpy)
        """
        path = self.files[index]
        filename = os.path.basename(path)

        # Read image
        image = self.read_image(path)
        assert image is not None, 'Image Not Found ' + path

        image = np.array(image)
        return image, filename

    def __len__(self):
        return self.num_files

    @classmethod
    def read_image(cls, path):
        if path.startswith("ftp"):
            image_bytes = cls.fetch_data(path)
            image = Image.open(io.BytesIO(image_bytes))
        elif os.path.exists(path):
            image = Image.open(path)
        else:
            raise Exception(f'Invalid path in image list: {path}')

        image.convert('RGB')
        return image

    @classmethod
    def fetch_data(cls, url):
        ftp_path = url.replace('#', '%23')
        return urllib.request.urlopen(ftp_path).read()


def collate_fn(batch):
    return tuple(zip(*batch))
