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
import matplotlib.pyplot as plt


class DatasetFromMaskImages(object):
    def __init__(self, root, transforms = None ):
        self.root = root
        self.transforms = transforms
        self.idx_to_class = {1: "meter"}
        # load all image files, sorting them to ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "images"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "masks"))))

    def __getitem__(self, idx):
        # load images and masks
        img_path = os.path.join(self.root, "images", self.imgs[idx])
        mask_path = os.path.join(self.root, "masks", self.masks[idx])
        img = cv2.imread( img_path )
        img = cv2.cvtColor( img, cv2.COLOR_BGR2RGB )

        mask = cv2.imread( mask_path, cv2.IMREAD_GRAYSCALE )
        mask = np.array( mask/255, dtype=np.int)

        # visualize
        # plt.imshow(img)
        # plt.show()
        # plt.imshow(mask, cmap="gray")
        # plt.show()

        # instances are encoded as different colors
        obj_ids = np.unique(mask)

        # drop 0 which is background
        obj_ids = np.delete( obj_ids, np.argwhere( obj_ids == 0  ) )

        # split the color-encoded mask into a set of binary masks
        masks = mask == obj_ids[:, None, None]
        masks = np.array( masks , dtype=np.float )

        #todo: Assuming only single mask exist per image: need to rewrite code for multiple masks
        if self.transforms:
            transformed = self.transforms( image = img, masks = [x for x in masks] )
            img = transformed["image"]
            masks = transformed["masks"]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        # visualize
        # if idx % 50 == 0:
            # plt.imshow(img)
            # plt.show()
            # plt.imshow(masks[0], cmap="gray") # visualize one mask
            # plt.show()
            # img = CocoHelpers.draw_box( img, *boxes[0] )
            # plt.imshow(img)
            # plt.show()

        #todo: assuming there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        img = F.to_tensor(img)
        image_id = torch.tensor([idx])

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        return img, target

    def __len__(self):
        return len(self.imgs)


class DatasetFromJSON(object):
    def __init__(self, images_dir: str, json_file: str, transforms = None ):
        self.images_dir = images_dir
        self.json_file  = json_file
        self.transforms = transforms

        # load all image files, sorting them to
        # ensure that they are aligned
        self.images = list(sorted(os.listdir(self.images_dir)))
        self.annotations = json.load(open(self.json_file))

    def __getitem__(self, idx):
        # load images
        img_name = self.images[idx]
        img_path = os.path.join(self.images_dir, img_name)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # plt.imshow(img)
        # plt.show()

        image_annotation = self.annotations[ img_name ]
        regions = image_annotation["regions"]
        height, width, channels = img.shape
        masks = np.zeros((len(regions),height, width), dtype=np.uint8)

        label = []
        # instances are encoded as different colors
        for i,r in enumerate(regions):
            xx, yy = skimage.draw.polygon(r["shape_attributes"]["all_points_x"], r["shape_attributes"]["all_points_y"])
            masks[i,yy,xx] = 1
            if str(r["region_attributes"]["mano"]).strip() == '':
                print("Invalid label ID found", image_annotation)
            label.append(np.int(r["region_attributes"]["mano"]))

        if self.transforms:
            transformed = self.transforms(image=img, masks=[x for x in masks])
            img = transformed["image"]
            masks = transformed["masks"]

        # get bounding box coordinates for each mask
        num_objs = len(label)
        boxes = []
        for i in range(len(label)):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        # convert all to tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(label, dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        # plt.imshow(img)
        # plt.show()
        # for m in masks:
        #     plt.imshow(m)
        #     plt.show()

        img = F.to_tensor(img)
        return img, target

    def __len__(self):
        return len(self.images_dir)


class DatasetFromCOCO(object):
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

            annot_ids           = self.coco.getAnnIds(imgIds=[img_id])
            image_annotations   = self.coco.loadAnns(ids=annot_ids)
            height, width, _    = img.shape
            masks               = np.zeros((len(image_annotations), height, width), dtype=np.uint8)

            label = []
            for i,annot in enumerate(image_annotations):
                xx, yy = annot["segmentation"][0][::2], annot["segmentation"][0][1::2]
                xx, yy = skimage.draw.polygon(xx, yy)
                masks[i, yy, xx] = 1
                label.append( annot["category_id"] )

            if self.transforms:
                transformed = self.transforms(image=img, masks=[x for x in masks])
                img     = transformed["image"]
                masks   = transformed["masks"]

            # get bounding box coordinates for each mask
            num_objs = len(label)
            boxes = []
            for i in range(len(label)):
                pos = np.where(masks[i])
                # print( "pos", np.sum(masks[i]) , pos )
                xmin = np.min(pos[1])
                xmax = np.max(pos[1])
                ymin = np.min(pos[0])
                ymax = np.max(pos[0])
                boxes.append([xmin, ymin, xmax, ymax])

            # convert all to tensors
            boxes = torch.as_tensor(boxes, dtype=torch.int32)
            labels = torch.as_tensor(label, dtype=torch.int64)
            masks = torch.as_tensor(masks, dtype=torch.uint8)
            image_id = torch.tensor([idx])
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

            # suppose all instances are not crowd
            iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

            # Visualize image and masks after transform
            # plt.imshow(img)
            # plt.show()
            # for m in masks:
            #     plt.imshow(m)
            #     plt.show()

            target = {}
            target["boxes"] = boxes
            target["labels"] = labels
            target["masks"] = masks
            target["image_id"] = image_id
            target["area"] = area
            target["iscrowd"] = iscrowd

            img = F.to_tensor(img)
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
