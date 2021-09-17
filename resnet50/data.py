import cv2
import os
import random
import torch
from torchvision import transforms
from torch import cuda
from torch.utils.data import DataLoader
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.folder import default_loader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import numpy as np

# from torchsummary import summary

def get_image_albumentations():
    # Image transformations

    image_transforms = {
        # Train uses data augmentation
        'train':
            A.Compose([
                A.RandomRotate90(p=0.4),
                A.HorizontalFlip(p=0.3),
                A.VerticalFlip(p=0.3),
                A.IAAAffine(always_apply=True, shear=0.2, translate_percent=0.2, p=0.3),
                A.ShiftScaleRotate(scale_limit=0.1,
                                   rotate_limit=20,
                                   shift_limit=0.12,
                                   p=0.3,
                                   border_mode=cv2.BORDER_CONSTANT,
                                   value=0,
                                   ),
                A.GaussianBlur(p=0.3),
                A.RandomRain(p=0.3),
                A.Resize(224, 244),
                A.Normalize(),
                ToTensorV2()
            ]),
        # Validation does not use augmentation
        'val':
            A.Compose([
                A.Resize(224, 244),
                A.Normalize(),
                ToTensorV2()
            ]),
        # Test does not use augmentation
        'test':
            A.Compose([
                A.Resize(224, 244),
                A.Normalize(),
                ToTensorV2()
            ]),
    }
    return image_transforms


def get_image_augmentation():
    # Image transformations

    image_transforms = {
        # Train uses data augmentation
        'train':
            transforms.Compose([
                # transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(10),
                # transforms.RandomRotation((90,92)),
                # transforms.ColorJitter(),
                transforms.RandomHorizontalFlip(),  # test
                transforms.Resize(size=(224, 224)),
                # Image net standards; Always resize right before ToTensor()
                transforms.ToTensor(),  # Auto normalize image to range of 0, 1
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])  # Imagenet standards
            ]),
        # Validation does not use augmentation
        'val':
            transforms.Compose([
                transforms.Resize(size=(224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        # Test does not use augmentation
        'test':
            transforms.Compose([
                transforms.Resize(size=(224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
    }
    return image_transforms


def create_dataset(images_path: str, batch_size: int, image_transforms, val_split: float = 0.2):
    # datasets load and return single data point (image, record, ...)
    # Datasets from each folder
    num_worker = 0
    train_on_gpu = cuda.is_available()
    if train_on_gpu:
        gpu_count = cuda.device_count()
        # For remote debugging, set num_worker = 0
        num_worker = 4 * gpu_count

    # Load data and split into train and valid sets
    train_images, val_images, labels, class_to_idx = get_data(images_path, val_split)

    data = {
        'train':
            MultiLabelDataset(root=images_path, samples=train_images, classes=labels,
                              class_to_idx=class_to_idx, transform=image_transforms['train']),
        'val':
            MultiLabelDataset(root=images_path, samples=val_images, classes=labels,
                              class_to_idx=class_to_idx, transform=image_transforms['val'])
    }
    print(f'images_path: {images_path}')
    print(f'Number of train images: {len(data["train"])}')
    print(f'Number of val images: {len(data["val"])}')
    print(f"{len(data['val'].classes)} classes: {data['val'].classes}")

    # Dataloader handle batching and parallelism
    data_loaders = {
        'train': DataLoader(data['train'], batch_size=batch_size, shuffle=True,
                               num_workers=num_worker, pin_memory=True),
        'val': DataLoader(data['val'], batch_size=batch_size, shuffle=True,
                          num_workers=num_worker, pin_memory=True)
    }
    return data, data_loaders


def get_data(root: str, val_split: float = 0.2):
    # Get list of all labels
    all_folders = next(os.walk(root))[1]
    labels = []
    label_dict = dict()
    for folder in all_folders:
        temp = []
        folder_labels = folder.split('+')
        for folder_label in folder_labels:
            if folder_label not in labels:
                labels.append(folder_label)
            temp.append(labels.index(folder_label))

        label_dict[folder] = temp.copy()

    len_labels = len(labels)

    # Get dict to transform folder into 1 hot labels
    for folder, label_indexes in label_dict.items():
        label_dict[folder] = torch.zeros(len_labels)
        for i in label_indexes:
            label_dict[folder][i] = 1

    # Split data into train and valid sets
    train_images = []
    val_images = []
    img_extensions = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')
    root_data_path = os.path.expanduser(root)
    for folder in sorted(os.listdir(root_data_path)):
        folder_dir = os.path.join(root_data_path, folder)
        folder_images = []
        if not os.path.isdir(folder_dir):
            continue

        # count = 0
        for file in sorted(os.listdir(folder_dir)):
            # if count > 100:
            #     break
            # count += 1
            filename = os.fsdecode(file)
            file_dir = os.path.join(folder_dir, filename)
            if file_dir.lower().endswith(img_extensions):
                folder_images.append((file_dir, label_dict[folder]))

        random.shuffle(folder_images)
        val_num_images = int(len(folder_images) * val_split)
        val_images.extend(folder_images[:val_num_images])
        train_images.extend(folder_images[val_num_images:])

    class_to_idx = {labels[idx]: idx for idx in range(len_labels)}
    # idx_to_class = {v: k for k, v in class_to_idx.items()}

    return train_images, val_images, labels, class_to_idx


class MultiLabelDataset(VisionDataset):
    """
    Very basic dataset, need 'get_data' to run
    """

    def __init__(self, root: str, samples: list, classes: list, class_to_idx: dict,
                 loader=default_loader, transform=None, target_transform=None):
        super(MultiLabelDataset, self).__init__(root, transform=transform,
                                                target_transform=target_transform)

        # train_samples, val_samples, classes, class_to_idx = self._get_data()
        if len(samples) == 0:
            raise (RuntimeError(f"Found 0 image in {root} "))

        self.loader = loader
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.imgs = samples
        self.targets = [s[1] for s in samples]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            # sample = self.transform(sample)
            sample = np.array(sample)
            transformed = self.transform(image = sample)
            sample = transformed["image"]
        if self.target_transform is not None:
            target = self.target_transform(target)


        # x = sample.permute(1,2,0)
        # x = x.cpu().numpy()
        # plt.imshow(x)
        # plt.show()
        return sample, target

    def __len__(self):
        return len(self.samples)


####################################################################################
# For valid dataset
def create_valid_dataset(images_path: str, batch_size: int, image_transforms,
                         class_to_idx: dict):
    # datasets load and return single data point (image, record, ...)
    # Datasets from each folder
    num_worker = 0
    train_on_gpu = cuda.is_available()
    if train_on_gpu:
        gpu_count = cuda.device_count()
        # For remote debugging, set num_worker = 0
        num_worker = 4 * gpu_count

    # Load data and split into train and valid sets

    data = {
        'val': MultiLabelValidDataset(root=images_path, class_to_idx=class_to_idx,
                                      transform=image_transforms['val'])
    }
    print(f'images_path: {images_path}')
    print(f'Number of valid images: {len(data["val"])}')
    print(f"{len(class_to_idx)} classes: {list(class_to_idx.keys())}")

    # Dataloader handle batching and parallelism
    data_loaders = {
        'val': DataLoader(data['val'], batch_size=batch_size, shuffle=True,
                          num_workers=num_worker, pin_memory=True)
    }
    return data, data_loaders


class MultiLabelValidDataset(VisionDataset):
    """
    Very basic dataset, need 'get_data' to run
    """

    def __init__(self, root: str, class_to_idx: dict,
                 loader=default_loader, transform=None, target_transform=None):
        super(MultiLabelValidDataset, self).__init__(root, transform=transform,
                                                     target_transform=target_transform)

        self.class_to_idx = class_to_idx
        samples = self.__get_data__()
        if len(samples) == 0:
            raise RuntimeError(f"Found 0 image in {root} ")

        self.loader = loader
        self.classes = class_to_idx.keys()
        self.samples = samples
        self.imgs = samples
        self.targets = [s[1] for s in samples]

    def __get_data__(self):
        # Get list of all labels
        all_folders = next(os.walk(self.root))[1]
        label_dict = dict()
        class_to_idx = self.class_to_idx
        class_to_idx_keys = class_to_idx.keys()
        for folder in all_folders:
            label_dict[folder] = []
            folder_labels = folder.split('+')
            for folder_label in folder_labels:
                if folder_label not in class_to_idx_keys:
                    raise RuntimeError(f'Label {folder_label} is not in the training class list.')
                label_dict[folder].append(class_to_idx[folder_label])

        len_labels = len(class_to_idx)

        # Get dict to transform folder into 1 hot labels
        for folder, label_indexes in label_dict.items():
            label_dict[folder] = torch.zeros(len_labels)
            for i in label_indexes:
                label_dict[folder][i] = 1

        # Get valid set
        test_images = []
        img_extensions = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')
        root_data_path = os.path.expanduser(self.root)
        for folder in sorted(os.listdir(root_data_path)):
            folder_dir = os.path.join(root_data_path, folder)
            if not os.path.isdir(folder_dir):
                continue

            # count = 0
            for file in sorted(os.listdir(folder_dir)):
                # if count > 100:
                #     break
                # count += 1
                filename = os.fsdecode(file)
                file_dir = os.path.join(folder_dir, filename)
                if file_dir.lower().endswith(img_extensions):
                    test_images.append((file_dir, label_dict[folder]))

        random.shuffle(test_images)

        return test_images

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return path, sample, target

    def __len__(self):
        return len(self.samples)


####################################################################################
# For test dataset
def create_test_dataset(images_path: str, batch_size: int, image_transforms,
                        group_to_class_tensor: dict):
    # datasets load and return single data point (image, record, ...)
    # Datasets from each folder
    num_worker = 0
    train_on_gpu = cuda.is_available()
    if train_on_gpu:
        gpu_count = cuda.device_count()
        # For remote debugging, set num_worker = 0
        num_worker = 4 * gpu_count

    # Load data and split into train and valid sets

    data = {
        'test': MultiLabelTestDataset(root=images_path, group_to_class_tensor=group_to_class_tensor,
                                      transform=image_transforms['test'])
    }
    classes = list(group_to_class_tensor.keys())
    print(f'images_path: {images_path}')
    print(f'Number of test images: {len(data["test"])}')
    print(f"{len(classes)} classes: {classes}")

    # Dataloader handle batching and parallelism
    data_loaders = {
        'test': DataLoader(data['test'], batch_size=batch_size, shuffle=True,
                           num_workers=num_worker, pin_memory=True)
    }
    return data, data_loaders


class MultiLabelTestDataset(VisionDataset):
    def __init__(self, root: str, group_to_class_tensor: dict,
                 loader=default_loader, transform=None, target_transform=None):
        super(MultiLabelTestDataset, self).__init__(root, transform=transform,
                                                    target_transform=target_transform)

        self.classes = list(group_to_class_tensor.keys())
        samples = self.__get_data__(group_to_class_tensor)
        if len(samples) == 0:
            raise (RuntimeError(f"Found 0 image in {root} "))

        self.loader = loader
        self.samples = samples
        self.imgs = samples
        self.targets = [s[1] for s in samples]

    def __get_data__(self, group_to_class_tensor: dict):
        # Get list of all labels

        # Get test set
        test_images = []
        img_extensions = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')
        root_data_path = os.path.expanduser(self.root)
        for folder in sorted(os.listdir(root_data_path)):
            folder_dir = os.path.join(root_data_path, folder)
            if not os.path.isdir(folder_dir):
                continue

            # count = 0
            for file in sorted(os.listdir(folder_dir)):
                # if count > 100:
                #     break
                # else:count += 1
                filename = os.fsdecode(file)
                file_dir = os.path.join(folder_dir, filename)
                if file_dir.lower().endswith(img_extensions):
                    test_images.append((file_dir, group_to_class_tensor[folder]))

        random.shuffle(test_images)

        return test_images

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (path, group, sample, target) where target is class_index of the target class,
            path is image path.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            raise RuntimeError(f'Target is a string and cannot be transformed')

        return path, sample, target

    def __len__(self):
        return len(self.samples)
