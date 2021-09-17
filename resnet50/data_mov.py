import os
import io
import urllib.request
from torchvision.datasets.vision import VisionDataset
from PIL import Image
from torchvision import transforms
from torch import cuda
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import uuid
from pathlib import Path

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class MultiLabelTestDataset(VisionDataset):
    def __init__(self, input_images: list,transform=None, target_transform=None):
        super(MultiLabelTestDataset, self).__init__(None, transform=transform,
                                                    target_transform=target_transform)

        self.samples = [ img for img in input_images if  Path(img).suffix in [".jpg", ".jpeg", ".png"]]

        print("Total images provided ", len(input_images))
        print("Valid images ", len(self.samples))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns: image_arr
        """
        try:

            path = self.samples[index]
            if isinstance(path, str) and path.startswith("ftp"):
                image_bytes = self.fetch_data(path)
                sample = Image.open(io.BytesIO(image_bytes))
                sample.convert('RGB')

            elif isinstance(path, str) and os.path.exists(path):
                img0 = Image.open(path).convert("RGB")  # RGB

            elif isinstance(path, (np.ndarray, np.generic)):
                img0 = Image.fromarray(path).convert("RGB")
                path = get_unique_file_name(extension="jpg") # get a unique file name if input is array
            else:
                raise Exception('ERROR: input %s does not exist' % path)


            if self.transform is not None:
                image_arr = self.transform(img0)
            if self.target_transform is not None:
                pass

            return image_arr

        except Exception:
            print(path)
            raise Exception


    def __len__(self):
        return len(self.samples)


    @staticmethod
    def fetch_data(url):
        ftp_path = url.replace('#', '%23')
        return urllib.request.urlopen(ftp_path).read()

def create_dataset(input_images: list, image_transforms,batch_size: int = 5):
    # datasets load and return single data point (image, record, ...)
    # Datasets from each folder
    num_worker = 0
    train_on_gpu = cuda.is_available()
    if train_on_gpu:
        gpu_count = cuda.device_count()
        # For remote debugging, set num_worker = 0
        num_worker = 4 * gpu_count * 0
        # num_worker = 0

    # create Dataset
    data = MultiLabelTestDataset(input_images,transform=image_transforms)

    # Dataloader handle batching and parallelism
    data_loaders = DataLoader(data, batch_size=batch_size, shuffle=False,
                              num_workers=num_worker, pin_memory=True)

    return data, data_loaders


def get_unique_file_name(extension:str):
    unique_file_name = str(uuid.uuid4()) + f".{extension}"
    return unique_file_name