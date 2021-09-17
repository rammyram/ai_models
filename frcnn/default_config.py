import albumentations as A
import torch, cv2

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Datset paths
images_dir = None #"/home/sohoa1/rammy/datasets/2021/object_detection/fire_extinguisher/frcnn/all_images"
json_file_train = None #"/home/sohoa1/rammy/datasets/2021/object_detection/fire_extinguisher/frcnn/all_images_fire_meter_coco_train.json"
json_file_val = None #"/home/sohoa1/rammy/datasets/2021/object_detection/fire_extinguisher/frcnn/all_images_fire_meter_coco_val.json"
json_file_test = None #"/home/sohoa1/rammy/datasets/2021/object_detection/fire_extinguisher/frcnn/all_images_fire_meter_coco_test.json"
json_eval_file = json_file_test


# Save all std out to log file
project = "Project name"
pre_trained_weights = "None" # Path to model pretrained weigts for transfer learning
model_checkpoint = None  # if None then pretrained weights are used
trained_weights = None # if None then pretrained weights are used
best_model_path = f"{project}/{project}_best.pt"
last_model_path = f"{project}/{project}_last.pt"

eval_weights = best_model_path
log_path = f"{project}/log.txt"
eval_log_path = f"{project}/eval_log.txt"


# classes
num_classes = 1
num_classes += 1 # adding background


# Hyper parameters
batch_size = 5
num_workers = 0
lr = 0.0001
momentum = 0.9
weight_decay = 0.0003
scheduler_step_size = 10
scheduler_gamma = 0.1
num_epochs = 50
early_stop_limit = 20


# Augmentations :
# Note : Using augmentations like scaling or rotation cold lead to loss of masks at edges leading to reduced
# target classes and dataset may throw error, if using such augmentation handle accordingly in dataset
train_aug = [
            A.HorizontalFlip(p=0.3),
            A.VerticalFlip(p=0.3),
            A.ShiftScaleRotate(scale_limit=0.2,
                                       rotate_limit=30,
                                       shift_limit=0.2,
                                       p=0.3,
                                       border_mode=cv2.BORDER_CONSTANT,
                                       value=0,
                               ),
            A.GaussianBlur(p=0.3),
            A.HueSaturationValue( p=0.3),
            A.RandomRain(p=0.3)
        ]
test_aug = []
eval_aug = test_aug