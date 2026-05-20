from torch.utils.data import Dataset
from PIL import Image
# import cv2
import os
import numpy as np
from glob import glob
from torchvision import transforms, datasets
from torch.utils.data.dataset import Dataset
import torch
import math
import torch.utils.data as data
import random
NUM_DATASET_WORKERS = 8
SCALE_MIN = 0.75
SCALE_MAX = 0.95


class HR_image(Dataset):
    files = {"train": "train", "test": "test", "val": "validation"}

    def __init__(self, config, data_dir):
        self.imgs = []
        for dir in data_dir:
            self.imgs += glob(os.path.join(dir, '*.jpg'))
            self.imgs += glob(os.path.join(dir, '*.png'))
        _, self.im_height, self.im_width = config.image_dims
        self.crop_size = self.im_height
        self.image_dims = (3, self.im_height, self.im_width)
        self.transform = self._transforms()

    def _transforms(self,):
        """
        Up(down)scale and randomly crop to `crop_size` x `crop_size`
        """
        transforms_list = [
            # transforms.RandomCrop((self.im_height, self.im_width)),
            transforms.RandomCrop((256, 256)),
            transforms.ToTensor()]

        return transforms.Compose(transforms_list)

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        img = Image.open(img_path)
        img = img.convert('RGB')
        transformed = self.transform(img)
        return transformed

    def __len__(self):
        return len(self.imgs)


class Datasets(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.imgs = []
        for dir in self.data_dir:
            self.imgs += glob(os.path.join(dir, '*.jpg'))
            self.imgs += glob(os.path.join(dir, '*.png'))
        self.imgs.sort()


    def __getitem__(self, item):
        image_ori = self.imgs[item]
        name = os.path.basename(image_ori)
        image = Image.open(image_ori).convert('RGB')
        self.im_height, self.im_width = image.size
        if self.im_height % 128 != 0 or self.im_width % 128 != 0:
            self.im_height = self.im_height - self.im_height % 128
            self.im_width = self.im_width - self.im_width % 128
        self.transform = transforms.Compose([
            transforms.CenterCrop((self.im_width, self.im_height)),
            transforms.ToTensor()])
        img = self.transform(image)
        return img, name
    def __len__(self):
        return len(self.imgs)

class CIFAR10(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.len = dataset.__len__()

    def __getitem__(self, item):
        return self.dataset.__getitem__(item % self.len)

    def __len__(self):
        return self.len * 10


class ImageNetDataset(Dataset):
    """
    ImageNet1k dataset with configurable train/test split
    Default: 99/1 split for fine-tuning scenarios
    Supports both training and testing modes
    """
    def __init__(self, root_dir, split='train', transform=None, train_split=0.9, seed=42):
        """
        Args:
            root_dir: Path to imagenet1k folder with class subfolders
            split: 'train' or 'test'
            transform: Transformations to apply
            train_split: Fraction of data for training (default: 0.9)
            seed: Random seed for reproducible split
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.train_split = train_split
        
        # Get all class folders (sorted for consistency)
        self.class_folders = sorted([d for d in os.listdir(root_dir) 
                                     if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.class_folders)}
        
        # Collect all images with labels
        self.samples = []
        for cls_idx, cls_name in enumerate(self.class_folders):
            cls_path = os.path.join(root_dir, cls_name)
            images = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPEG', '*.JPG']:
                images.extend(glob(os.path.join(cls_path, ext)))
            
            # Sort for consistency
            images.sort()
            
            # Split images for this class
            random.Random(seed).shuffle(images)
            n_train = int(len(images) * train_split)
            
            if split == 'train':
                class_samples = images[:n_train]
            else:  # test
                class_samples = images[n_train:]
            
            for img_path in class_samples:
                self.samples.append((img_path, cls_idx))
        
        print(f"ImageNet {split}: {len(self.samples)} images across {len(self.class_folders)} classes")
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a black image as fallback
            image = Image.new('RGB', (224, 224), color='black')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def __len__(self):
        return len(self.samples)
    
    def get_class_name(self, idx):
        """Get class name from index"""
        return self.class_folders[idx]


def get_loader(args, config):
    if args.trainset == 'DIV2K':
        train_dataset = HR_image(config, config.train_data_dir)
        test_dataset = Datasets(config.test_data_dir)
        # test_dataset = HR_image(config, config.test_data_dir)
    elif args.trainset == 'CIFAR10':
        dataset_ = datasets.CIFAR10
        if config.norm is True:
            transform_train = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        else:
            transform_train = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()])

            transform_test = transforms.Compose([
                transforms.ToTensor()])
        train_dataset = dataset_(root=config.train_data_dir,
                                 train=True,
                                 transform=transform_train,
                                 download=False)

        test_dataset = dataset_(root=config.test_data_dir,
                                train=False,
                                transform=transform_test,
                                download=False)

        train_dataset = CIFAR10(train_dataset)
    
    elif args.trainset == 'ImageNet':
        # ImageNet with 90/10 split
        if config.norm is True:
            transform_train = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])])
            
            transform_test = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])])
        else:
            transform_train = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()])
            
            transform_test = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor()])
        
        # Use the imagenet1k folder path
        imagenet_root = config.train_data_dir[0] if isinstance(config.train_data_dir, list) else config.train_data_dir
        
        train_dataset = ImageNetDataset(
            root_dir=imagenet_root,
            split='train',
            transform=transform_train,
            train_split=0.99,  # 99% for training, 1% for testing (fine-tuning scenario)
            seed=42
        )
        
        test_dataset = ImageNetDataset(
            root_dir=imagenet_root,
            split='test',
            transform=transform_test,
            train_split=0.99,  # 99% for training, 1% for testing (fine-tuning scenario)
            seed=42
        )

    else:
        train_dataset = Datasets(config.train_data_dir)
        test_dataset = Datasets(config.test_data_dir)

    def worker_init_fn_seed(worker_id):
        seed = 10
        seed += worker_id
        np.random.seed(seed)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               num_workers=NUM_DATASET_WORKERS,
                                               pin_memory=True,
                                               batch_size=config.batch_size,
                                               worker_init_fn=worker_init_fn_seed,
                                               shuffle=True,
                                               drop_last=True)
    if args.trainset == 'CIFAR10':
        test_loader = data.DataLoader(dataset=test_dataset,
                                  batch_size=1024,
                                  shuffle=False)
    elif args.trainset == 'ImageNet':
        test_loader = data.DataLoader(dataset=test_dataset,
                                  batch_size=64,  # Smaller batch for larger images
                                  shuffle=False,
                                  num_workers=NUM_DATASET_WORKERS,
                                  pin_memory=True)
    else:
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=1,
                                              shuffle=False)

    return train_loader, test_loader

