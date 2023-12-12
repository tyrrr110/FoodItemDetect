import os
import json

import PIL
import torch
from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder, default_loader

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform


class INatDataset(ImageFolder):
    def __init__(self, root, train=True, year=2018, transform=None, target_transform=None,
                 category='name', loader=default_loader):
        self.transform = transform
        self.loader = loader
        self.target_transform = target_transform
        self.year = year
        # assert category in ['kingdom','phylum','class','order','supercategory','family','genus','name']
        path_json = os.path.join(root, f'{"train" if train else "val"}{year}.json')
        with open(path_json) as json_file:
            data = json.load(json_file)

        with open(os.path.join(root, 'categories.json')) as json_file:
            data_catg = json.load(json_file)

        path_json_for_targeter = os.path.join(root, f"train{year}.json")

        with open(path_json_for_targeter) as json_file:
            data_for_targeter = json.load(json_file)

        targeter = {}
        indexer = 0
        for elem in data_for_targeter['annotations']:
            king = []
            king.append(data_catg[int(elem['category_id'])][category])
            if king[0] not in targeter.keys():
                targeter[king[0]] = indexer
                indexer += 1
        self.nb_classes = len(targeter)

        self.samples = []
        for elem in data['images']:
            cut = elem['file_name'].split('/')
            target_current = int(cut[2])
            path_current = os.path.join(root, cut[0], cut[2], cut[3])

            categors = data_catg[target_current]
            target_current_true = targeter[categors[category]]
            self.samples.append((path_current, target_current_true))

    # __getitem__ and __len__ inherited from ImageFolder

N = 224
IMAGE_PATH = '../Ingredient-Guided-Cascaded-Multi-Attention-Network-for-Food-Recognition-master/data'   # the path of images folder
DIR_TRAIN_IMAGES_INGREDIENT   = '../Ingredient-Guided-Cascaded-Multi-Attention-Network-for-Food-Recognition-master/metadata/class_ingredient.txt'
def My_loader(path):
    return PIL.Image.open(path).convert('RGB')


class YOLOSet(torch.utils.data.Dataset):
    def __init__(self, root_dir, is_Train=True, transform=None):
        self.root_dir = root_dir
        self.is_train = is_Train
        self.transform = transform
        self.num_classes = 9

        # 根据是训练集还是测试集选择相应的目录
        self.data_path = os.path.join(self.root_dir, 'train' if self.is_train else 'test')

        # 获取所有图片的文件名
        self.images = os.listdir(os.path.join(self.data_path, 'images'))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # 加载图片
        img_name = self.images[idx]
        img_path = os.path.join(self.data_path, 'images', img_name)
        image = PIL.Image.open(img_path)

        # 加载并处理标签
        label_path = os.path.join(self.data_path, 'labels', img_name.replace('.png', '.txt'))
        label = self._load_label(label_path)

        # 应用转换（如果有）
        if self.transform:
            image = self.transform(image)

        return image, label

    def _load_label(self, label_path):
        # 初始化一个全为零的张量
        label_tensor = torch.zeros(self.num_classes, dtype=torch.float)

        with open(label_path, 'r') as file:
            for line in file:
                # 获取标签类别
                class_id = int(line.split()[0])
                label_tensor[class_id] = 1

        return label_tensor

class DishDataset(torch.utils.data.Dataset):

    def __init__(self, is_Train=True, transform=None):
        self.root_dir = IMAGE_PATH
        self.transform = transform
        self.data = []
        self.labels = []
        self._create_dataset()
        all_data = []
        all_labels = []

        def _create_dataset(self):
            subdirs = sorted([d for d in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, d))])
            for label, subdir in enumerate(subdirs):
                subdir_path = os.path.join(self.root_dir, subdir)
                for filename in os.listdir(subdir_path):
                    if filename.endswith('.jpg'):
                        filepath = os.path.join(subdir_path, filename)
                        all_data.append(filepath)
                        all_labels.append(label)
            split_len = int(0.9 * len(all_labels))
            if is_Train:
                self.data = all_data[:split_len]
                self.labels = all_labels[:split_len]
                # self.data = all_data[:800]
                # self.labels = all_labels[:800]
            else:
                self.data = all_data[split_len:]
                self.labels = all_labels[split_len:]

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            while True:
                img_path = self.data[idx]
                label = self.labels[idx]
                try:
                    image = PIL.Image.open(img_path).convert('RGB')
                except OSError as e:
                    print(f"Skipping corrupted image {img_path}: {e}")
                    index = (index + 1) % len(self.data)  # Move to the next image in the dataset
                    continue  # Skip the rest of this loop iteration and try again

                if self.transform:
                    image = self.transform(image)
                    label = torch.Tensor(label)

                return image, label

def build_dataset(is_train, args):
    train_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),  # default value is 0.5
        transforms.Resize((N, N)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    if args.data_set == 'YOLO':
        dataset = YOLOSet(args.data_path, is_Train=is_train, transform=train_transforms)
        nb_classes = 9
    elif args.data_set == 'IMNET':
        dataset = DishDataset(is_Train=is_train,transform= build_transform(is_train, args))
        nb_classes = 70


    return dataset, nb_classes


def build_transform(is_train, args):
    resize_im = args.input_size > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        size = int((256 / 224) * args.input_size)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)
