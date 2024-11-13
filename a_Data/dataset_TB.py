import os
import PIL.Image
import numpy as np
import pandas as pd
import torch
import albumentations as A
from torch.utils.data import Dataset
from albumentations.pytorch.transforms import ToTensorV2
from PIL import Image


class Dataset_TB(Dataset):
    def __init__(self, data_dir, csv_path, phase, img_size):
        img_folder = data_dir + '/img_224/'
        if phase == 'train':
            mouse_folder = data_dir + '/map_224/'
            mouse_paths = [mouse_folder + path for path in os.listdir(mouse_folder)]
            mouse_paths.sort()
            self.mouse_paths = mouse_paths

        img_paths = [img_folder + path for path in os.listdir(img_folder)]
        self.csv = pd.read_csv(csv_path, usecols=[0, 1])

        new_img_paths = []
        for img_path in img_paths:
            name = img_path.split('/')[-1]
            class_id = self.csv.loc[self.csv.image_id == name].class_id.iloc[0]
            if class_id == 'Normal':
                new_img_paths.append(img_path)
            elif class_id == 'TB':
                new_img_paths.append(img_path)
            elif class_id == 'Pneumonia':
                continue
        img_paths = new_img_paths

        img_paths.sort()

        self.img_paths = img_paths
        self.phase = phase
        self.transform = get_transform(phase, img_size)

    def __getitem__(self, index):
        if self.phase == 'train':
            img_path, mouse_path = self.img_paths[index], self.mouse_paths[index]
            img = np.array(Image.open(img_path))
            mouse = np.array(Image.open(mouse_path))
            transformed = self.transform(image=img, mask=mouse)
            img = transformed["image"] / 255.
            mouse = transformed["mask"] / 255.
            mouse = mouse.unsqueeze(0)
        else:
            img_path = self.img_paths[index]
            img = np.array(Image.open(img_path))
            transformed = self.transform(image=img)
            img = transformed["image"] / 255.
            mouse = torch.randn([1, 224, 224])

        # img = (img - 0.5) / 0.5
        # img = (img - torch.mean(img)) / torch.std(img)

        name = img_path.split('/')[-1]
        class_id = self.csv.loc[self.csv.image_id == name].class_id.iloc[0]
        cls = -1
        if class_id == 'Normal':
            cls = 0
        elif class_id == 'TB':
            cls = 1
        elif class_id == 'Pneumonia':
            cls = 2
        return img, mouse, cls, img_path

    def __len__(self):
        return len(self.img_paths)


def get_transform(phase, img_size):
    if phase == 'train':
        return A.Compose(
            [
                # A.Resize(img_size, img_size),
                # A.ShiftScaleRotate(shift_limit=0, scale_limit=0, rotate_limit=15, p=0.5),
                # A.RandomGamma(),
                # A.RandomBrightnessContrast(),
                # ToTensorV2(),

                # A.Resize(img_size, img_size),
                A.HorizontalFlip(),
                A.ShiftScaleRotate(rotate_limit=15),
                A.RandomBrightnessContrast(),
                A.RandomGamma(),
                ToTensorV2(),

                # A.Resize(img_size, img_size),
                # ToTensorV2(),
            ])
    else:
        return A.Compose(
            [
                A.Resize(img_size, img_size),
                ToTensorV2()
            ])


if __name__ == '__main__':
    imgs_folder = 'TB-Mouse/train/img/'
    imgs_names = os.listdir(imgs_folder)
    imgs_paths = [imgs_folder + img_name for img_name in imgs_names]
    new_folder_img = 'TB-Mouse/train/img_224/'
    new_folder_gaze = 'TB-Mouse/train/map_224/'

    for img_path in imgs_paths:
        gaze_path = img_path.split('img')[0] + 'mouse_heatmap_1_and_2' + img_path.split('img')[1]
        img = np.array(Image.open(img_path))

        if os.path.exists(gaze_path):
            print(1)
            gaze = np.array(Image.open(gaze_path))
            transform = A.Compose([A.Resize(224, 224),])
            transformed = transform(image=img, mask=gaze)
            img = transformed["image"]
            gaze = transformed["mask"]
            img = Image.fromarray(img).save(new_folder_img + img_path.split('/')[-1])
            p = new_folder_gaze + gaze_path.split('/')[-1]
            gaze = Image.fromarray(gaze).save(new_folder_gaze + gaze_path.split('/')[-1])
        else:
            transform = A.Compose([A.Resize(224, 224)])
            transformed = transform(image=img)
            img = transformed["image"]
            img = Image.fromarray(img).save(new_folder_img + img_path.split('/')[-1])
        # break