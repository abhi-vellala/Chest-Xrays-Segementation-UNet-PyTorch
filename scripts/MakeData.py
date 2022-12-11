import os
import pandas as pd
import torch
import torchvision
import numpy as np
from PIL import Image

def make_paths_table(data_path, save_path):
    images = []
    masks_right = []
    masks_left = []
    for path, dirs, files in os.walk(data_path):
        for file in files:
            if 'masks' in path:
                if file.endswith(".gif") and "right" in path:
                    masks_right.append(os.path.join(path, file))
                if file.endswith(".gif") and "left" in path:
                    masks_left.append(os.path.join(path, file))
            if 'images' in path and file.endswith(".bmp"):
                images.append(os.path.join(path, file))

    images.sort(key=lambda x: '{0:0>8}'.format(x.split("/")[-1]).lower())
    masks_right.sort(key=lambda x: '{0:0>8}'.format(x.split("/")[-1]).lower())
    masks_left.sort(key=lambda x: '{0:0>8}'.format(x.split("/")[-1]).lower())

    df = pd.DataFrame()
    df['images'] = images
    df['masks_right_lung'] = masks_right
    df['masks_left_lung'] = masks_left
    saved_file_path = os.path.join(save_path, "images_masks_path.xlsx")
    df.to_excel(saved_file_path, index=False)
    return df, saved_file_path


class BuildData(torch.utils.data.Dataset):

    def __init__(self, images_folder_list, masks_folder_list, transforms=None, class_type="single"):
        self.images_folder_list = images_folder_list
        self.masks_folder_list = masks_folder_list
        self.transforms = transforms
        self.class_type = class_type

    def __getitem__(self, idx):
        image_name = self.images_folder_list[idx]
        right_mask_name = self.masks_folder_list[idx][0]
        left_mask_name = self.masks_folder_list[idx][1]
        image = Image.open(image_name).convert("P")
        right_mask = Image.open(right_mask_name)
        left_mask = Image.open(left_mask_name)

        if self.transforms is not None:
            left_mask, right_mask = self.transforms((left_mask, right_mask))

        left_mask = torch.tensor(np.array(left_mask))
        right_mask = torch.tensor(np.array(right_mask))
        if self.class_type == 'single':
            mask = torch.add(right_mask,left_mask)
            mask = (torch.tensor(mask.numpy()) == 255).long() # mask[mask == 255] = 1
        if self.class_type == "multi":
            left_mask = (torch.tensor(left_mask.numpy()) == 255).long()
            right_mask = (torch.tensor(right_mask.numpy()) == 255).long()
            right_mask[right_mask == 1.0] = 2.0
            mask = torch.tensor((np.array(right_mask) | np.array(left_mask)))
            # mask = torch.tensor(np.stack([np.array(right_mask, dtype=np.uint8), np.array(left_mask, dtype=np.uint8)],
            #                              axis=-1))

        image = torchvision.transforms.functional.to_tensor(image) - 0.5
        return image, mask

    def __len__(self):
        return len(self.images_folder_list)


class Resize():
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        left_mask, right_mask = sample
        right_mask = torchvision.transforms.functional.resize(right_mask, self.output_size,
                                                              torchvision.transforms.InterpolationMode.NEAREST)
        left_mask = torchvision.transforms.functional.resize(left_mask, self.output_size,
                                                             torchvision.transforms.InterpolationMode.NEAREST)

        return right_mask, left_mask






if __name__ == '__main__':

    data_path = "/Users/abhi/Desktop/Projects/UCPHTasks/Task1-Segmentation/data/Lung segmentation"
    save_path = "/Users/abhi/Desktop/Projects/UCPHTasks/Task1-Segmentation/data/"
    make_paths_table(data_path, save_path)