# -*- coding: UTF-8 -*-

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transfroms
import matplotlib.pyplot as plt
import os
import PIL.Image as Image
import PIL
import cv2
import numpy as np
import numpy.random as npr
from utils import iouFun,nms,readTag,writeTag,processImage
from MyEnum import MyEnum


class WiderFaceDataset(Dataset):
    def __init__(self, images_folder, ground_truth_file, transform=None, target_transform=None):
        super(WiderFaceDataset, self).__init__()
        self.images_folder = images_folder
        self.ground_truth_file = ground_truth_file
        self.images_name_list = []
        self.ground_truth = []
        with open(ground_truth_file, 'r') as f:
            for i in f:
                self.images_name_list.append(i.rstrip())
                self.ground_truth.append(i.rstrip())

        self.images_name_list = list(filter(lambda x: x.endswith('.jpg') or x.endswith('.bmp'),
                                       self.images_name_list))

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.images_name_list)

    def __getitem__(self, index):
        image_name = self.images_name_list[index]
        # 查找文件名
        loc = self._search(image_name)
        # 解析人脸个数
        face_nums = int(self.ground_truth[loc + 1])
        # 读取矩形框
        rects = []
        a=[]
        for i in range(loc + 2, loc + 2 + face_nums):
            line = self.ground_truth[i]
            if line.split(' ')[4]=='2':
                return image_name
                # print("image_name:",image_name)
            
            # x, y, w, h = line.split(' ')[:4]
            # x, y, w, h = list(map(lambda k: int(k), [x, y, w, h]))
            # rects.append([x, y, w, h])
        # b=list(set(a))
        # print("b:",len(b))
        return 1
        # # 图像
        # image = PIL.Image.open(os.path.join(self.images_folder, image_name))

        # short_image_name=image_name.split('/',1)[1]
        # if self.transform:
        #     image = self.transform(image)

        # if self.target_transform:
        #     rects = list(map(lambda x: self.target_transform(x), rects))

        # return {'image': image, 'label': rects,'image_name_detail': short_image_name , 'image_name': os.path.join(self.images_folder, image_name) }

    def _search(self, image_name):
        for i, line in enumerate(self.ground_truth):
            if image_name == line:
                return i
if __name__ == '__main__':
    images_folder = '/media/chinasilva/编程资料/deeplearning/datasets/wider_face/WIDER_train/images'
    save_images_folder = ''#/mnt/my_wider_face_val'
    save_tag_path = ''#/mnt/my_wider_face_val/'
    
    ground_truth_file = open('/media/chinasilva/编程资料/deeplearning/datasets/wider_face/wider_face_split/wider_face_train_bbx_gt.txt', 'r')

    dataset = WiderFaceDataset(images_folder=images_folder,
                                ground_truth_file='/media/chinasilva/编程资料/deeplearning/datasets/wider_face/wider_face_split/wider_face_train_bbx_gt.txt',
                                transform=transfroms.ToTensor(),
                                target_transform=lambda x: torch.tensor(x))
    for i, sample in enumerate(dataset):
        # a.append(sample)
        if sample!=1:
            print(sample)
    # print(a)
            
                