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
        for i in range(loc + 2, loc + 2 + face_nums):
            line = self.ground_truth[i]
            x, y, w, h = line.split(' ')[:4]
            x, y, w, h = list(map(lambda k: int(k), [x, y, w, h]))
            rects.append([x, y, w, h])

        # 图像
        image = PIL.Image.open(os.path.join(self.images_folder, image_name))

        short_image_name=image_name.split('/',1)[1]
        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            rects = list(map(lambda x: self.target_transform(x), rects))

        return {'image': image, 'label': rects,'image_name_detail': short_image_name , 'image_name': os.path.join(self.images_folder, image_name) }

    def _search(self, image_name):
        for i, line in enumerate(self.ground_truth):
            if image_name == line:
                return i
if __name__ == '__main__':
    images_folder = '/media/chinasilva/编程资料/deeplearning/datasets/wider_face/WIDER_train/images'
    save_images_folder = '/mnt/my_wider_face'
    save_tag_path = '/mnt/my_wider_face/'
    
    ground_truth_file = open('/media/chinasilva/编程资料/deeplearning/datasets/wider_face/wider_face_split/wider_face_train_bbx_gt.txt', 'r')

    dataset = WiderFaceDataset(images_folder='/media/chinasilva/编程资料/deeplearning/datasets/wider_face/WIDER_train/images',
                                ground_truth_file='/media/chinasilva/编程资料/deeplearning/datasets/wider_face/wider_face_split/wider_face_train_bbx_gt.txt',
                                transform=transfroms.ToTensor(),
                                target_transform=lambda x: torch.tensor(x))
    newImgNameLst=['a','b','c','d','e']
    for my_format in [48,24,12]:
        for i, sample in enumerate(dataset):
            try:
            
                var = sample
                image_transformed = var['image']
                label_transformed = var['label']
                image_name = var['image_name']
                image_name_detail=var['image_name_detail']
                #plt.figure()
                image_transformed = image_transformed.numpy().transpose((1, 2, 0))
                image_transformed = np.floor(image_transformed * 255).astype(np.uint8)
                image = cv2.imread(image_name)
                s=0
                for rect in label_transformed:
                    s=s+1
                    x, y, w, h = rect
                    x, y, w, h = list(map(lambda k: k.item(), [x, y, w, h]))
                    x1=x
                    y1=y
                    x2=x1+w
                    y2=y1+h
                    #对下面操作执行多次，产生多张图片
                    j=0
                    bigsize=0
                    while j<1:
                        bigsize=bigsize+1
                        if bigsize>1000: # 重复1000次，仍然没有匹配认为匹配不了
                            j=j+1
                        size = npr.randint(int(min(w, h) * 0.8), np.ceil(1.25 * max(w, h))) 
                        # delta_x = npr.randint(-w * 0.2, w * 0.2)
                        # delta_y = npr.randint(-h * 0.2, h * 0.2)    
                        delta_x = npr.randint(-w * 0.4, w * 0.4)
                        delta_y = npr.randint(-h * 0.4, h * 0.4) 
                        min_delta=min(delta_y,delta_x)
                        nx1 = int(max(x1 + w / 2 + min_delta - size / 2, 0))
                        ny1 = int(max(y1 + h / 2 + min_delta - size / 2, 0))
                        nx2 = nx1 + size
                        ny2 = ny1 + size    
                        crop_box = np.array([nx1, ny1, nx2, ny2])
                        offset_x1 = (nx1 - x1) / float(size)
                        offset_y1 = (ny1 - y1) / float(size)
                        offset_x2 = (nx2 - x2) / float(size)
                        offset_y2 = (ny2 - y2) / float(size)    
                        # cropped_im = image[ny1 : ny2, nx1 : nx2, :]
                        # resized_im = cv2.resize(cropped_im, (my_format, my_format), interpolation=cv2.INTER_LINEAR)

                        #对原图和新图求IOU
                        p1=(x1,y1)
                        p2=(x2,y2)
                        newP1=(nx1,ny1)
                        newP2=(nx2,ny2)
                        iouValue= iouFun((p1,p2,0),(newP1,newP2,0))
                        newImgPosition=(nx1,ny1,nx2,ny2)
                        #使用三个不同值进行范围缩放
                        #分别执行，1.从原图抠图 2.保存不同尺寸图片 3.保存坐标文件
                        imgPath2=''
                        confidence=0
                        # if iouValue>0.65:
                        #     imgPath2='positive'
                        #     confidence=MyEnum.positive.value
                        if iouValue<0.3:
                            imgPath2='negative'
                            confidence=MyEnum.negative.value
                        # elif iouValue>0.4 or iouValue<0.65:
                        #     imgPath2='part'
                        #     confidence=MyEnum.part.value
                        if imgPath2:
                            newImgName='b'+"-"+str(s)+"-"+str(j)+"-"+image_name_detail
                            if imgPath2=='negative':
                                offset=(newImgName,confidence,0,0,0,0)
                            else:
                                offset=(newImgName,confidence,offset_x1,offset_y1,offset_x2,offset_y2)
                            j=j+1
                            print("生成{}尺寸，第{}轮，第{}次".format(my_format,i,j))
                            processImage(newImgName,image_name_detail,image_name,save_images_folder,imgPath2,save_tag_path,offset,newImgPosition,outImgSize=my_format)
            except Exception as e:
                print("ERROR:","__name__"+str(e))
    #    cv2.rectangle(image, pt1=(x, y), pt2=(x + w, y + h),color=(255,0,0))

   # for i, sample in enumerate(dataset):
   #     print(i, sample['image'])
   # 
   # print(len(dataset))