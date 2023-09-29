from PIL import Image 
import torchvision.transforms as transforms
from torchvision.datasets import VOCDetection
from torch.utils.data.distributed import DistributedSampler

import xmltodict
import numpy as np
import torch

import os

class YOLO_PASCAL_VOC(VOCDetection):
    def __getitem__(self, index):
        img = (Image.open(self.images[index]).convert('RGB')).resize((448, 448))
        img_transform = transforms.Compose([transforms.PILToTensor(), transforms.Resize((448, 448), antialias=True)])
        img = torch.divide(img_transform(img), 255)

        target = xmltodict.parse(open(self.annotations[index]).read())

        classes = ["aeroplane", "bicycle", "bird", "boat", "bottle",
                   "bus", "car", "cat", "chair", "cow", "diningtable",
                   "dog", "horse", "motorbike", "person", "pottedplant",
                   "sheep", "sofa", "train", "tvmonitor"]

        label = np.zeros((7, 7, 30), dtype = float) # groundtruth bounding box, (7, 7, 25)

        Image_Height = float(target['annotation']['size']['height'])
        Image_Width  = float(target['annotation']['size']['width'])

        # 바운딩 박스 정보 받아오기
        try:
            for obj in target['annotation']['object']:
                
                # class의 index 휙득
                class_index = classes.index(obj['name'].lower())
                
                # min, max좌표 얻기
                x_min = float(obj['bndbox']['xmin']) 
                y_min = float(obj['bndbox']['ymin'])
                x_max = float(obj['bndbox']['xmax']) 
                y_max = float(obj['bndbox']['ymax'])

                # 448*448에 맞게 변형시켜줌
                x_min = float((448.0/Image_Width)*x_min)
                y_min = float((448.0/Image_Height)*y_min)
                x_max = float((448.0/Image_Width)*x_max)
                y_max = float((448.0/Image_Height)*y_max)

                # 변형시킨걸 x,y,w,h로 만들기 
                x = (x_min + x_max)/2.0
                y = (y_min + y_max)/2.0
                w = x_max - x_min
                h = y_max - y_min

                # x,y가 속한 cell알아내기
                x_cell = int(x/64) # 0~6
                y_cell = int(y/64) # 0~6 -> 7개 grid 안에 들어감
                # cell의 중심 좌표는 (0.5, 0.5)다
                x_val_inCell = float((x - x_cell * 64.0)/64.0) # 0.0 ~ 1.0
                y_val_inCell = float((y - y_cell * 64.0)/64.0) # 0.0 ~ 1.0

                # w, h 를 0~1 사이의 값으로 만들기
                w = w / 448.0
                h = h / 448.0

                class_index_inCell = class_index

                label[y_cell][x_cell][21] = x_val_inCell
                label[y_cell][x_cell][22] = y_val_inCell
                label[y_cell][x_cell][23] = w
                label[y_cell][x_cell][24] = h
                label[y_cell][x_cell][20] = 1.0
                label[y_cell][x_cell][class_index_inCell] = 1.0


        # single-object in image
        except TypeError as e : 
            # class의 index 휙득
            class_index = classes.index(target['annotation']['object']['name'].lower())
                
            # min, max좌표 얻기
            x_min = float(target['annotation']['object']['bndbox']['xmin']) 
            y_min = float(target['annotation']['object']['bndbox']['ymin'])
            x_max = float(target['annotation']['object']['bndbox']['xmax']) 
            y_max = float(target['annotation']['object']['bndbox']['ymax'])

            # 224*224에 맞게 변형시켜줌
            x_min = float((448.0/Image_Width)*x_min)
            y_min = float((448.0/Image_Height)*y_min)
            x_max = float((448.0/Image_Width)*x_max)
            y_max = float((448.0/Image_Height)*y_max)

            # 변형시킨걸 x,y,w,h로 만들기 
            x = (x_min + x_max)/2.0
            y = (y_min + y_max)/2.0
            w = x_max - x_min
            h = y_max - y_min

            # x,y가 속한 cell알아내기
            x_cell = int(x/64) # 0~6
            y_cell = int(y/64) # 0~6
            x_val_inCell = float((x - x_cell * 64.0)/64.0) # 0.0 ~ 1.0
            y_val_inCell = float((y - y_cell * 64.0)/64.0) # 0.0 ~ 1.0

            # w, h 를 0~1 사이의 값으로 만들기
            w = w / 448.0
            h = h / 448.0

            class_index_inCell = class_index

            label[y_cell][x_cell][21] = x_val_inCell
            label[y_cell][x_cell][22] = y_val_inCell
            label[y_cell][x_cell][23] = w
            label[y_cell][x_cell][24] = h
            label[y_cell][x_cell][20] = 1.0
            label[y_cell][x_cell][class_index_inCell] = 1.0
            
        return img, torch.tensor(label)