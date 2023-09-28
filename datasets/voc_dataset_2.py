import os

import numpy as np
import xml.etree.ElementTree as ET
import collections

import matplotlib.pyplot as plt

from torchvision.datasets import VOCDetection
from PIL import Image, ImageDraw
from torchvision.transforms.functional import to_tensor, to_pil_image

#import albumentations as A
#from albumentations.pytorch import ToTensor

path2data = 'C:/Users/RTL/Documents/GitHub/PyTorch-FusionStudio'
if not os.path.exists(path2data):
    os.mkdir(path2data)

classes = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor"
]

class myVOCDetection(VOCDetection):
    def __getitem__(self, index):
        img = np.array(Image.open(self.images[index]).convert('RGB'))
        target = self.parse_voc_xml(ET.parse(self.annotations[index]).getroot()) # xml파일 분석하여 dict으로 받아오기

        targets = [] # 바운딩 박스 좌표
        labels = [] # 바운딩 박스 클래스

        # 바운딩 박스 정보 받아오기
        for t in target['annotation']['object']:
            label = np.zeros(5)
            label[:] = t['bndbox']['xmin'], t['bndbox']['ymin'], t['bndbox']['xmax'], t['bndbox']['ymax'], classes.index(t['name'])

            targets.append(list(label[:4])) # 바운딩 박스 좌표
            labels.append(label[4])         # 바운딩 박스 클래스

        if self.transforms:
            augmentations = self.transforms(image=img, bboxes=targets)
            img = augmentations['image']
            targets = augmentations['bboxes']

        return img, targets, labels

    def parse_voc_xml(self, node: ET.Element) -> dict[str, any]: # xml 파일을 dictionary로 반환
        voc_dict: dict[str, any] = {}
        children = list(node)
        if children:
            def_dic: dict[str, any] = collections.defaultdict(list)
            for dc in map(self.parse_voc_xml, children):
                for ind, v in dc.items():
                    def_dic[ind].append(v)
            if node.tag == "annotation":
                def_dic["object"] = [def_dic["object"]]
            voc_dict = {node.tag: {ind: v[0] if len(v) == 1 else v for ind, v in def_dic.items()}}
        if node.text:
            text = node.text.strip()
            if not children:
                voc_dict[node.tag] = text
        return voc_dict

# 시각화 함수
def show(img, targets, labels, classes=classes):
    img = to_pil_image(img)
    draw = ImageDraw.Draw(img)
    targets = np.array(targets)
    W, H = img.size

    for tg,label in zip(targets,labels):
        id_ = int(label) # class
        bbox = tg[:4]    # [x1, y1, x2, y2]

        color = [int(c) for c in colors[id_]]
        name = classes[id_]

        draw.rectangle(((bbox[0], bbox[1]), (bbox[2], bbox[3])), outline=tuple(color), width=3)
        draw.text((bbox[0], bbox[1]), name, fill=(255,255,255,0))


if __name__ == '__main__':

    train_ds = myVOCDetection(path2data, year='2007', image_set='train', download=True)
    val_ds = myVOCDetection(path2data, year='2007', image_set='test', download=True)
    train_2_ds = myVOCDetection(path2data, year='2012', image_set='train', download=True)
    val_2_ds = myVOCDetection(path2data, year='2012', image_set='val', download=True)
    trainval_ds = myVOCDetection(path2data, year='2012', image_set='trainval', download=True)
    img, target, label = train_ds[10]
    colors = np.random.randint(0, 255, size=(80,3), dtype='uint8') # 바운딩 박스 색상
    print("what is train_ds length? :", len(train_ds))
    print("what's val_ds length ? ", len(val_ds))
    print("train_2 length : ", len(train_2_ds))
    print("val_2_ds length :", len(val_2_ds))
    print("what is image shape ? :", img.shape)
    print("what is label? :", label)
    print("what's target? :", target)