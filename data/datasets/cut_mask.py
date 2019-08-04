#--------------------------------------------------------------------------
#这个脚本为了实验显示根据keypoints划分bbox的结果,同时保存crop的结果，生成presudo_mask
#根据market1501中图片的新名字，对presudo_mask中的图片重命名保存
#--------------------------------------------------------------------------

import os
import os.path as opt
import pandas as pd
import json
import shutil
import cv2 as cv
from PIL import Image
import seaborn as sns
import warnings
import matplotlib.pyplot as plt
import numpy as np
from pycocotools.coco import COCO
import skimage.io as io
from tqdm import tqdm
from scipy.spatial.distance import pdist
import math
import time

#对原图片的爪子进行mask操作
def masked(img, x1, x2, x3, x4, y1, y2, y3, y4):
    img = img = cv.cvtColor(np.asarray(img),cv.COLOR_RGB2BGR)           #PIL转换成opencv
    (h, w, c) = img.shape
    mask = np.zeros((h, w, 3), np.uint8)                                #定义mask的图片尺寸
    triangle = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])       #定义mask部分位置
    cv.fillConvexPoly(mask, triangle, (255, 255, 255))                  #取mask
    img_mask = cv.bitwise_and(img, mask)                                #mask与原图融合
    # plt.imshow(img_mask)
    # plt.show()
    #防止越界
    xmin = min(x1, x2, x3, x4)
    xmax = max(x1, x2, x3, x4)
    ymin = min(y1, y2, y3, y4)
    ymax = max(y1, y2, y3, y4)

    if xmin <= 0:
        xmin = 0
    if ymin <= 0:
        ymin = 0
    if xmax > w:
        xmax = w
    if ymax > h:
        ymax = h
    if xmax <= 0 or ymax <=0 or xmin >= xmax or ymin >= ymax:
        return 1
    else:
        # print('name:{}, xmin:{},xmax:{}, ymin:{}, ymax:{}'.format(img_path, xmin, xmax, ymin, ymax))
        img_crop = img_mask[ymin: ymax, xmin: xmax]
        return img_crop

#根据原来的爪子两个点的坐标，得出框的做标
#ind为区分爪子的类别：
#1,2为前爪
#3,5为后爪上面
#4,6为后爪下面
def aabb_box(x1, y1, x2, y2, ind):
    L = math.sqrt(pow(x2 - x1, 2) + pow(y2 - y1, 2))

    tha = math.atan((x2 -x1)/((y2 -y1)+0.000000000001))
    if ind == 1 or ind == 2:
        detalx = math.cos(tha) * 0.3 * L
        detaly = math.sin(tha) * 0.3 * L
    if ind == 3 or ind == 5:
        detalx = math.cos(tha) * 0.45 * L
        detaly = math.sin(tha) * 0.45 * L
    else:
        detalx = math.cos(tha) * 0.3 * L
        detaly = math.sin(tha) * 0.3 * L
    #按照从左上角的位置开始顺时针1,2,3,4
    xx1 = int(x1 - detalx)
    yy1 = int(y1 + detaly)
    xx2 = int(x1 + detalx)
    yy2 = int(y1 - detaly)
    xx4 = int(x2 - detalx)
    yy4 = int(y2 + detaly)
    xx3 = int(x2 + detalx)
    yy3 = int(y2 - detaly)

    return xx1, xx2, xx3, xx4, yy1, yy2, yy3, yy4

#移除x列表中所有的0
def remove0(x):
    while len(x) != 0:
        if 0 in x:
            x.remove(0)
        else:
            break
    return x

def body_mask(img, pos, partsize):
    #切分身体mask
    # 1.body:
    x1 = [pos[2, 0], pos[13, 0], pos[0, 0], pos[1, 0], pos[10, 0], pos[7, 0]]  # x，点3（鼻子），点14（尾巴），点1、2（耳朵），点8、11（屁股）
    x2 = [pos[2, 0], pos[13, 0], pos[0, 0], pos[1, 0], pos[10, 0], pos[7, 0]]  # x，点3（鼻子），点14（尾巴），点1、2（耳朵），点8、11（屁股）
    y1 = [pos[1, 1], pos[0, 1], pos[13, 1]]  # y1，点1、2（耳朵），点14（尾巴）
    y2 = [pos[10, 1], pos[7, 1], pos[3, 1], pos[5, 1]]  # y2，点8、11（屁股），点4、6（肩膀）

    x1 = remove0(x1)
    x2 = remove0(x2)
    y1 = remove0(y1)
    y2 = remove0(y2)

    if x1 and x2 and y1 and y2:  # 都不为空
        # 计算x
        if len(x1) != 1:
            xmin = min(x1)
            xmax = max(x2)
            ymin = min(y1)
            ymax = max(y2)
            img_body = img.crop((xmin, ymin, xmax, ymax))
            return img_body
        else:
            h = partsize[0]
            w = partsize[1]
            img = np.zeros((h, w, 3), np.uint8)
            return Image.fromarray(img, mode='RGB')
    else:
        h = partsize[0]
        w = partsize[1]
        img = np.zeros((h, w, 3), np.uint8)
        return Image.fromarray(img, mode='RGB')

def paw_mask(img, pos, pair, ind, partsize):
    # 2. paw
    i, j = pair[0], pair[1]
    if 0 not in [pos[i, 2], pos[j, 2]]:  # 判断是否有0在每一对线段中
        x1 = pos[i, 0]
        y1 = pos[i, 1]
        x2 = pos[j, 0]
        y2 = pos[j, 1]
        xx1, xx2, xx3, xx4, yy1, yy2, yy3, yy4 = aabb_box(x1, y1, x2, y2, ind)

        img1 = masked(img, xx1, xx2, xx3, xx4, yy1, yy2, yy3, yy4)
        if type(img1) == int:
            h = partsize[0]
            w = partsize[1]
            img = np.zeros((h, w, 3), np.uint8)
            return Image.fromarray(img, mode='RGB')
        else:
            img1 = Image.fromarray(cv.cvtColor(img1, cv.COLOR_BGR2RGB))  # 转成PIL格式，避免opencv保存图片会产生错误
            return img1
    else:
        h = partsize[0]
        w = partsize[1]
        img = np.zeros((h, w, 3), np.uint8)
        return Image.fromarray(img, mode='RGB')