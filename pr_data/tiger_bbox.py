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

pose_ann_path = './pr_data/atrw_anno_reid_train/reid_keypoints_train.json'
img_path = './pr_data/atrw_reid_train/train/'

'''关节点bbox可视化'''
print('*'*50)
#可以连线的点对
skeleton = [[0, 2], [1, 2], [2, 14], [14, 5], [5, 6], [14, 3], [3, 4], [14, 13],
            [13, 7], [13, 10], [7, 8], [8, 9], [10, 11], [11, 12]]

#对原图片的爪子进行mask操作
def masked(img_path, x1, x2, x3, x4, y1, y2, y3, y4):
    img = cv.imread(img_path)
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

def show_ann_pic_bbox(name, pos):
    img = Image.open(opt.join(img_path, name))
    fig = plt.figure()
    ax = fig.add_subplot(111)

    #画出关节点的连线
    for pair in skeleton:
        if np.all(pos[pair, 2] > 0):             #关节点可见
            color = (np.random.random((1, 3)) * 0.6 + 0.4).tolist()[0]
            plt.plot((pos[pair[0], 0], pos[pair[1], 0]), (pos[pair[0], 1], pos[pair[1], 1]), linewidth=3, color=color)

    #标记出关节点
    for i in range(15):
        if pos[i, 2] > 0:
            plt.plot(pos[i, 0], pos[i, 1], 'o', markersize=8, markeredgecolor='k', markerfacecolor='r')

    #===========================画出bbox
    #1.body:
    x1 = [pos[2, 0], pos[13, 0], pos[0, 0], pos[1, 0], pos[10, 0], pos[7, 0]]     #x，点3（鼻子），点14（尾巴），点1、2（耳朵），点8、11（屁股）
    x2 = [pos[2, 0], pos[13, 0], pos[0, 0], pos[1, 0], pos[10, 0], pos[7, 0]]     #x，点3（鼻子），点14（尾巴），点1、2（耳朵），点8、11（屁股）
    y1 = [pos[1, 1], pos[0, 1], pos[13, 1]]                                       #y1，点1、2（耳朵），点14（尾巴）
    y2 = [pos[10, 1], pos[7, 1], pos[3, 1], pos[5, 1]]                            #y2，点8、11（屁股），点4、6（肩膀）

    x1 = remove0(x1)
    x2 = remove0(x2)
    y1 = remove0(y1)
    y2 = remove0(y2)

    if x1 and x2 and y1 and y2:       #都不为空
        #计算x
        if len(x1) != 1:
            xmin = min(x1)
            xmax = max(x2)
            ymin = min(y1)
            ymax = max(y2)
            rect = plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, edgecolor='yellow', fill=False, linewidth=2)
            ax.add_patch(rect)

            #切图 & 保存图片
            img_body = img.crop((xmin, ymin, xmax, ymax))
            img_body.save('./pr_data/Pseudo_Mask/'+ name.split('.')[0] +'_body.jpg')
        else:
            pass
    else:
        pass

    #2. paw
    pairs = [[5, 6], [3, 4], [7, 8], [8, 9], [10, 11], [11, 12]] #四个爪子的线段配对：左前，右前，右后上，右后下，左后上，左后下
    for ind, pair in enumerate(pairs):
        ind += 1                                                 #身体部位的编号
        i, j = pair[0], pair[1]
        if 0 not in [pos[i, 2], pos[j, 2]]:                      #判断是否有0在每一对线段中
            x1 = pos[i, 0]
            y1 = pos[i, 1]
            x2 = pos[j, 0]
            y2 = pos[j, 1]
            xx1, xx2, xx3, xx4, yy1, yy2, yy3, yy4 = aabb_box(x1, y1, x2, y2, ind)
            plt.plot((xx1, xx2), (yy1, yy2), linewidth=3, color='yellow')
            plt.plot((xx2, xx3), (yy2, yy3), linewidth=3, color='yellow')
            plt.plot((xx3, xx4), (yy3, yy4), linewidth=3, color='yellow')
            plt.plot((xx4, xx1), (yy4, yy1), linewidth=3, color='yellow')

            img1 = masked(opt.join(img_path, name), xx1, xx2, xx3, xx4, yy1, yy2, yy3, yy4)
            if type(img1) == int:
                break
            else:
                img1 = Image.fromarray(cv.cvtColor(img1, cv.COLOR_BGR2RGB))              #转成PIL格式，避免opencv保存图片会产生错误
                img1.save('./pr_data/Pseudo_Mask/'+ name.split('.')[0] +'_'+ str(ind) +'part.jpg')
                # cv.imwrite('data/Pseudo_Mask/'+ name.split('.')[0] +'_'+ str(ind) +'part.jpg' ,img1)

    #显示图片
    plt.imshow(img)
    plt.axis('off')
    # plt.show()
    # plt.savefig(opt.join('data/split_by_id/pose/', name))
    plt.close('all')                                #清理画布，防止一张图上多个图

def Second():
    if not opt.exists('./pr_data/Pseudo_Mask/'):
        os.mkdir('./pr_data/Pseudo_Mask/')
        with open(pose_ann_path, 'r') as f:
            pos_json = json.load(f)
            for i in tqdm(pos_json):
                pic_name = i
                pos = np.array(pos_json[i]).reshape((15, 3))
                show_ann_pic_bbox(pic_name, pos)

    #=======================================================================
    #重命名保存
    print('*'*25)
    for i in range(4):
        map_dic = {}
        with open('./data/AmurTiger/flod' + str(i) + '/' + 'map_picname.txt', 'r') as f:
            lines = f.readlines()
            for line in lines:
                orign_name = line.split(' ')[0].split('.')[0]
                new_name = line.split(' ')[-1].split('\n')[0].split('.')[0]
                map_dic[orign_name] = new_name
        os.mkdir('./data/AmurTiger/flod' + str(i) + '/' + 'Pseudo_Mask/')
        for pic in os.listdir('./pr_data/Pseudo_Mask/'):
            name, part = pic.split('_')[0], pic.split('_')[1].split('.')[0]
            new_name = map_dic[name] + '.' + part + '.jpg'
            shutil.copyfile('./pr_data/Pseudo_Mask/' + pic, './data/AmurTiger/flod' + str(i) + '/' + 'Pseudo_Mask/' + new_name)