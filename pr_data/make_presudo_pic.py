#---------------------------------------------------------
#这个脚本是为了生成新的伪id图片，根据原图片进行水平翻转，生成新的id，扩充数据
#将原来整理好的图片翻转，复制，重命名（图片名字前加999）
#---------------------------------------------------------

import os
import os.path as osp
import shutil
from PIL import Image
from tqdm import tqdm

path = './data/AmurTiger/'

to_do_list = ['bounding_box_train', 'Pseudo_Mask']

def Third():
    for i in range(4):
        for n in to_do_list:
            if not osp.exists(path + 'flod' + str(i) + '/' + n + '_filp'):
                os.mkdir(path + 'flod' + str(i) + '/' + n + '_filp')
            for pic in tqdm(os.listdir(path + 'flod' + str(i) + '/' + n)):
                im = Image.open(path + 'flod' + str(i) + '/' + n + '/' + pic)
                new_im = im.transpose(Image.FLIP_LEFT_RIGHT)    #水平翻转

                new_im.save(path + 'flod' + str(i) + '/' + n + '_filp' + '/' + '999' + pic.lstrip())
                im.save(path + 'flod' + str(i) + '/' + n + '_filp' + '/' + pic)

#制作包含为标签的翻转
