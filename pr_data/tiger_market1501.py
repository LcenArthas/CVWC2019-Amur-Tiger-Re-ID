#---------------------------------------------------------
#把老虎数据集制作成Market1501格式
#其中bounding_box_test与query是相同的
#同时对生成新旧名字对照表 map_picname
#---------------------------------------------------------

import os
import os.path as osp
import pandas as pd
import random
import shutil
from tqdm import tqdm

def First():
    id_path = './pr_data/atrw_anno_reid_train/reid_list_train.csv'
    id_df = pd.read_csv(id_path, header=None, names=['id', 'img_file'])
    id_list = list(id_df['id'].unique())
    random.shuffle(id_list)                        #随机打乱id列表

    split_num = int(len(id_list)*0.2)            #对训练集划分4份，4-flod
    # split_num = int(len(id_list)*0)                #不分折

    for i in range(4):
        start_split = i * split_num
        end_split = (i + 1) * split_num

        train_id_list = id_list[:start_split] + id_list[end_split:]
        test_id_list = id_list[start_split : end_split]
        # test_id_list = id_list[: int(len(id_list)*0.2)]            #不分折，随机选

        if not osp.exists('./data/AmurTiger/flod' + str(i)):
            os.mkdir('./data/AmurTiger/flod' + str(i))
            os.mkdir('./data/AmurTiger/flod' + str(i) + '/' + 'bounding_box_train/')
            os.mkdir('./data/AmurTiger/flod' + str(i) + '/' + 'bounding_box_test/')
            os.mkdir('./data/AmurTiger/flod' + str(i) + '/' + 'query/')

            with open('./data/AmurTiger/flod' + str(i) + '/' + 'map_picname.txt', 'w') as f:
                kk = 0
                for id , file in tqdm(id_df.groupby('id')):
                    img_list = list(file['img_file'])
                    if id in train_id_list:
                        for name in img_list:
                            new_name = '{:>4}_c{}s{}_{}.jpg'.format(id, random.randint(1, 6), random.randint(1, 6), kk)
                            new_path = osp.join('./data/AmurTiger/flod' + str(i) + '/' + 'bounding_box_train/', new_name)
                            shutil.copyfile(osp.join('./pr_data/atrw_reid_train/train/', name), new_path)
                            f.write(name + ' ' + new_name + '\n')
                            kk += 1
                    if id in test_id_list:
                        for name in img_list:
                            new_name = '{:>4}_c{}s{}_{}.jpg'.format(id, random.randint(1, 6), random.randint(1, 6), kk)
                            new_path_1 = osp.join('./data/AmurTiger/flod' + str(i) + '/' + 'bounding_box_test/', new_name)
                            new_path_2 = osp.join('./data/AmurTiger/flod' + str(i) + '/' + 'query/', new_name)
                            shutil.copyfile(osp.join('./pr_data/atrw_reid_train/train/', name), new_path_1)
                            shutil.copyfile(osp.join('./pr_data/atrw_reid_train/train/', name), new_path_2)
                            f.write(name + ' ' + new_name + '\n')
                            kk += 1
                f.close()

