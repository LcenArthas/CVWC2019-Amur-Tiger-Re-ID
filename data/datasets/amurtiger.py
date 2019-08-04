import glob
import re

import os.path as osp
import os
import json
import numpy as np
from PIL import Image

from .bases import BaseImageDataset


class AmurTiger(BaseImageDataset):
    """
    AmurTiger
    Reference:
    Zheng et al. Scalable Person Re-identification: A Benchmark. ICCV 2015.
    URL: http://www.liangzheng.org/Project/project_reid.html

    Dataset statistics:
    # identities: 1501 (+1 for background)
    # images: 12936 (train) + 3368 (query) + 15913 (gallery)
    """
    dataset_dir_demo = 'AmurTiger'

    def __init__(self, index_flod=0, root='/data', verbose=True, is_demo=False, **kwargs):
        super(AmurTiger, self).__init__()
        self.dataset_dir_flod = 'AmurTiger/flod' + str(index_flod) + '/'
        self.dataset_dir = osp.join(root, self.dataset_dir_flod)
        self.dataset_dir_demo = osp.join(root, self.dataset_dir_demo)
        self.train_dir = osp.join(self.dataset_dir, 'bounding_box_train_filp')
        self.gallery_dir = osp.join(self.dataset_dir, 'bounding_box_test')
        self.mask = osp.join(self.dataset_dir, 'Pseudo_Mask_filp')      #pos中的关节点标注
        # demo
        if is_demo:
            self.query_dir = osp.join(self.dataset_dir_demo, 'reid_test')
            self.gallery_dir = osp.join(self.dataset_dir_demo, 'reid_test')
        else:
            self.query_dir = osp.join(self.dataset_dir, 'query')

        self._check_before_run()

        train = self._process_dir_train(self.train_dir, self.mask, relabel=True)

        if is_demo:
            query = self._process_dir(self.query_dir, relabel=False, is_demo=True)
            gallery = self._process_dir(self.gallery_dir, relabel=False, is_demo=True)

            self.train = train
            self.query = query
            self.gallery = gallery
            self.num_train_pids = 100

        else:
            query = self._process_dir(self.query_dir, relabel=False)
            gallery = self._process_dir(self.gallery_dir, relabel=False)

            if verbose:
                print("=> AmurTiger loaded")
                self.print_dataset_statistics(train, query, gallery)

            #this are we needed
            self.train = train
            self.query = query
            self.gallery = gallery

            self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train, is_train=True)
            self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query, is_train=False)
            self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery, is_train=False)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir(self, dir_path, relabel=False, is_demo=False):
        if is_demo:
            dataset = []
            img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
            for img_path in img_paths:
                dataset.append((img_path, 1, 1))                   # 相当于手动对原来的图片加上id和cam
            return dataset
        else:
            """在送入网络之前，对训练集、查询集的图片标签重新编码"""
            #TODO:标签重新编码，与后面评估函数对应
            img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
            pattern = re.compile(r'([-\d]+)_c(\d)')

            pid_container = set()
            for img_path in img_paths:
                pid, _ = map(int, pattern.search(img_path).groups())
                if pid == -1: continue  # junk images are just ignored
                pid_container.add(pid)
            pid2label = {pid: label for label, pid in enumerate(pid_container)}

            dataset = []
            for img_path in img_paths:
                pid, camid = map(int, pattern.search(img_path).groups())
                if pid == -1: continue  # junk images are just ignored
                camid -= 1  # index starts from 0
                if relabel: pid = pid2label[pid]
                dataset.append((img_path, pid, camid))

            return dataset

    def _process_dir_train(self, dir_path, mask_path, relabel=True):
        """在送入网络之前，对训练集、查询集的图片标签重新编码"""

        img_paths_globale = glob.glob(osp.join(dir_path, '*.jpg'))
        parts_id_list = [pic.split('.')[0] for pic in os.listdir(mask_path)]

        pid_container = set()
        for img_path in img_paths_globale:
            img_path = img_path.split('/')[-1]
            pid= img_path.split('_')[0]
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in img_paths_globale:
            img_path = img_path.split('/')[-1]
            pid ,camid= img_path.split('_')[0], int(list(img_path.split('_')[1])[1])
            camid -= 1  # index starts from 0
            if relabel: pid = pid2label[pid]

            #处理part部分====================================
            #当part文件夹中有这个图片的部分时候
            pic_id = img_path.split('.')[0]
            pic_id = pic_id.lstrip()                    #去掉开头的空格
            if pic_id in parts_id_list:
                if osp.exists(osp.join(mask_path, pic_id + '.body.jpg')):
                    body = osp.join(mask_path, pic_id + '.body.jpg')
                else:
                    body = 'no'
                if osp.exists(osp.join(mask_path, pic_id + '.1part.jpg')):
                    part_1 = osp.join(mask_path, pic_id + '.1part.jpg')
                else:
                    part_1 = 'no'
                if osp.exists(osp.join(mask_path, pic_id + '.2part.jpg')):
                    part_2 = osp.join(mask_path, pic_id + '.2part.jpg')
                else:
                    part_2 = 'no'
                if osp.exists(osp.join(mask_path, pic_id + '.3part.jpg')):
                    part_3 = osp.join(mask_path, pic_id + '.3part.jpg')
                else:
                    part_3 = 'no'
                if osp.exists(osp.join(mask_path, pic_id + '.4part.jpg')):
                    part_4 = osp.join(mask_path, pic_id + '.4part.jpg')
                else:
                    part_4 = 'no'
                if osp.exists(osp.join(mask_path, pic_id + '.5part.jpg')):
                    part_5 = osp.join(mask_path, pic_id + '.5part.jpg')
                else:
                    part_5 = 'no'
                if osp.exists(osp.join(mask_path, pic_id + '.6part.jpg')):
                    part_6 = osp.join(mask_path, pic_id + '.6part.jpg')
                else:
                    part_6 = 'no'
            else:
                body = 'no'
                part_1 = 'no'
                part_2 = 'no'
                part_3 = 'no'
                part_4 = 'no'
                part_5 = 'no'
                part_6 = 'no'
            img_path = osp.join(dir_path, img_path)
            dataset.append((img_path, pid, camid, body, part_1, part_2, part_3, part_4, part_5, part_6))
        return dataset
