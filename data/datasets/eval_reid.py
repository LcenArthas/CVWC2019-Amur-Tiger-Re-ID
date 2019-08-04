# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import numpy as np
from sklearn.preprocessing import MinMaxScaler


def eval_func(distmat, q_pids=None, g_pids=None, q_camids=None, g_camids=None, q_paths=None, g_paths=None, max_rank=50, is_demo=False):
    """Evaluation with market1501 metric
        修改这个测评函数，可以测评相同摄像头
        Key: for each query identity, its gallery images from the same camera view are discarded.
        """
    num_q, num_g = distmat.shape

    if is_demo:
        # max_rank=11
        indices = np.argsort(distmat, axis=1)                                  # 排序
        # matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)  # 首先对g_pids得出的大小数据排序，然后每行与query
        #                                                                        # 的序号比较
        # num_valid_q = 0.  # number of valid query
        all_pic_path = []
        for q_idx in range(num_q):
            order = indices[q_idx]
            # 得出每个pic_path

            pic = list(g_paths[order[1:]])                                 # 剔除原图片，即第一张图片的干扰
            pic = pic[:max_rank]
            pic.append(q_paths[q_idx])                                     #最后一张为原图

            all_pic_path.append(pic)
        return  all_pic_path
    else:
        if num_g < max_rank:
            max_rank = num_g
            print("Note: number of gallery samples is quite small, got {}".format(num_g))
        indices = np.argsort(distmat, axis=1)                                                     #排序
        matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)                     #首先对g_pids得出的大小数据排序，然后每行与query
                                                                                                  #的序号比较

        # compute cmc curve for each query
        all_cmc = []
        all_AP = []
        all_pic_path = []
        all_pic_score = []                                                                        #存储返回图片的得分
        num_valid_q = 0.  # number of valid query
        for q_idx in range(num_q):
            # get query pid and camid
            q_pid = q_pids[q_idx]
            q_camid = q_camids[q_idx]

            # remove gallery samples that have the same pid and camid with query
            order = indices[q_idx]
            #TODO:这是可以选出和需要查询的图片相同的id且cam也相同，返回的是一个列表由0,1组成，1表示要剔出的
            remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
            remove = (g_camids[order] == 7)                                                      #让remove都为False
            keep = np.invert(remove)                                                             #np.invert是可以把0变1，把1变0
            keep[0] = False

            # compute cmc curve
            # binary vector, positions with value 1 are correct matches
            orig_cmc = matches[q_idx][keep]
            if not np.any(orig_cmc):
                # this condition is true when query identity does not appear in gallery
                continue

            cmc = orig_cmc.cumsum()                                                             #列表中每个数都是这个数加上之前的数的和
            cmc[cmc > 1] = 1

            all_cmc.append(cmc[:max_rank])                                                      #算出前50位命中的个数
            num_valid_q += 1.

            #得出每个pic_path
            pic = g_paths[order[1:]]                                                            #剔除原图片，即第一张图片的干扰
            all_pic_path.append(pic[:max_rank])

            # if is_demo:
            #     #得出每个图片得分
            #     score = distmat[q_idx][order[1:]]
            #     score = MinMaxScaler().fit_transform(score[:, np.newaxis])                           #归一化处理
            #     all_pic_score.append(score[:max_rank])
            #     all_pic_score = np.squeeze(all_pic_score[0])

            # compute average precision
            # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
            num_rel = orig_cmc.sum()
            tmp_cmc = orig_cmc.cumsum()
            tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
            tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
            AP = tmp_cmc.sum() / num_rel
            all_AP.append(AP)

        assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

        all_cmc = np.asarray(all_cmc).astype(np.float32)
        all_cmc = all_cmc.sum(0) / num_valid_q
        mAP = np.mean(all_AP)

        if is_demo:
            return all_cmc, mAP, all_pic_path
        else:
            return all_cmc, mAP, all_pic_path
