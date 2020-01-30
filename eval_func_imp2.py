import scipy.io as io
import numpy as np
import torch

def evaluate( query_feat, q_vid, q_cam, gallery_feats, g_vids, g_cams):
    """
    :param query_index: index i
    :param query_feat: query_feature[i]
    :param q_vid: query_vid[i]
    :param q_cam: query_camid[i]
    :param gallery_feats: gallery_features
    :param g_vids: gallery_vids
    :param g_cams: gallery_camids
    :return: temporary CMC curve
    """
    query = query_feat

    score = [np.linalg.norm(query - x) for x in gallery_feats]
    score = np.array(score)

    index = np.argsort(score)

    query_index = np.argwhere(g_vids == q_vid)
    camera_index = np.argwhere(g_cams == q_cam)
    good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
    junk_index = np.intersect1d(query_index, camera_index)

    CMC_tmp = compute_mAP(index, good_index, junk_index)
    return CMC_tmp


def compute_mAP(index, good_index, junk_index):
    ap = 0
    cmc = torch.IntTensor(len(index)).zero_()

    if good_index.size == 0:
        cmc[0] = -1
        return ap, cmc

    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]

    ngood = len(good_index)
    mask = np.in1d(index, good_index)
    rows_good = np.argwhere(mask == True)
    rows_good = rows_good.flatten()

    cmc[rows_good[0]:] = 1

    for i in range(ngood):
        d_recall = 1.0 / ngood
        precision = (i+1) * 1.0 / (rows_good[i] + 1)
        if rows_good[i] != 0:
            old_precision = i * 1.0/ rows_good[i]
        else:
            old_precision = 1
        ap = ap + d_recall * (old_precision + precision) / 2

    return ap, cmc

dist = io.loadmat('./distmat.mat')
distmat = dist['distmat']

# read mat file
result = io.loadmat('./result.mat')
query_feat = result['query_f']
gallery_feat = result['gallery_f']
g_vids = result['gallery_id'].squeeze()
q_vids = result['query_id'].squeeze()
g_camids = result['gallery_cam'].squeeze()
q_camids = result['query_cam'].squeeze()

CMC = torch.IntTensor(len(g_vids)).zero_()
ap = 0.0
low_count, more_count = 0, 0

for i in range(len(q_vids)):
    ap_tmp, CMC_tmp = evaluate(query_feat[i], q_vids[i],q_camids[i],
                               gallery_feat, g_vids, g_camids)
    if CMC_tmp[0] == -1:
        continue
    CMC = CMC + CMC_tmp
    ap += ap_tmp
    # print(ap_tmp)

    if ap_tmp < 0.8:
        low_count += 1
    else:
        more_count += 1

CMC = CMC.float()
CMC = CMC/len(q_vids)

print(low_count, more_count, ap/len(q_vids), CMC[0], CMC[4], CMC[9])
