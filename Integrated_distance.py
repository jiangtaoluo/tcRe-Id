import torch
import numpy as np
import scipy.io as io
from spatial_temporal_distribution import get_st_distribution


def dist_compute(query_feats, gallery_feats, query_path, gallery_path, alpha=1):
    m, n = query_feats.size(0), gallery_feats.size(0)
    xx = torch.pow(query_feats, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(gallery_feats, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, query_feats, gallery_feats.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability

    if alpha == 0:
        return dist
    else:
        dist = dist.numpy()
        # st_distribution = io.loadmat('./st_distribution.mat')
        # st_distribution = np.load('/home/eini/project/reid/st_distribution.npy')
        st_distribution = get_st_distribution()

        distmat = np.zeros((len(query_feats), len(gallery_feats)))

        for i in range(len(query_feats)):
            q_path = query_path[i]
            query_filename = q_path.split('/')[-1].replace('.jpg', '')
            st_gain = []
            for j in range(len(gallery_path)):
                g_path = gallery_path[j]
                gallery_filename = g_path.split('/')[-1].replace('.jpg', '')
                # print(query_filename, gallery_filename)
                st_gain.append(st_distribution[query_filename][gallery_filename])
            st_gain = np.array(st_gain)
            score = dist[i] - alpha * st_gain
            distmat[i] = score

        return distmat
