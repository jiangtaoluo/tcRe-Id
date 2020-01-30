import numpy as np
import scipy.io as io
import torch


def euclidean_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


# read mat file
result = io.loadmat('./result.mat')
query_feat = result['query_f']
gallery_feat = result['gallery_f']
g_vids = result['gallery_id'].squeeze()
q_vids = result['query_id'].squeeze()
g_camids = result['gallery_cam'].squeeze()
q_camids = result['query_cam'].squeeze()

distmat = euclidean_dist(torch.tensor(query_feat), torch.tensor(gallery_feat))
num_q, num_g = distmat.shape

indices = np.argsort(distmat, axis=1)
matches = (g_vids[indices] == q_vids[:, np.newaxis]).astype(np.int32)

# compute cmc curve for each query
all_cmc = []
all_AP = []
num_valid_q = 0.  # number of valid query
for q_idx in range(num_q):
    # get query id and camid
    q_vid = q_vids[q_idx]
    q_camid = q_camids[q_idx]

    # remove gallery samples that have the same pid and camid with query
    order = indices[q_idx]
    # print(g_vids[order] == q_vid)
    # print(g_camids[order] == q_camid)

    remove = (g_vids[order] == q_vid) & (g_camids[order] == q_camid)
    keep = np.invert(remove)

    # compute cmc curve
    # binary vector, positions with value 1 are correct matches
    orig_cmc = matches[q_idx][keep]
    if not np.any(orig_cmc):
        # this condition is true when query identity does not appear in gallery
        continue

    cmc = orig_cmc.cumsum()
    cmc[cmc > 1] = 1

    all_cmc.append(cmc[:50])
    num_valid_q += 1.

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

print(mAP, all_cmc[:10])

