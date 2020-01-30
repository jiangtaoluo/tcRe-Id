from .eval_reid import eval_func
from .re_ranking import re_ranking
import torch
from Integrated_distance import dist_compute


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

def re_rank(q, g):
    qq_dist = euclidean_dist(q, q).numpy()
    gg_dist = euclidean_dist(g, g).numpy()
    qg_dist = euclidean_dist(q, g).numpy()
    distmat = re_ranking(qg_dist, qq_dist, gg_dist)
    return distmat

def re_rank_with_st(q, g, query_path, gallery_path, alpha=6):
    qq_dist = euclidean_dist(q, q).numpy()
    # qq_dist = dist_compute(q, q, query_path, query_path, alpha)
    gg_dist = euclidean_dist(g, g).numpy()
    # gg_dist = dist_compute(g, g, gallery_path, gallery_path, alpha)
    # qg_dist = euclidean_dist(q, g).numpy()
    qg_dist = dist_compute(q, g,
                 query_path, gallery_path, alpha)
    distmat = re_ranking(qg_dist, qq_dist, gg_dist)
    return distmat
