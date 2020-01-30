import numpy as np
import os.path
from torchvision import datasets
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import math
from collections import Counter
import networkx as nx
import scipy.io as io


# get cam_id, labels, frames from filename
def get_id(img_path):
    camids, labels, frames = [], [], []
    for path in img_path:
        filename = path[0].split('/')[-1].replace('.jpg', '')
        label, camid, timestamp, _ = filename.split('_')
        labels.append(int(label))
        camids.append(int(camid[1:]))
        frames.append(int(timestamp))
    return labels, camids, frames


# get single label, camid, frame
def get_single_id(path):
    filename = path.split('/')[-1].replace('.jpg', '')
    label, camid, timestamp, _ = filename.split('_')
    return int(label), int(camid[1:]), int(timestamp)


# clutering those images into tracks
def track_cluster(train_labels, train_camids, train_frames):
    # build a dict
    # key: label
    #       key: cam
    #             value: frames
    label2cam2frames = {}
    for i in range(len(train_frames)):
        label2cam2frames.setdefault(train_labels[i], {}).\
            setdefault(train_camids[i], []).append(int(train_frames[i]))

    label2cam2center_points = {}
    # find those center points of each label appears in each camera
    for label in label2cam2frames.keys():
        for cam in label2cam2frames[label].keys():
            frames = np.array(label2cam2frames[label][cam])
            # build coor to a 2D array, for the conveilliance of the Kmeans cluster
            coor = [[cam, frame] for frame in frames]
            #  determine how many cluster it should be
            # there is an assumption which each label at most can be captured in one camera for 6 times
            # and use ceil() to get the closest int number
            k = math.ceil(len(frames) / 6)
            kmeans = KMeans(n_clusters=k, random_state=0).fit(coor)
            center_points = [int(center[1]) for center in kmeans.cluster_centers_]
            labels = set(kmeans.labels_)
            assert len(center_points) == len(labels)
            for center_point, lab in zip(center_points, labels):
                ind = np.where(kmeans.labels_ == lab)[0]
                label2cam2center_points.\
                    setdefault(label, {}).\
                    setdefault(cam, {}).\
                    setdefault(center_point, []).\
                    extend(frames[ind])

    label2track = {}
    for label in label2cam2center_points.keys():
        # loop over label
        cam2center = []
        for cam in label2cam2center_points[label].keys():
            times = label2cam2center_points[label][cam]
            for time in times:
                cam2center.append((cam, time))
        cam2center = sorted(cam2center, key=lambda t: t[1], reverse=False)
        #     print(cam2center)
        cams = [cam for cam, _ in cam2center]
        centers = [center for _, center in cam2center]

        camid, k = Counter(cams).most_common(1)[0][0], Counter(cams).most_common(1)[0][1]
        if k == 2:
            duplicate_centers = []
            for cam, c in cam2center:
                if cam == camid:
                    duplicate_centers.append(c)
            c1, c2 = duplicate_centers
            if abs(c1 - c2) / c1 < 0.1:
                k = 1
        if k == 1:
            for i in range(len(centers)):
                if i + 1 == len(centers):
                    continue
                #             print(centers[i],centers[i+1])
                delta_center = abs(centers[i] - centers[i + 1])
                #             print(delta_center)
                if delta_center > centers[i] * 0.2:
                    k += 1

        kmeans = KMeans(n_clusters=k, random_state=0).fit(cam2center)
        #     print(kmeans.labels_)

        for i in set(kmeans.labels_):
            good_index = np.where(i == kmeans.labels_)[0]
            #         print(i, good_index)
            for ind in good_index:
                cam = cams[ind]
                label2track.setdefault(label, {})\
                    .setdefault(i, {})\
                    .setdefault(cam, {})\
                    .setdefault(cam2center[ind][1], [])\
                    .extend(label2cam2center_points[label][cam][cam2center[ind][1]])

    return label2track


# get the camera topology adjacent matrix and build a weighted directional graph
def get_cam_topology(label2track, num_cams, draw_graph=False):
    # build the adjacent matrix
    cam_count = np.zeros((num_cams, num_cams))
    for label in label2track.keys():
        for track in label2track[label].keys():
            # get the cams in one vehicle track
            cams = [cam for cam in label2track[label][track].keys()]
            # print(cams)
            for i in range(len(cams)):
                if i + 1 == len(cams):
                    continue
                # print(cams[i], cams[i + 1])
                cam_count[cams[i] - 1][cams[i + 1] - 1] += 1

    # create a weighted directional graph
    G = nx.Graph()
    for i in range(num_cams):
        for j in range(num_cams):
            if cam_count[i][j] != 0:
                # put the index back to cam_id
                G.add_edge(i + 1, j + 1, weight=cam_count[i][j])

    if draw_graph:
        elarge = [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] > 100]
        esmall = [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] <=10]

        pos = nx.spring_layout(G)  # positions for all nodes

        # nodes
        nx.draw_networkx_nodes(G, pos, node_size=50)
        # edges
        nx.draw_networkx_edges(G, pos, edgelist=elarge,
                               width=5)
        nx.draw_networkx_edges(G, pos, edgelist=esmall,
                               width=1, alpha=0.5, edge_color='b', style='dashed')

        # labels
        nx.draw_networkx_labels(G, pos, font_size=20, font_family='sans-serif')

        plt.axis('off')
        plt.show()

    return cam_count, G


# get those path2index dict
def get_filename2index(paths):
    # filename2index use to record the img_filename and index in the feature embeddings mat
    filename2ind = {}
    for i, (path, _) in enumerate(paths):
        filename = path.split('/')[-1].replace('.jpg','')
        filename2ind[filename] = i

    return filename2ind


# get the candidate label2track dict
def get_candidate_track(label2track, adj_cams, cam, frame, threshold=0.1):
    candidate_labels = {}
    num_tracks = 0
    # loop over all label and track to find the nearest tracks
    for label in label2track.keys():
        for track in label2track[label].keys():
            cams = label2track[label][track].keys()
            prev, nex = adj_cams
            if (prev in cams and cam in cams) and (nex not in cams and cam in cams):
                prev_center_point = sorted(label2track[label][track][prev].keys())[0]
                if abs(prev_center_point - frame) / frame < threshold:
#                     print("-" * 5, label, track)
#                     print(label, track, prev, prev_center_point, abs(prev_center_point - frame) / frame)
                    num_tracks += 1
                    candidate_labels.setdefault(label, []).append(track)
            elif (prev not in cams and cam in cams) and (nex in cams and cam in cams):
                nex_center_point = sorted(label2track[label][track][nex].keys())[0]
                if abs(nex_center_point - frame) / frame < threshold:
#                     print("-" * 5, label, track)
#                     print(label, track, nex, nex_center_point, abs(nex_center_point - frame) / frame)
                    num_tracks += 1
                    candidate_labels.setdefault(label, []).append(track)
            elif (prev in cams and cam in cams) and (nex in cams and cam in cams):
                prev_center_point = sorted(label2track[label][track][prev].keys())[0]
                nex_center_point = sorted(label2track[label][track][nex].keys())[0]
                if frame == 0:
                    frame = frame + 1e-7
                if abs(prev_center_point - frame) / frame < threshold or abs(nex_center_point - frame) / frame < threshold:
#                     print("-" * 5, label, track)
#                     print(label, track, prev, prev_center_point, nex, nex_center_point, \
#                           abs(prev_center_point - frame) / frame, abs(nex_center_point - frame) / frame)
                    num_tracks += 1
                    candidate_labels.setdefault(label, []).append(track)
            else:
                if (prev not in cams and nex not in cams and cam in cams):
                    cam_center_point = sorted(label2track[label][track][cam].keys())[0]
                    if abs(cam_center_point - frame) / frame < threshold:
                        num_tracks += 1
                        candidate_labels.setdefault(label, []).append(track)

    return candidate_labels, num_tracks


# rebuild filename from infos
def get_filename(label, cam, frame, train_gallery=True):
    label, cam, frame = str(label), str(cam), str(frame)
    label = label.zfill(4)
    cam = 'c' + cam.zfill(3)
    frame = frame.zfill(8)
    #     print(label, cam ,frame)
    filename = label + '_' + cam + '_' + frame

    root = '/mnt/storage/Dataset/VeRi/VeRi_st_reformat/image_train' if train_gallery \
        else '/mnt/storage/Dataset/VeRi/VeRi_st_reformat/image_test'
    root = os.path.join(root, label)

    count_suffix = ['_0.jpg', '_1.jpg', '_2.jpg', '_3.jpg', '_4.jpg']
    poss_filename = [filename + suffix for suffix in count_suffix]

    filepath = [os.path.join(root, fname) for fname in poss_filename]
    flags = np.array([os.path.exists(fpath) for fpath in filepath])
    #     print(flags, filepath)
    ind = int(np.where(True == flags)[0])
    #     print(filepath[ind])
    return filepath[ind]


# get candidate path
def get_candidate_img(label2track, candidate):
    candidate_img = dict()
    num_imgs = []
    for label, tracks in candidate.items():
        for track in tracks:
            num_img = 0
            for cam in label2track[label][track].keys():
                for center in label2track[label][track][cam].keys():
                    frames = label2track[label][track][cam][center]
                    for f in frames:
                        filename = get_filename(label, cam, f, train_gallery=False)
                        candidate_img.setdefault(label, []).append(filename)
                        num_img += 1
            num_imgs.append(num_img)
    return candidate_img, num_imgs


# get st_dist
def get_st_distribution():
    # get the image data
    root = '/mnt/storage/Dataset/VeRi/VeRi_st_reformat/'
    image_datasets = {x: datasets.ImageFolder(os.path.join(root, x))
                      for x in ['image_train_all', 'image_query', 'image_test']}
    # get image paths
    train_path = image_datasets['image_train_all'].imgs
    query_path = image_datasets['image_query'].imgs
    gallery_path = image_datasets['image_test'].imgs
    # prepare the training info
    train_labels, train_camids, train_frames = get_id(train_path)
    gallery_labels, gallery_camids, gallery_frames = get_id(gallery_path)
    # get query image infos
    q_labels, q_camids, q_frames = get_id(query_path)
    # get the dict from filename to index
    gallery_filename2index = get_filename2index(gallery_path)

    # use training info to get the camera topology
    train_vid2track = track_cluster(train_labels, train_camids, train_frames)
    gallery_vid2track = track_cluster(gallery_labels, gallery_camids, gallery_frames)

    num_cams = len(set(train_camids))
    cam_adj_matrix, Graph = get_cam_topology(train_vid2track, num_cams, draw_graph=False)

    # get the st_dist
    st_distribution = {}

    for i in range(len(query_path)):
        q_filename = query_path[i][0].split('/')[-1].replace('.jpg', '')
        q_label, q_camid, q_frame = q_labels[i], q_camids[i], q_frames[i]
        # get the adj_cams
        adj_cams = np.argsort(-cam_adj_matrix[q_camid - 1])[:2] + 1
        # get the candidate tracks in gallery_tracks
        candidate_tracks, num_tracks = get_candidate_track(gallery_vid2track, adj_cams, q_camid, q_frame, threshold=0.04)

        # get the candidate imgs from candidate_tracks
        candidate_imgs, num_imgs = get_candidate_img(gallery_vid2track, candidate_tracks)

        # prepare the st_gain
        st_gain = {}

        for g_path in gallery_path:
            g_filename = g_path[0].split('/')[-1].replace('.jpg', '')
            st_gain[g_filename] = 0

        for (can_label, can_imgs), num_img in zip(candidate_imgs.items(), num_imgs):
            for img in can_imgs:
                g_filename = img.split('/')[-1].replace('.jpg', '')
                g_label, _, g_frame = get_single_id(img)
                # option 1：mAP 稍微减低
                w = abs(q_frame - g_frame) / q_frame
                gain = 1.0 - 1.0 / (1 + np.exp(-w))
                # option 2
                # gain = 1 / (abs(q_frame - g_frame) + 1)
                st_gain[g_filename] = gain
        st_distribution[q_filename] = st_gain

    # io.savemat('./st_distribution.mat', st_distribution)

    return st_distribution


if __name__ == "__main__":
    get_st_distribution()
