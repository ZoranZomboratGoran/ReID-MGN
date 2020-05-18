from utils.extract_feature import extract_feature
from opt import opt
from utils.metrics import mean_ap, cmc, re_ranking

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from tqdm import tqdm
from scipy.spatial.distance import cdist

def evaluate_model(app, epoch_label):

    app.model.train(False)
    query_feature = extract_feature(app.model, app.query_loader).numpy()
    gallery_feature_tensor = extract_feature(app.model, app.test_loader)
    gallery_feature = gallery_feature_tensor.numpy()

    data = app.data

    def rank(dist):
        r = cmc(dist, data.queryset.ids, data.testset.ids, data.queryset.cameras, data.testset.cameras,
                separate_camera_set=False,
                single_gallery_shot=False,
                first_match_break=True)
        m_ap = mean_ap(dist, data.queryset.ids, data.testset.ids, data.queryset.cameras, data.testset.cameras)

        return r, m_ap

    dist = cdist(query_feature, gallery_feature)
    r, m_ap = rank(dist)

    print("Epoch %s mAP: %.4f rank1: %.4f rank3: %.4f rank5: %.4f rank10: %.4f"
          % (epoch_label ,m_ap, r[0], r[2], r[4], r[9]))

    app.writer.add_scalar('mAP/epoch', m_ap, epoch_label)
    app.writer.add_scalar('rank1/epoch', r[0], epoch_label)
    app.writer.add_scalar('rank5/epoch', r[4], epoch_label)
    app.writer.add_scalar('rank10/epoch', r[9], epoch_label)
    app.writer.flush()

    # top 10 rank
    single_query_feature = extract_feature(app.model, [(torch.unsqueeze(app.data.query_image, 0), 1)])

    gallery_path = data.testset.imgs
    gallery_label = data.testset.ids
    query_label = data.queryset.id(data.query_image_path)

    # sort images
    single_query_feature = single_query_feature.view(-1, 1)

    score = torch.mm(gallery_feature_tensor, single_query_feature)
    score = score.squeeze(1).cpu()
    score = score.numpy()

    index = np.argsort(score)  # from small to large
    index = index[::-1]  # from large to small

    # Visualize the rank result
    fig = plt.figure(figsize=(16, 4))

    ax = plt.subplot(1, 11, 1)
    ax.axis('off')
    plt.imshow(plt.imread(app.data.query_image_path))
    ax.set_title('query')

    for i in range(10):
        img_path = gallery_path[index[i]]
        ax = plt.subplot(1, 11, i + 2)
        ax.axis('off')
        plt.imshow(plt.imread(img_path))

        label = gallery_label[index[i]]
        if label == query_label:
            ax.set_title(img_path.split('/')[-1][:9], color='green')
        else:
            ax.set_title(img_path.split('/')[-1][:9], color='red')

    fig.tight_layout()
    images_dir = 'images'
    os.makedirs(images_dir, exist_ok=True)
    fig_filename = 'top10_%s.png' % epoch_label
    fig.savefig(os.path.join(images_dir, fig_filename))

def load_network(model):

    load_filename = opt.weight
    checkpoint = torch.load(load_filename)
    model.load_state_dict(checkpoint['model'])
    if opt.usecpu == False and torch.cuda.is_available():
        model.cuda()

def evaluate_rerank(app):

    load_network(app.model)
    app.model.eval()

    gf = extract_feature(app.model, app.test_loader).numpy()
    qf = extract_feature(app.model, app.query_loader).numpy()

    def rank(dist):
        r = cmc(dist, app.queryset.ids, app.testset.ids, app.queryset.cameras, app.testset.cameras,
                separate_camera_set=False,
                single_gallery_shot=False,
                first_match_break=True)
        m_ap = mean_ap(dist, app.queryset.ids, app.testset.ids, app.queryset.cameras, app.testset.cameras)

        return r, m_ap

    #########################   re rank##########################
    q_g_dist = np.dot(qf, np.transpose(gf))
    q_q_dist = np.dot(qf, np.transpose(qf))
    g_g_dist = np.dot(gf, np.transpose(gf))
    dist = re_ranking(q_g_dist, q_q_dist, g_g_dist)

    r, m_ap = rank(dist)

    print('[With    Re-Ranking] mAP: {:.4f} rank1: {:.4f} rank3: {:.4f} rank5: {:.4f} rank10: {:.4f}'
            .format(m_ap, r[0], r[2], r[4], r[9]))

    #########################no re rank##########################
    dist = cdist(qf, gf)

    r, m_ap = rank(dist)

    print('[Without Re-Ranking] mAP: {:.4f} rank1: {:.4f} rank3: {:.4f} rank5: {:.4f} rank10: {:.4f}'
            .format(m_ap, r[0], r[2], r[4], r[9]))