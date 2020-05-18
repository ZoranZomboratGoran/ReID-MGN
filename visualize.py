from opt import opt
from utils.extract_feature import extract_feature
from utils.metrics import mean_ap, cmc, re_ranking

from tqdm import tqdm
import os
import numpy as np
import matplotlib
matplotlib.use('agg')
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import torch

def load_network(model):

    load_filename = opt.weight
    checkpoint = torch.load(load_filename)
    model.load_state_dict(checkpoint['model'])
    if opt.usecpu == False and torch.cuda.is_available():
        model.cuda()


def visualize(app):

    load_network(app.model)
    app.model.eval()

    gallery_path = app.data.testset.imgs
    gallery_label = app.data.testset.ids

    # Extract feature
    print('extract features, this may take a few minutes')
    query_image = app.data.query_image
    if opt.usecpu == False and torch.cuda.is_available():
        query_image = query_image.cuda()
    query_feature = extract_feature(app.model, tqdm([(torch.unsqueeze(app.data.query_image, 0), 1)]))
    gallery_feature = extract_feature(app.model, tqdm(app.test_loader))

    query_label = app.data.queryset.id(app.data.query_image_path)

    # sort images
    query_feature = query_feature.view(-1, 1)
    score = torch.mm(gallery_feature, query_feature)
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

    print('Top 10 images are as follow:')

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

    fig.savefig("show.png")
    print('result saved to show.png')