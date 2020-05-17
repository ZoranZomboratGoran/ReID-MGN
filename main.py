import os
import numpy as np
import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt

from opt import opt
from data import Data
from network import MGN
from loss import Loss

from app import App
from train import train_model
from evaluate import evaluate_model, evaluate_rerank
from visualize import visualize

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

if __name__ == '__main__':

    data = Data()
    model = MGN()
    loss = Loss()
    app = App(model, loss, data)

    if opt.mode == 'train':
        train_model(app)

    if opt.mode == 'evaluate':
        evaluate_rerank(app)

    if opt.mode == 'visualize':
        visualize(app)
