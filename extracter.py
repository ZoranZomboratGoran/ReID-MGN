from tensorboard.backend.event_processing import event_accumulator
import sys
import pandas as pd
import os
import numpy as np
from collections import defaultdict
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

size_guidance = {
        event_accumulator.COMPRESSED_HISTOGRAMS: 1,
        event_accumulator.IMAGES: 1,
        event_accumulator.AUDIO: 1,
        event_accumulator.SCALARS: 0,
        event_accumulator.HISTOGRAMS: 1,
    }

fig_dir = "figures"
os.makedirs(fig_dir, exist_ok=True)

def extract_and_plot_reid(log_file, fig_file, adapt = False):
    ea = event_accumulator.EventAccumulator(log_file, size_guidance = size_guidance)
    ea.Reload()

    if adapt:
        adapt_loss_list = [
            [1.0, 1.0],
            [187/1600, 187/200],
            [1.0, 1.0]
        ]
    else:
        adapt_loss_list = [
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0]
        ]

    loss_tags = [
        ['ce_loss/epoch/train', 'ce_loss/epoch/validate'],
        ['triplet_loss/epoch/train', 'triplet_loss/epoch/validate'],
        ['total_loss/epoch/train', 'total_loss/epoch/validate']
    ]

    titles = ['Cross entropy loss', 'Triplet loss', 'Total loss']

    plt.figure(figsize=(20,10))
    for idx, loss_tag_entry in enumerate(loss_tags):
        adapt_loss = adapt_loss_list[idx]
        plt.subplot(3, 1, idx + 1)
        for tag_idx, tag in enumerate(loss_tag_entry):
            loss_scalar = ea.Scalars(tag)
            # limit to just the first 100 epochs
            x_vals = [e.step for e in loss_scalar][:100]
            y_vals = [e.value * adapt_loss[tag_idx] for e in loss_scalar][:100]
            plt.plot(x_vals, y_vals, label = tag)
            plt.legend()
        plt.xlabel('epoch', fontsize=14)
        plt.ylabel('loss', fontsize=14)
        plt.title(titles[idx], fontsize = 20)
        plt.grid(True)
        plt.legend(loc='best', fontsize=14)
    plt.tight_layout()

    fig_path = os.path.join(fig_dir, fig_file)
    plt.savefig(fig_path)
    # plt.show()


def extract_and_compare_plot_reid(log_file, log_file_aug, fig_file):

    ea = event_accumulator.EventAccumulator(log_file, size_guidance = size_guidance)
    ea.Reload()

    ea_aug = event_accumulator.EventAccumulator(log_file_aug, size_guidance = size_guidance)
    ea_aug.Reload()

    loss_tags = ['total_loss/epoch/train', 'total_loss/epoch/validate']
    accums = [ea, ea_aug]
    title_aux = ['', '_aug']

    titles = ['Train loss improvement', 'Test loss improvement']

    plt.figure(figsize=(20,10))
    for idx, tag in enumerate(loss_tags):

        plt.subplot(2, 1, idx + 1)
        for acc_idx, accum in enumerate(accums):
            loss_scalar = accum.Scalars(tag)
            # limit to just the first 100 epochs
            x_vals = [e.step for e in loss_scalar][:100]
            y_vals = [e.value for e in loss_scalar][:100]
            plt.plot(x_vals, y_vals, label = loss_tags[idx] + title_aux[acc_idx])
            plt.legend()
        plt.xlabel('epoch', fontsize=14)
        plt.ylabel('loss', fontsize=14)
        plt.title(titles[idx], fontsize = 20)
        plt.grid(True)
        plt.legend(loc='best', fontsize=14)
    plt.tight_layout()

    fig_path = os.path.join(fig_dir, fig_file)
    plt.savefig(fig_path)
    plt.show()


def extract_and_plot_precision(log_file, fig_file):
    ea = event_accumulator.EventAccumulator(log_file, size_guidance = size_guidance)
    ea.Reload()

    prec_tags = [
        ['mAP/epoch'],
        ['rank1/epoch', 'rank5/epoch', 'rank10/epoch']
    ]

    titles = ['Mean average precision', 'TopK ranking precision']

    plt.figure(figsize=(20,10))
    for idx, prec_tag_entry in enumerate(prec_tags):
        plt.subplot(1, 2, idx + 1)
        for _, tag in enumerate(prec_tag_entry):
            prec_scalar = ea.Scalars(tag)
            # limit to just the first 100 epochs
            x_vals = [e.step for e in prec_scalar]
            y_vals = [e.value for e in prec_scalar]
            plt.plot(x_vals, y_vals, label = tag)
            plt.legend()
        plt.xlabel('epoch', fontsize=14)
        plt.ylabel('precision', fontsize=14)
        plt.title(titles[idx], fontsize = 20)
        plt.grid(True)
        plt.legend(loc='best', fontsize=14)

    plt.tight_layout()

    fig_path = os.path.join(fig_dir, fig_file)
    plt.savefig(fig_path)
    plt.show()

def extract_and_compare_plot_precision(log_file, log_file_aug, fig_file):

    ea = event_accumulator.EventAccumulator(log_file, size_guidance = size_guidance)
    ea.Reload()

    ea_aug = event_accumulator.EventAccumulator(log_file_aug, size_guidance = size_guidance)
    ea_aug.Reload()

    prec_tags = ['mAP/epoch', 'rank1/epoch']
    accums = [ea, ea_aug]
    title_aux = ['', '_aug']

    titles = ['Mean average precision improvement', 'Rank 1 precision improvement']

    plt.figure(figsize=(20,10))
    for idx, tag in enumerate(prec_tags):

        plt.subplot(1, 2, idx + 1)
        for acc_idx, accum in enumerate(accums):
            prec_scalar = accum.Scalars(tag)
            # limit to just the first 100 epochs
            x_vals = [e.step for e in prec_scalar]
            y_vals = [e.value for e in prec_scalar]
            plt.plot(x_vals, y_vals, label = prec_tags[idx] + title_aux[acc_idx])
            plt.legend()
        plt.title(titles[idx], fontsize = 20)
        plt.xlabel('epoch', fontsize=14)
        plt.ylabel('precision', fontsize=14)
        plt.grid(True)
        plt.legend(loc='best', fontsize=14)
    plt.tight_layout()

    fig_path = os.path.join(fig_dir, fig_file)
    plt.savefig(fig_path)
    plt.show()

log_file = sys.argv[1]
log_file_aug = sys.argv[2]

extract_and_plot_reid(log_file, "reid_loss_epoch100.png", adapt=False)
# extract_and_plot_reid(log_file_aug, "reid_loss_epoch100_aug.png", adapt=True)
extract_and_compare_plot_reid(log_file, log_file_aug, "reid_loss_epoch100_aug_comp.png")
extract_and_plot_precision(log_file, "mAP_epoch100.png")
# extract_and_plot_precision(log_file_aug, "mAP_epoch100_aug.png")
extract_and_compare_plot_precision(log_file, log_file_aug, "mAP_epoch100_aug_comp.png")


