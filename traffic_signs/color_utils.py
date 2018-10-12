#!/usr/bin/env python3
import os
import glob
import colorsys
from collections import Counter
import multiprocessing as mp

import imageio
import numpy as np
from sklearn.cluster import KMeans

import matplotlib as mpl
from matplotlib import pyplot as plt


def dominant_colors(img, mask, bbox, k=7, n=2):
    tly, tlx, bry, brx = bbox

    img_patch = img[tly:bry,tlx:brx]
    mask_patch = mask[tly:bry,tlx:brx]
    mask_patch = np.repeat(mask_patch[:,:,np.newaxis], 3, axis=2)
    pixels = img_patch[np.nonzero(mask_patch)].reshape((-1, 3))

    clt = KMeans(n_clusters=k)
    labels = clt.fit_predict(pixels)

    c = Counter(labels)
    most_common, _ = zip(*c.most_common(n))
    most_common = list(most_common)

    return clt.cluster_centers_[most_common].astype(np.uint8)


def pixel_colors(img, mask, bbox):
    tly, tlx, bry, brx = bbox

    img_patch = img[tly:bry, tlx:brx]
    mask_patch = mask[tly:bry, tlx:brx]
    mask_patch = np.repeat(mask_patch[:, :, np.newaxis], 3, axis=2)
    pixels = img_patch[np.nonzero(mask_patch)].reshape((-1, 3))

    return pixels

def rgb2ihsl(R,G,B):
    """
    Convert from RGB to IHSL color space. IHSL stands for Improved HSL color space (H. Fleyeh).
    """

    #R = im[0]
    #G = im[ 1]
    #B = im[2]

    numerador = R - G / 2 - B / 2
    denominador = np.sqrt(np.square(R) + np.square(G) + np.square(B) - R * G - R * B - G * B)
    theta = np.arccos(numerador / denominador)

    if G >= B:
        H= np.nan_to_num(theta)
    else:
        H = 360 - theta
    S = max(R, G, B) - min(R, G, B)
    L = 0.212 * R + 0.715 * G + 0.072 * B

    return H, S, L

def worker(img_file):
    name = os.path.splitext(os.path.split(img_file)[1])[0]
    mask_file = 'data/train/mask/mask.{}.png'.format(name)
    gt_file = 'data/train/gt/gt.{}.txt'.format(name)
    img = imageio.imread(img_file)
    mask = imageio.imread(mask_file)
    gts = [line.split(' ') for line in open(gt_file, 'r').read().splitlines()]
    colors = []
    for gt in gts:
        bbox = np.round(list(map(int, map(float, gt[:4]))))
        colors.extend(dominant_colors(img, mask, bbox))
    return colors


def reference_colors(images):
    with mp.Pool(8) as p:
        r = p.map(worker, images)
    rgb_colors = np.concatenate(r)

    hsv_colors = []
    for rgb in rgb_colors:
        hsv = list(colorsys.rgb_to_hsv(*(rgb/255)))
        hsv[2] = 1
        hsv_colors.append(hsv)

    clt = KMeans(n_clusters=3)
    labels = clt.fit_predict(hsv_colors)

    return clt.cluster_centers_

def reference_colors_ihsl(images):
    with mp.Pool(8) as p:
        r = p.map(worker, images)
    rgb_colors = np.concatenate(r)

    hsv_colors = []
    for rgb in rgb_colors:
        print(rgb)
        hsv = list(rgb2ihsl(*(rgb / 255)))
        hsv[2] = 1
        hsv_colors.append(hsv)

    clt = KMeans(n_clusters=3)
    labels = clt.fit_predict(hsv_colors)

    return clt.cluster_centers_


def hue_histograms(images):
    with mp.Pool(8) as p:
        r = p.map(worker, images)
    rgb_colors = np.concatenate(r)

    hsv_colors = []
    for rgb in rgb_colors:
        hsv = list(colorsys.rgb_to_hsv(*(rgb / 255)))
        hsv[2] = 1
        hsv_colors.append(hsv)

    clt = KMeans(n_clusters=3)
    labels = clt.fit_predict(hsv_colors)
    for k in range(3):
        color_list = [i for (i, v) in zip(hsv_colors, labels) if v == k]
        H_list = [color[0]*255 for color in color_list]
        fig = plt.figure(figsize=(6, 1), frameon=False)
        plt.hist(H_list)
        m = np.mean(H_list)
        std = np.std(H_list)
        print('Color {0}: H in [{1}, {2}]'.format(k, m-std/2, m+std/2))

        plt.title(k)


def rgb_histograms(images):
    with mp.Pool(8) as p:
        r = p.map(worker, images)
    rgb_colors = np.concatenate(r)

    clt = KMeans(n_clusters=3)
    labels = clt.fit_predict(rgb_colors)
    channel = ['R', 'G', 'B']
    m = [0, 0, 0]
    for k in range(3):
        color_values = [i for (i, v) in zip(rgb_colors, labels) if v == k]
        for col in range(3):
            col_list = [color[col] for color in color_values]
            m[col] = np.mean(col_list)
            std = np.std(col_list)
            fig = plt.figure(figsize=(6, 1), frameon=False)
            plt.hist(col_list)
            plt.title('{0}: Channel {1}'.format(k, channel[col]))
            print('Channel {0}: values in [{1}, {2}]'.format(channel[col], m[col] - std, m[col] + std))

        print(m)
        plt.show()


def show_colors(colors):
    fig = plt.figure(figsize=(6, 1), frameon=False)
    ax = fig.add_subplot(111)
    for x, color in enumerate(colors):
        ax.add_patch(mpl.patches.Rectangle((x, 0), 1, 1, facecolor=color))
    ax.set_xlim((0, len(colors)))
    ax.set_ylim((0, 1))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')


def main():
    images = glob.glob('data/train/*.jpg')

    ref_colors = reference_colors_ihsl(images)
    [print('({:.4f}, {:.4f}, {:.4f})'.format(*c)) for c in ref_colors]

    #rgb_histograms(images)
    #hue_histograms(images)


if __name__ == '__main__':
    main()
