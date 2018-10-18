#!/usr/bin/env python3
import os, glob
import data_analysis as da
import numpy as np
import imageio
import cv2
from scipy.signal import correlate
from skimage import transform

SIZE_MAX = 30000
SIZE_MIN = 460
FILLING_RATIOS = [0.49, 0.74, 1.0]


def ccl_window_evaluation(mask, bbox):
    window_size = da.size(mask, bbox)
    window_filling_ratio = da.filling_ratio(mask, bbox)
    threshold = 0.1
    dist = min(abs(np.array(FILLING_RATIOS) - window_filling_ratio))
    if dist < threshold and SIZE_MIN < window_size < SIZE_MAX:
        return True
    return False


def integral_image_window_evaluation(window_size, bbox):
    tly, tlx, bry, brx = bbox
    width = brx - tlx
    height = bry - tly
    bbox_area = width * height
    window_filling_ratio = window_size / bbox_area
    threshold = 0.1
    dist = min(abs(np.array(FILLING_RATIOS) - window_filling_ratio))
    if dist < threshold and SIZE_MIN < window_size < SIZE_MAX:
        return True
    return False

def correlation_coefficient(patch1, patch2):
    product = np.mean((patch1 - patch1.mean()) * (patch2 - patch2.mean()))
    stds = patch1.std() * patch2.std()
    if stds == 0:
        return 0
    else:
        product /= stds
        return product


def template_matching_evaluation(mask, template, bbox):
    window = mask[bbox[0]:bbox[2], bbox[1]:bbox[3]]
    threshold = 0.6
    #matched = cv2.matchTemplate(window, template, cv2.TM_CCORR_NORMED)  # returns float32
    matched = correlation_coefficient(window, template)
    if abs(matched) > threshold:
        return True
    return False


def main():
    template = imageio.imread('shape_templates/circle.png')

    for img_file in sorted(glob.glob('data/train/*.jpg')):
        name = os.path.splitext(os.path.split(img_file)[1])[0]
        mask_file = 'data/train/mask/mask.{}.png'.format(name)
        gt_file = 'data/train/gt/gt.{}.txt'.format(name)
        img = imageio.imread(img_file, as_gray = True)
        img = np.round(img).astype(np.uint8)
        mask = imageio.imread(mask_file)
        gts = [line.split(' ') for line in open(gt_file, 'r').read().splitlines()]
        for gt in gts:
            bbox = np.round(list(map(int, map(float, gt[:4]))))
            template = transform.resize(template, (bbox[2]-bbox[0], bbox[3]-bbox[1]), preserve_range=True)
            template = np.round(template).astype(np.uint8)
            #print(ccl_window_evaluation(mask, bbox))
            template_matching_evaluation(img, template, bbox)


if __name__ == '__main__':
    main()
