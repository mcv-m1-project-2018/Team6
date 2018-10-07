#!/usr/bin/env python3
import os
import glob
import argparse
import shutil
from collections import defaultdict

import numpy as np
import imageio


def get_patch_lightness(img, mask, bbox):
    tly, tlx, bry, brx = bbox
    img_patch = img[tly:bry,tlx:brx]
    mask_patch = mask[tly:bry,tlx:brx]
    mask_patch = np.repeat(mask_patch[:,:,np.newaxis], 3, axis=2)
    pixels = img_patch[np.nonzero(mask_patch)].reshape((-1, 3))
    
    norm_rgb = pixels / 255
    c_max = norm_rgb.max(axis=1)
    c_min = norm_rgb.min(axis=1)
    L = (c_max + c_min) / 2
    
    return np.mean(L)


def split(image_files, mask_files, gt_files):
    images_per_class = defaultdict(list)
    lightness = {}
    for image_file, mask_file, gt_file in zip(image_files, mask_files, gt_files):
        img = imageio.imread(image_file)
        mask = imageio.imread(mask_file)
        gts = [line.split(' ') for line in open(gt_file, 'r').read().splitlines()]
        gt = gts[np.random.choice(range(len(gts)))]
        bbox = np.round(list(map(int, map(float, gt[:4]))))
        label = gt[4]
        images_per_class[label].append(image_file)
        lightness[image_file] = get_patch_lightness(img, mask, bbox)

    train, val = [], []
    for label in sorted(images_per_class.keys()):
        class_images = images_per_class[label]
        np.random.shuffle(class_images)
        
        # intra-class split considering lightness
        class_images = sorted(class_images, key=lambda x: lightness[x])
        bins = 10
        bin_size = int(np.ceil(len(class_images)/bins))
        c = int(np.ceil(bin_size * args.train_pct))
        for i in range(0, len(class_images), bin_size):
            train.extend(class_images[i:i+c])
            val.extend(class_images[i+c:i+bin_size])
            
    return train, val


def main(args):
    image_files = glob.glob(os.path.join(args.data_path, '*.jpg'))
    mask_files, gt_files = [], []
    for image_file in image_files:
        name = os.path.splitext(os.path.split(image_file)[1])[0]
        mask_files.append(os.path.join(args.data_path, 'mask/mask.{}.png'.format(name)))
        gt_files.append(os.path.join(args.data_path, 'gt/gt.{}.txt'.format(name)))
    
    train, val = split(image_files, mask_files, gt_files)
    print('{}/{}'.format(len(train), len(val)))

    #with open(os.path.join(args.output_path, 'train.txt'), 'w') as f:
    #    f.write('\n'.join(train))
    #    
    #with open(os.path.join(args.output_path, 'val.txt'), 'w') as f:
    #    f.write('\n'.join(val))
    
    train_path = 'data/train_val/train'
    train_mask_path = 'data/train_val/train/mask'
    train_gt_path = 'data/train_val/train/gt'
    val_path = 'data/train_val/val'
    val_mask_path = 'data/train_val/val/mask'
    val_gt_path = 'data/train_val/val/gt'
    for path in [train_path, train_mask_path, train_gt_path, val_path, val_mask_path, val_gt_path]:
        if not os.path.exists(path):
            os.makedirs(path)
    
    for src_image_file in train:
        name = os.path.splitext(os.path.split(src_image_file)[1])[0]
        src_mask_file = os.path.join('data/train/mask/mask.{}.png'.format(name))
        src_gt_file = os.path.join('data/train/gt/gt.{}.txt'.format(name))
        dst_image_file = os.path.join(train_path, os.path.split(src_image_file)[1])
        dst_mask_file = os.path.join(train_mask_path, os.path.split(src_mask_file)[1])
        dst_gt_file = os.path.join(train_gt_path, os.path.split(src_gt_file)[1])
        shutil.copyfile(src_image_file, dst_image_file)
        shutil.copyfile(src_mask_file, dst_mask_file)
        shutil.copyfile(src_gt_file, dst_gt_file)
        print('copy: {} -> {}'.format(src_image_file, dst_image_file))

    for src_image_file in val:
        name = os.path.splitext(os.path.split(src_image_file)[1])[0]
        src_mask_file = os.path.join('data/train/mask/mask.{}.png'.format(name))
        src_gt_file = os.path.join('data/train/gt/gt.{}.txt'.format(name))
        dst_image_file = os.path.join(val_path, os.path.split(src_image_file)[1])
        dst_mask_file = os.path.join(val_mask_path, os.path.split(src_mask_file)[1])
        dst_gt_file = os.path.join(val_gt_path, os.path.split(src_gt_file)[1])
        shutil.copyfile(src_image_file, dst_image_file)
        shutil.copyfile(src_mask_file, dst_mask_file)
        shutil.copyfile(src_gt_file, dst_gt_file)
        print('copy: {} -> {}'.format(src_image_file, dst_image_file))
        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data/train')
    parser.add_argument('--train_pct', type=float, default=0.7)
    parser.add_argument('--output_path', type=str, default='.')
    args = parser.parse_args()
    main(args)