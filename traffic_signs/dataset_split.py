#!/usr/bin/env python3
import os
import glob
import argparse
from collections import defaultdict

import numpy as np


def main(args):
    images_per_class = defaultdict(list)
    for image_file in sorted(glob.glob(os.path.join(args.data_path, '*.jpg'))):
        name = os.path.splitext(os.path.split(image_file)[1])[0]
        mask_file = os.path.join(args.data_path, 'mask/mask.{}.png'.format(name))
        gt_file = os.path.join(args.data_path, 'gt/gt.{}.txt'.format(name))
        gts = [line.split(' ') for line in open(gt_file, 'r').read().splitlines()]
        labels = [gt[4] for gt in gts]
        label = np.random.choice(labels)
        images_per_class[label].append(image_file)

    train, val = [], []
    for label in sorted(images_per_class.keys()):
        image_files = images_per_class[label]
        np.random.shuffle(image_files)  # TODO: improve intra-class split
        c = int(np.ceil(len(image_files) * args.train_pct))
        train.extend(image_files[:c])
        val.extend(image_files[c:])
        print('{}: {}/{}'.format(label, c, len(image_files) - c))

    with open(os.path.join(args.output_path, 'train.txt'), 'w') as f:
        f.write('\n'.join(train))

    with open(os.path.join(args.data_path, 'val.txt'), 'w') as f:
        f.write('\n'.join(val))
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data/train')
    parser.add_argument('--train_pct', type=float, default=0.7)
    parser.add_argument('--output_path', type=str, default='.')
    args = parser.parse_args()
    main(args)