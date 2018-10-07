#!/usr/bin/env python3
import os
import glob
import argparse
import multiprocessing as mp

import numpy as np
import imageio

from candidate_generation_pixel import candidate_generation_pixel


def worker(x):
    image_file, output_dir, pixel_method = x
    
    name = os.path.splitext(os.path.split(image_file)[1])[0]

    im = imageio.imread(image_file)
    print(image_file)
    
    pixel_candidates = candidate_generation_pixel(im, pixel_method)
    
    fd = os.path.join(output_dir, pixel_method)
    if not os.path.exists(fd):
        os.makedirs(fd)
        
    out_mask_name = os.path.join(fd, name + '.png')
    imageio.imwrite (out_mask_name, np.uint8(np.round(pixel_candidates)))


def generate_masks(images_dir, output_dir, pixel_method):
    images = sorted(glob.glob(os.path.join(images_dir, '*.jpg')))
    
    with mp.Pool(processes=8) as p:
        p.map(worker, [(image_file, output_dir, pixel_method) for image_file in images])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('images_dir')
    parser.add_argument('output_dir')
    parser.add_argument('pixel_method')
    args = parser.parse_args()
    
    generate_masks(args.images_dir, args.output_dir, args.pixel_method)