#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
from skimage import color


def candidate_generation_pixel_normrgb(im):
    # convert input image to the normRGB color space

    normrgb_im = np.zeros(im.shape)
    eps_val = 0.00001
    norm_factor_matrix = im[:, :, 0] + im[:, :, 1] + im[:, :, 2] + eps_val

    normrgb_im[:, :, 0] = im[:, :, 0] / norm_factor_matrix
    normrgb_im[:, :, 1] = im[:, :, 1] / norm_factor_matrix
    normrgb_im[:, :, 2] = im[:, :, 2] / norm_factor_matrix

    # Develop your method here:
    # Example:
    pixel_candidates = normrgb_im[:, :, 1] > 100;

    return pixel_candidates


def candidate_generation_pixel_hsv(im):
    # convert input image to HSV color space
    hsv_im = color.rgb2hsv(im)

    # Develop your method here:
    # Example:
    pixel_candidates = hsv_im[:, :, 1] > 0.4;

    return pixel_candidates


def candidate_generation_pixel_ihsl1(im):
    # convert input image to IHSL color space
    H, S, L = rgb2ihsl(im)

    # set color references as White, Red and Blue:
    SW = 0.1891
    HW = 0.4846
    SR = 0.6111
    HR = 0.0196
    SB = 0.6459
    HB = 0.6219

    # compute threshold given the mean lightness of the image
    m = len(L)
    n = len(L[0])
    suma = 0
    for i in range(m):
        for j in range(n):
            suma += L[i, j]
    mean = suma / (m * n)
    Nmean = mean / 256
    threshold = np.exp(-Nmean)

    # compute Euclidean distance from pixel of input image to color reference values (white, red and blue)
    Wx = (SW * np.cos(HW) - S * np.cos(H)) ** 2 + (SW * np.sin(HW) - S * np.sin(H)) ** 2
    Rx = (SR * np.cos(HR) - S * np.cos(H)) ** 2 + (SR * np.sin(HR) - S * np.sin(H)) ** 2
    Bx = (SB * np.cos(HB) - S * np.cos(H)) ** 2 + (SB * np.sin(HB) - S * np.sin(H)) ** 2
    Ed_W = np.nan_to_num(np.power(np.sqrt(Wx), 0.5))
    Ed_R = np.nan_to_num(np.power(np.sqrt(Rx), 0.5))
    Ed_B = np.nan_to_num(np.power(np.sqrt(Bx), 0.5))

    # pixel candidates are those where the Euclidean distance is higher than the threshold
    Ed_W[Ed_W < threshold] = 0
    Ed_R[Ed_R < threshold] = 0
    Ed_B[Ed_B < threshold] = 0

    pixel_candidates = Ed_W + Ed_R + Ed_B
    pixel_candidates[pixel_candidates > 0] = 1

    return pixel_candidates


def rgb2ihsl(im):
    # Convert from RGB color space to IHSL color space
    # IHSL stands for Improved HSL color space (H.Fleyeh)

    R = im[:, :, 0]
    G = im[:, :, 1]
    B = im[:, :, 2]
    R, G, B = R / 255.0, G / 255.0, B / 255.0

    numerador = R - G / 2 - B / 2
    denominador = np.sqrt(np.square(R) + np.square(G) + np.square(B) - R * G - R * B - G * B)
    theta = np.arccos(numerador / denominador)

    H = np.zeros([len(G), len(G[0])])
    S = np.zeros([len(G), len(G[0])])
    print(len(G), len(G[0]))
    for i in range(len(G)):
        for j in range(len(G[0])):
            if G[i, j] >= G[i, j]:
                H[i, j] = theta[i, j]
            else:
                H[i, j] = 360 - theta[i, j]

            S[i, j] = max(R[i, j], G[i, j], B[i, j]) - min(R[i, j], G[i, j], B[i, j])
    L = 0.212 * R + 0.715 * G + 0.072 * B

    return H, S, L


# Create your own candidate_generation_pixel_xxx functions for other color spaces/methods
# Add them to the switcher dictionary in the switch_color_space() function
# These functions should take an image as input and output the pixel_candidates mask image

def switch_color_space(im, color_space):
    switcher = {
        'normrgb': candidate_generation_pixel_normrgb,
        'hsv': candidate_generation_pixel_hsv,
        'ihsl_1': candidate_generation_pixel_ihsl1
        # 'lab'    : candidate_generation_pixel_lab,
    }
    # Get the function from switcher dictionary
    func = switcher.get(color_space, lambda: "Invalid color space")

    # Execute the function
    pixel_candidates = func(im)

    return pixel_candidates


def candidate_generation_pixel(im, color_space):
    pixel_candidates = switch_color_space(im, color_space)

    return pixel_candidates


if __name__ == '__main__':
    pixel_candidates1 = candidate_generation_pixel(im, 'normrgb')
    pixel_candidates2 = candidate_generation_pixel(im, 'hsv')
    pixel_candidates3 = candidate_generation_pixel(im, 'ihsl_1')


