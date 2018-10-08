#!/usr/bin/env python3
import numpy as np
import imageio
from skimage import color
import cv2


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


# Create your own candidate_generation_pixel_xxx functions for other color spaces/methods
# Add them to the switcher dictionary in the switch_color_space() function
# These functions should take an image as input and output the pixel_candidates mask image


def rgb2ihsl(im):
    """
    Convert from RGB to IHSL color space. IHSL stands for Improved HSL color space (H. Fleyeh).
    """

    R = im[:, :, 0]
    G = im[:, :, 1]
    B = im[:, :, 2]
    R, G, B = R / 255.0, G / 255.0, B / 255.0

    numerador = R - G / 2 - B / 2
    denominador = np.sqrt(np.square(R) + np.square(G) + np.square(B) - R * G - R * B - G * B)
    theta = np.arccos(numerador / denominador)

    H = np.zeros([len(G), len(G[0])])
    S = np.zeros([len(G), len(G[0])])
    #print(len(G), len(G[0]))
    for i in range(len(G)):
        for j in range(len(G[0])):
            if G[i, j] >= G[i, j]:
                H[i, j] = theta[i, j]
            else:
                H[i, j] = 360 - theta[i, j]

            S[i, j] = max(R[i, j], G[i, j], B[i, j]) - min(R[i, j], G[i, j], B[i, j])
    L = 0.212 * R + 0.715 * G + 0.072 * B

    return H, S, L


def candidate_generation_pixel_ihsl1(im):
    """
    Convert from RGB to IHLS and filter pixels by Euclidean distance to reference
    colors. Thresholds are dinamically computed using the global mean of the luminance.

    The method is explained in 'Color Detection And Segmentation For Road And Traffic Signs' (H. Fleyeh).
    """

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


def candidate_generation_pixel_ihsl2(im):
    """
    Convert from RGB to IHLS and normalize H and S channels. Then H and S values
    are discretized and pixels are filtered using the binary mask resulting from
    the logical AND between H and S channels.

    The method is explained in 'Color Detection And Segmentation For Road And Traffic Signs' (H. Fleyeh).
    """

    #convert input image to IHLS color space
    H, S, L = rgb2ihsl(im)
    size = im.shape #(n, m, channels)

    Hout = np.zeros((size[0], size[1]))
    Sout = np.zeros((size[0], size[1]))
    pixel_candidates = np.zeros((size[0], size[1]))

    #normalize S and H to [0, 255]

    S = 255*S/np.max(S)
    H = np.nan_to_num(H)
    H = 255*H/np.max(H)

    Smin = 51
    Smax = 170

    #the range of H defines the color (we want white, red and blue)
    #values obtained from hue histograms in train images, in color_ranges.py
    Hmin = [146, 2, 104]
    Hmax = [170, 7, 142]

    for i in range(size[0]):
        for j in range(size[1]):
            if S[i,j] <= Smin:
                Sout[i,j] = 0
            elif S[i,j] < Smax:
                Sout[i, j] = S[i, j]
            else:
                Sout[i, j] = 255
            for k in range(len(Hmin)):
                if H[i,j] >= Hmin[k] or H[i,j] <= Hmax[k]:
                    Hout[i, j] = max(255, Hout[i, j])

            pixel_candidates[i,j] = (Sout[i,j] and Hout[i,j])

    return pixel_candidates


def candidate_generation_pixel_hsv_euclidean(rgb):
    """
    Convert from RGB to HSV and filter pixels by Euclidean distance to reference
    values based on H and S. Thresholds are computed empirically.
    """

    ref_colors = [[0.0182, 0.6667, 1.0000], [0.6118, 0.6879, 1.0000]]
    thresholds = [0.25, 0.3]

    hsv = color.rgb2hsv(rgb)

    masks = []
    for ref, thresh in zip(ref_colors, thresholds):
        ref = np.tile(ref, hsv.shape[:2] + (1,))
        h1 = hsv[:,:,0]
        s1 = hsv[:,:,1]
        h2 = ref[:,:,0]
        s2 = ref[:,:,1]
        dist = ((h2-h1)**2 + (s2-s1)**2)**0.5
        mask = dist < thresh

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
        mask = cv2.medianBlur(mask, 7)

        masks.append(mask)

    pixel_candidates = np.zeros(rgb.shape[:2], dtype=np.uint8)
    for mask in masks:
        pixel_candidates += mask
    pixel_candidates = np.clip(pixel_candidates, 0, 1)
    return pixel_candidates


def candidate_generation_pixel_rgb(im):
    """
    Consider pixels in RGB space, pixels whose values are near the three most
    common RGB colors in the traffic signals are kept.
    """

    R = im[:, :, 0]
    G = im[:, :, 1]
    B = im[:, :, 2]

    col0_candidates = (R > 19) & (R < 97) & (G > 61) & (G < 120) & (B > 94) & (B < 161);
    col1_candidates = (R > 143) & (R < 231) & (G > 186) & (G < 253) & (B > 178) & (B < 252);
    col2_candidates = (R > 18) & (R < 104) & (G > 23) & (G < 55) & (B > 23) & (B < 67);

    pixel_candidates = col0_candidates | col1_candidates | col2_candidates;

    return pixel_candidates


def switch_color_space(im, color_space):
    switcher = {
        'normrgb': candidate_generation_pixel_normrgb,
        'hsv': candidate_generation_pixel_hsv,
        #'lab'    : candidate_generation_pixel_lab,
        'ihsl_1': candidate_generation_pixel_ihsl1,
        'ihsl_2': candidate_generation_pixel_ihsl2,
        'hsv_euclidean': candidate_generation_pixel_hsv_euclidean,
        'rgb': candidate_generation_pixel_rgb
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
    pixel_candidates4 = candidate_generation_pixel(im, 'ihsl_2')
    pixel_candidates4 = candidate_generation_pixel(im, 'hsv_euclidean')
    pixel_candidates5 = candidate_generation_pixel(im, 'rgb')
