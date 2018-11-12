import glob

import numpy as np
import imageio
import cv2

from matplotlib import pyplot as plt

from timer import Timer


img1 = imageio.imread(np.random.choice(glob.glob('../data/query_devel_random/*.jpg')))
img2 = imageio.imread(np.random.choice(glob.glob('../data/museum_set_random/*.jpg')))

# Initiate SIFT detector
#sift = cv2.SIFT()
sift = cv2.xfeatures2d.SIFT_create()

# find the keypoints and descriptors with SIFT
with Timer('detect'):
    kp1 = sift.detect(img1)
    kp2 = sift.detect(img2)
with Timer('compute'):
    kp1, des1 = sift.compute(img1, kp1)
    kp2, des2 = sift.compute(img2, kp2)

# BFMatcher with default params
with Timer('knnMatch'):
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

# ratio test as per Lowe's paper
good = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good.append([m])

print('{}/{}'.format(len(good), len(kp1)))

# cv2.drawMatchesKnn expects list of lists as matches.
img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)

plt.imshow(img3)
plt.show()
