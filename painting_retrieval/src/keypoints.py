import numpy as np
from skimage import feature
import cv2
import imageio


def _keypoints(image):
    """
    Extract keypoints of an image.

    Args:
        image (ndarray): (H x W) 2D array of type np.uint8 containing a grayscale image.

    Returns:
        list: list of keypoints.

    """

    pass


def sift_keypoints(image):
    """
    Extract keypoints of an image using Difference of Gaussians method.

    Args:
        image (ndarray): (H x W) 2D array of type np.uint8 containing a grayscale image.

    Returns:
        (list of cv2.KeyPoint objects): list of keypoints.

    """

    sift = cv2.xfeatures2d.SIFT_create()
    keypoints = sift.detect(image)
    return keypoints


def laplacian_of_gaussian(image):
    """
    Extract keypoints of an image using Laplacian of Gaussians method.

    Args:
        image (ndarray): (H x W) 2D array of type np.uint8 containing a grayscale image.

    Returns:
        (list of cv2.KeyPoint objects): list of keypoints.

    """

    blobs_log = feature.blob_log(image, max_sigma=30, num_sigma=10, threshold=.1)
    points2f = np.array(blobs_log[:, [0, 1]], dtype=np.float32)
    sizes = np.array(list(blobs_log[:, 2] * np.sqrt(2) * 2), dtype=np.float32)
    keypoints = []
    for i in range(len(points2f)):
        keypoints.append(cv2.KeyPoint_convert([points2f[i]], sizes[i])[0])
    return keypoints


def difference_of_gaussian(image):
    """
    Extract keypoints of an image using Difference of Gaussians method.

    Args:
        image (ndarray): (H x W) 2D array of type np.uint8 containing a grayscale image.

    Returns:
        (list of cv2.KeyPoint objects): list of keypoints.

    """

    blobs_dog = feature.blob_dog(image, max_sigma=30, threshold=.1)
    points2f = np.array(blobs_dog[:, [0, 1]], dtype=np.float32)
    sizes = np.array(list(blobs_dog[:, 2] * np.sqrt(2) * 2), dtype=np.float32)
    keypoints = []
    for i in range(len(points2f)):
        keypoints.append(cv2.KeyPoint_convert([points2f[i]], sizes[i])[0])
    return keypoints


def determinant_of_hessian(image):
    """
    Extract keypoints of an image using Determinant of Hessian method.

    Args:
        image (ndarray): (H x W) 2D array of type np.uint8 containing a grayscale image.

    Returns:
        (list of cv2.KeyPoint objects): list of keypoints.

    """

    blobs_doh = feature.blob_doh(image, max_sigma=30, threshold=.001)
    points2f = np.array(blobs_doh[:, [0, 1]], dtype=np.float32)
    sizes = np.array(list(blobs_doh[:, 2] * np.sqrt(2) * 2), dtype=np.float32)
    keypoints = []
    for i in range(len(points2f)):
        keypoints.append(cv2.KeyPoint_convert([points2f[i]], sizes[i])[0])
    return keypoints


# KEYPOINT DETECTORS


def harris_corner_detector(image):
    """
    Extract keypoints from image using Harris Corner Detector
    Args:
        image (ndarray): (H x W) 2D array of type np.uint8 containing a grayscale image.

    Returns:
        ndarray: list of 1D arrays of type np.float32 containing image descriptors.
    """

    dst = cv2.cornerHarris(image, 4, -1, 0.04)

    corners = np.argwhere(dst > dst.max() * 0.10)

    return [cv2.KeyPoint(corner[0], corner[1], 9) for corner in corners]


def harris_corner_subpixel_accuracy(image):
    """
    Extract keypoints from image using Harris Corner Detector with subpixel accuracy
    Args:
        image (ndarray): (H x W) 2D array of type np.uint8 containing a grayscale image.

    Returns:
        ndarray: list of 1D arrays of type np.float32 containing image descriptors.
    """

    # find Harris corners
    dst = cv2.cornerHarris(image, 4, -1, 0.04)
    dst = cv2.dilate(dst, None)
    ret, dst = cv2.threshold(dst, 0.10 * dst.max(), 255, 0)
    dst = np.uint8(dst)

    # find centroids
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)

    # define the criteria to stop and refine the corners
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv2.cornerSubPix(image, np.float32(centroids), (2, 2), (-1, -1), criteria)

    return [cv2.KeyPoint(corner[0], corner[1], 4) for corner in corners]


def detect_keypoints(image, method):
    func = {
        'dog': difference_of_gaussian,
        'log': laplacian_of_gaussian,
        'doh': determinant_of_hessian,
        'sift': sift_keypoints,
        'harris_corner_detector': harris_corner_detector,
        'harris_corner_subpixel': harris_corner_subpixel_accuracy
    }
    return func[method](image)


if __name__ == '__main__':
    image = imageio.imread('../data/query_devel_random/ima_000008.jpg')
    h = harris_corner_subpixel_accuracy(image)
    print(h)