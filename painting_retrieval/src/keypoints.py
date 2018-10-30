import cv2
import numpy as np

def _keypoints(image):
    """
    Extract keypoints of an image.

    Args:
        image (ndarray): (H x W x C) 3D array of type np.uint8 containing an image.

    Returns:
        list: list of keypoints.

    """

    pass


# KEYPOINT DETECTORS


def harris_corner_detector(image):
    """
    Extract keypoints from image using Harris Corner Detector
    Args:
        image (ndarray): (H x W x C) 3D array of type np.uint8 containing an image.

    Returns:
        ndarray: list of 1D arrays of type np.float32 containing image descriptors.
    """

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    # cornerHarris function takes as arguments the image, blockSize, ksize (aperture parameter of Sobel derivative),
    # k (Harris detector free parameter, which goes from 0.04 to 0.06)
    # If ksize = -1, a 3x3 Scharr filter is used which gives better results than 3x3 Sobel filter
    dst = cv2.cornerHarris(gray, 4, -1, 0.05)

    corners = np.argwhere(dst > dst.max() * 0.01)

    # result is dilated for marking the corners, not important
    #dst = cv2.dilate(dst, None)

    # Threshold for an optimal value, it may vary depending on the image. (It gives us the corners in the image)
    #image[dst > 0.01 * dst.max()] = [0, 0, 255]

    #cv2.imshow('dst', image)
    #if cv2.waitKey(0) & 0xff == 27:
    #    cv2.destroyAllWindows()

    return [cv2.KeyPoint(corner[0], corner[1], 10) for corner in corners]


def harris_corner_subpixel_accuracy(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # find Harris corners
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)
    dst = cv2.dilate(dst, None)
    ret, dst = cv2.threshold(dst, 0.01 * dst.max(), 255, 0)
    dst = np.uint8(dst)

    # find centroids
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)

    # define the criteria to stop and refine the corners
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv2.cornerSubPix(gray, np.float32(centroids), (5, 5), (-1, -1), criteria)

    return [cv2.KeyPoint(corner[0], corner[1], 5) for corner in corners]

def detect_keypoints(image, method):
    func = {
        'dog': lambda: None
    }
    return func[method](image)
