def _keypoints(image):
    """
    Extract keypoints of an image.

    Args:
        image (ndarray): (H x W x C) 3D array of type np.uint8 containing an image.

    Returns:
        list: list of keypoints.

    """

    pass


def extract_keypoints(image, method):
    func = {
        'dog': lambda: None
    }
    return func[method](image)
