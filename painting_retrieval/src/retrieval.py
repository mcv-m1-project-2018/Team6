import multiprocessing.dummy as mp

import imageio

from keypoints import extract_keypoints
from descriptors import extract_local_descriptors
from distances import match_descriptors


def _read_and_extract(image_file, keypoint_method, descriptor_method):
    image = imageio.imread(image_file)
    keypoints = extract_keypoints(image, keypoint_method)
    descriptors = extract_local_descriptors(image, keypoints, descriptor_method)
    return descriptors


def query(query_file, image_files, keypoint_method, descriptor_method, distance_metric, k=10):
    query_embd = _read_and_extract(query_file, keypoint_method, descriptor_method)

    with mp.Pool(processes=20) as p:
        image_descriptors = p.starmap(_read_and_extract, [(image_file, keypoint_method, descriptor_method) for image_file in image_files])
        matches = p.starmap(match_descriptors, [(query_embd, image_embd, distance_metric) for image_embd in image_descriptors])

    matches = sorted(matches, key=lambda x: x.distance)[:k]
    result = [(image_files[match.imgIdx], match.distance) for match in matches]
    return result


def query_batch(query_files, image_files, keypoint_method, descriptor_method, distance_metric, k=10):
    results = []
    with mp.Pool(processes=20) as p:
        query_descriptors = p.starmap(_read_and_extract, [(query_file, keypoint_method, descriptor_method) for query_file in query_files])
        image_descriptors = p.starmap(_read_and_extract, [(image_file, keypoint_method, descriptor_method) for image_file in image_files])

        for query_embd in query_descriptors:
            matches = p.starmap(match_descriptors, [(query_embd, image_embd, distance_metric) for image_embd in image_descriptors])
            matches = sorted(matches, key=lambda x: x.distance)[:k]
            result = [(image_files[match.imgIdx], match.distance) for match in matches]
            results.append(result)
    return results
