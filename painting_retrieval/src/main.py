import glob

import numpy as np
import imageio

from descriptors import descriptor
from similarity_measures import similarity


def main():
    image = imageio.imread(np.random.choice(glob.glob('../data/museum_set_random/*.jpg')))
    query = imageio.imread(np.random.choice(glob.glob('../data/query_devel_random/*.jpg')))

    u = descriptor(image)
    v = descriptor(query)

    print(similarity(u, v))


if __name__ == '__main__':
    main()
