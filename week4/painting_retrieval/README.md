# Painting Retrieval
Introduction to Human and Computer Vision project example code and data for the Painting Retrieval project.

### Execution instructions
The entrypoint for this project is the file `src/main.py`. To execute it, run the following command:

```bash
python src/main.py [--queries_path QUERIES_PATH]
                   [--images_path IMAGES_PATH]
                   [--corresp_file CORRESP_FILE]
                   [--results_path RESULTS_PATH]
                   {eval,test}
```

It can be run in two modes: `eval` and `test`. The `eval` mode will run all the queries in `queries_path` against all the images in `images_path` and compare the results to the true correspondences stored in `corresp_file` in order to compute a _MAP@K_ for every combination of descriptor method and distance metric. The `test` mode will run all the queries in `queries_path` against all the images in `images_path` and store the results in `results_path` in pickle format.

#### Parameters
For each combination (_keypoint\_method_, _descriptor\_method_, _distance\_metric_), the hyperparameters on `distances._filter_matches` and `distances._compute_similarity_score` should be updated correspondingly:

* SIFT-SIFT-L2: 
  * `image_desc=1000`
  * `query_desc=10000`
  * `ratio=0.5`
  * `matches_thresh=10`
  * `dist_thres=130`
* SIFT-SIFT-L1:
  * `image_desc=1000`
  * `query_desc=10000`
  * `ratio=0.5`
  * `matches_thresh=10`
  * `dist_thres=920`
* SURF-SURF-L2:
  * `image_hessianThreshold=1000`
  * `query_hessianThreshold=400`
  * `ratio=0.6`
  * `matches_thresh=19`
  * `dist_thres=0.13`
* SURF-SURF-L1:
  * `image_hessianThreshold=1000`
  * `query_hessianThreshold=400`
  * `ratio=0.5`
  * `matches_thresh=10`
  * `dist_thres=60`
* ORB-ORB-Hamming:
  * `ratio=0.5`
  * `matches_thresh=2`
  * `dist_thres=30`
* HarrisCorner-SIFT-L2:
  * `image_thresh=0.10`
  * `query_thresh=0.15`
  * `matches_thresh=7`
  * `dist_thresh=100`
* HarrisSubpixel-SIFT-L2: 
  * `image_thresh=0.10`
  * `query_thresh=0.10`
  * `matches_thresh=7`
  * `dist_thresh=100`

### Data files
- query_corresp_simple_devel.pkl: True correspondences between query images and museum database images for the development simple_query. The correspondences are stored in a python dictionary where the key is the number in the name of the query image and the value is the number in the name of the dictionary image. This is necessary to evaluate your development results using the provided mapk() function.

  The correspondences (week 3 - development): [(0, 76), (1, 105), (2, 34), (3, 83), (4, 109), (5, 101), (6, 57), (7, 27), (8, 50), (9, 84), (10, 25), (11, 60), (12, 45), (13, 99), (14, 107), (15, 44), (16, 65), (17, 63), (18, 111), (19, 92), (20, 67), (21, 22), (22, 87), (23, 85), (24, 13), (25, 39), (26, 103), (27, 6), (28, 62), (29, 41)]

  For instance, for key '0', the value is '76'. This means that, for query image '000000.jpg' the corresponding true correspondence is image '000076.jpg' in the museum database.

  The correspondences (week 3 - test): [(0, 30), (1, 102), (2, 100), (3, 94), (4, 56), (5, 10), (6, 101), (7, 0), (8, 107), (9, 82), (10, 108), (11, 106), (12, 12), (13, 78), (14, 63), (15, 97), (16, 34), (17, 47), (18, 21), (19, 15), (20, 68), (21, 46), (22, 26), (23, 32), (24, 75), (25, 19), (26, 57), (27, 98), (28, 93), (29, 20)]

  The correspondences (week4 - development): [[0, [-1]], [1, [-1]], [2, [115]], [3, [-1]], [4, [-1]], [5, [99]], [6, [-1]], [7, [89]], [8, [19]], [9, [85]], [10, [90]], [11, [121, 117]], [12, [-1]], [13, [-1]], [14, [130]], [15, [6, 84]], [16, [35, 48, 52]], [17, [118]], [18, [-1]], [19, [-1]], [20, [-1]], [21, [-1]], [22, [60]], [23, [119, 128]], [24, [-1]], [25, [47]], [26, [-1]], [27, [41]], [28, [-1]], [29, [126, 123]]]

  Note that in python you can format a number with trailing zeros by using:

  Python 3.6.2 (default, Mar 13 2018, 08:54:27)

  &gt;&gt;&gt; id = 76

  &gt;&gt;&gt; str = '{:06d}.jpg'.format(id)

  &gt;&gt;&gt; print (str)

  000076.jpg
