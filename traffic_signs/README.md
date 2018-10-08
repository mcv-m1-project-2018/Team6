# Traffic Sign Detection

#### 1. Dataset analysis:
`dataset_analysis.ipynb`: notebook that explores the training set. It computes class frequency, object size, object form factor, object filling factor, and dominant colors.


#### 2. Train/val split:
`dataset_split.py`: computes the training and validation splits while making sure both datasets are balanced in terms of:
- numbers of elements of each class
- lightness of traffic signs
 
To run it:
```bash
python3 dataset_split.py [--data_path DATA_PATH] [--train_pct TRAIN_PCT] [--output_path OUTPUT_PATH]
```
The above command takes images, masks and gts from `DATA_PATH` and leaves the resulting split in `OUTPUT_PATH`, using `TRAIN_PCT` of the images for training and `1 - TRAIN_PCT` of the images for validation.


#### 3. Color segmentation using different methods:
Four methods included in candidate_generation_pixel.py as:
- `candidate_generation_pixel_ihsl1(im)`: converts to IHSL space and uses thresholds from `reference_colors.py`
- `candidate_generation_pixel_ihsl2(im)`: converts to IHSL space and uses value ranges from `color_ranges.py`
- `candidate_generation_pixel_rgb(im)`: takes image in RGB and uses value ranges from `color_ranges.py`
- `candidate_generation_pixel_hsv_euclidean(rgb)`: converts to HSV and takes reference colors from
`reference_colors.py`

**`color_utils.py`: contains helper functions to compute dominant colors in HSV space, and to obtain histograms and ranges `[mean - std, mean + std]` for the most dominant colors in RGB space and also for hue in HSV space.


#### 4. Evaluation of methods:
Each method can be evaluated by running `traffic_sign_detection.py` as:

```bash
python3 traffic_sign_detection.py <dirName> <outPath> <pixelMethod> [--windowMethod=<wm>]
```
where:
- `images_dir`: path to the validation dataset ('data/train_val/val')
- `output_dir`: path where masks are saved
- `pixel_method`: color segmentation method we are using (method1: 'ihsl_1', method2: 'ihsl_2', method3: 'rgb',
 method4: 'hsv_euclidean')
- `window_method`: 'None' (it's not developed yet)

#### 5. Masks generation:
`generate_masks.py` is used for generating the masks for all images in the test set.

Run it by using:
```bash
python3 generate_masks.py <images_dir> <output_dir> <pixel_method>
```
where:
- `images_dir`: path to test dataset ('data/test')
- `args.output_dir`: path where masks are saved
- `args.pixel_method`: color segmentation method we are using (method1: 'ihsl_1', method2: 'ihsl_2', method3: 'rgb',
 method4: 'hsv_euclidean')
