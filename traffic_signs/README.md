# Traffic Sign Detection



The code consists in several parts:
1. Code analysis:
   dataset_analysis.ipynb: computes size, form factor, filling factor and class frequency in the train dataset

2. Train/val split:
    dataset_split.py: constructs train and validation split from train dataset by making sure both datasets are
    balanced in terms of:
        - numbers of elements of each class
        - lightness of traffic signs
    Running the script takes dataset from 'data/train' and outputs results to 'data/train_val', divided in two folders:
        - 'train'
        - 'val'
    both including images '*.jpg', masks and gt annotations

3. Color segmentation using different methods:
    Four methods included in candidate_generation_pixel.py as:
        - candidate_generation_pixel_ihsl1(im): converts to IHSL space and uses thresholds from reference_colors.py
        - candidate_generation_pixel_ihsl2(im): converts to IHSL space and uses value ranges from color_ranges.py
        - candidate_generation_pixel_rgb(im): takes image in RGB and uses value ranges from color_ranges.py
        - candidate_generation_pixel_hsv_euclidean(rgb): converts to HSV and takes reference colors from
        reference_colors.py

        **reference_colors.py: run it to obtain average dominant colors in HSV space, from dataset in 'data/train'
        **color_ranges.py: run it to obtain histograms and ranges [mean - std, mean + std] for the most dominant colors
        in RGB space and also for hue in HSV

4. Evaluation of methods:
    Each method can be evaluated by running traffic_sign_detection.py as:

        pixel_precision, pixel_accuracy, pixel_recall, pixel_specificity, pixel_sensitivity, pixel_F1, pixel_TP,
        pixel_FP, pixel_FN = traffic_sign_detection(images_dir, output_dir, pixel_method, window_method);

     where:
       ->'images_dir' = path to the validation dataset ('data/train_val/val')
       ->'output_dir' = path where masks are saved
       ->'pixel_method' = color segmentation method we are using (method1: 'ihsl_1', method2: 'ihsl_2', method3: 'rgb',
       method4: 'hsv_euclidean')
       ->'window_method' = 'None' (it's not developed yet)

5. Masks generation:
    generate_masks.py is used for generating the masks for all images in test.
    Run it by using:

        generate_masks(args.images_dir, args.output_dir, args.pixel_method)

     where:
        -args.images_dir = Path to test dataset ('data/test')
        -args.output_dir = Path where masks are saved
        -args.pixel_method = color segmentation method we are using (method1: 'ihsl_1', method2: 'ihsl_2', method3: 'rgb',
       method4: 'hsv_euclidean')
