'''
Given: parent directory, output directory, scale factor

1. Get all image file paths in parent director
2. Minimc parent directory structure (recursively) in output dir
3. For each image:
    3a. Run M-RCNN
    3b. Select largest human? Or make separate images for each human instance
    3c. Segment out human pixels
    3d. Save in correct outpu directory
'''

import os
import numpy as np
import segmentation_utils
import skimage.io
import cv2
from tqdm import tqdm
import time

def main():
    args = segmentation_utils.opts()
    image_paths = segmentation_utils.get_image_paths(args.inputdir, args.outputdir)
    segmentation_utils.configure_output_dirs(args.inputdir, args.outputdir)
    model = segmentation_utils.get_mrcnn_model()

    for input_file, output_file in tqdm(image_paths.items()):       
        
        # Load image and run through mrcnn
        image = cv2.imread(input_file)
        results = model.detect([image], verbose=0)[0]
    
        # Filter for people results and segment image
        results = segmentation_utils.filter_results_for_people(results)
        
        if len(np.unique(results['masks'])) == 0:
            continue  # skip if no people detected
        
        image_segmented = segmentation_utils.segment_image(image, results['masks'])

        # Write to output file path
        cv2.imwrite(output_file, image_segmented)


if __name__ == '__main__':
    main()
