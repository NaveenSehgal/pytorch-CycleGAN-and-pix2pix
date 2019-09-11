import pickle
import cv2
import argparse
import numpy as np
import os
import shutil

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']


def get_mrcnn_model():
    import sys
    import random
    import math
    import skimage.io
    import matplotlib
    import matplotlib.pyplot as plt

    # Root directory of the project
    ROOT_DIR = os.path.abspath("../")

    # Import Mask RCNN
    sys.path.append(ROOT_DIR)  # To find local version of the library
    from mrcnn import utils
    import mrcnn.model as modellib
    # Import COCO config
    sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
    import coco

    # Directory to save logs and trained model
    MODEL_DIR = os.path.join(ROOT_DIR, "logs")

    # Local path to trained weights file
    COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
    # Download COCO trained weights from Releases if needed
    if not os.path.exists(COCO_MODEL_PATH):
        utils.download_trained_weights(COCO_MODEL_PATH)

    class InferenceConfig(coco.CocoConfig):
        # Set batch size to 1 since we'll be running inference on
        # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1

    config = InferenceConfig()
    config.display()

    # Create model object in inference mode.
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

    # Load weights trained on MS-COCO
    model.load_weights(COCO_MODEL_PATH, by_name=True)
    return model

    # Run detection
    # results = model.detect([image], verbose=0)


def resize_mask(mask, scale_factor=1.0):
    mask = np.array(mask).astype(np.uint8)  # Make compatible with cv2
    input_width, input_height = mask.shape

    # Resize mask image
    scaled_width, scaled_height = np.array(mask.shape) * scale_factor
    resized = cv2.resize(mask, (int(scaled_width), int(scaled_height)))

    # Crop center input_width x input_height of resized mask image
    middle = (scaled_width // 2, scaled_height // 2)
    dx, dy = input_width // 2, input_height // 2
    left, right = int(middle[0] - dx), int(middle[0] + dx)
    bottom, top = int(middle[0] - dy), int(middle[1] + dy)
    resized = resized[left:right, bottom:top]

    # Return union of resized mask and original mask
    output = np.logical_or(resized, mask).astype(np.uint8)
    return output


def segment_image(image, mask):
    ''' Multiply image by mask to segment foreground '''
    mask = np.reshape(mask, (mask.shape[0], mask.shape[1]))
    mask = mask.astype(np.uint8)
    img = np.array([image[:, :, x] * mask for x in range(3)])
    img = np.rollaxis(img, 0, 3)
    return img


def opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--inputdir', help="The input parent directory", required=True)
    parser.add_argument('-o', "--outputdir", help="The output directory", required=True)
    parser.add_argument('-s', '--scalefactor', default=1.0, help="Scale factor for human masks")
    args = parser.parse_args()

    print('Input Directory: {}\nOutput Directory: {}\nScale Factor: {}'.format(args.inputdir, args.outputdir, args.scalefactor))

    return args

def get_image_paths(input_dir, output_dir):
    ''' Returns dictionary matching image input paths to their relative output paths '''
        # Find all images in input directory
    image_paths = {}

    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".png") or file.endswith(".jpg"):
                img_input_path = os.path.join(root, file)
                img_output_path = output_dir + root.replace(
                    input_dir, '') + "/{}".format(file)

                image_paths[img_input_path] = img_output_path

    return image_paths

def configure_output_dirs(input_dir, output_dir):
    ''' We want to copy input dir structure to output dir '''
    def ig_f(dir, files):
        return [f for f in files if os.path.isfile(os.path.join(dir, f))]

    # Don't let this function change an existing folder
    if os.path.exists(output_dir):
        print("ERROR: Output directory already exists!")
        exit(1)

    shutil.copytree(input_dir, output_dir, ignore=ig_f)


def filter_results_for_people(r):
    person_id = class_names.index('person')
    people = r['class_ids'] == person_id
    r['rois'] = r['rois'][people]
    r['masks'] = r['masks'][:, :, people]
    r['class_ids'] = r['class_ids'][people]
    r['scores'] = r['scores'][people]

    # If multiple people, get the union of the masks
    masks_channels = r['masks'].shape[2]
    if masks_channels > 1:
        masks = r['masks'][:, :, 0]
        for channel in range(1, masks_channels):
            masks = np.logical_or(masks, r['masks'][:, :, channel])
        r['masks'] = masks

    return r
