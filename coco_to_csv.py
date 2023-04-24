"""
Given a the path to a directory containing a COCO formated dataset (i.e. containing images and 
an _annotations.coco.json file), this routine creates a new directory with the suffix _bb 
containing images with bounding boxes drawn in.
"""

import os
import sys
import shutil
import json
from copy import copy, deepcopy
import cv2
import supervision as sv
import torchvision
import random
import numpy as np


def coco_to_csv(src_dir, new_w, new_h, output_csv):
    if not os.path.exists(src_dir):
        print(f"ERROR: {src_dir} does not exist")
        sys.exit(1)
    if not os.path.isdir(src_dir):
        print(f"ERROR: Specified path {src_dir} is not a directory")
        sys.exit(2)

    # Verify src_dir has _annotations.coco.json
    annots_path = os.path.join(src_dir, "_annotations.coco.json")
    if not os.path.exists(annots_path):
        print(f"ERROR: No annotations file f{annots_path}")
        sys.exit(3)

    annots_path = os.path.join(src_dir, "_annotations.coco.json")
    with open(annots_path, "r") as fd:
        annots = json.load(fd)
    if annots is None:
        print(f"ERROR: Reading {annots_path}")

    dataset = torchvision.datasets.CocoDetection(src_dir, annots_path)

    categories = dataset.coco.cats
    id2label = {k: v['name'] for k, v in categories.items()}

    img_ids = dataset.coco.getImgIds()
    output = "filename,width,height,class,xmin,ymin,xmax,ymax\n"
    for img_id in img_ids:
        # Read in image and obtain its annotations
        img_info = dataset.coco.loadImgs(img_id)[0]
        img_annots = dataset.coco.imgToAnns[img_id]
        image_path = os.path.join(dataset.root, img_info['file_name'])
        img = cv2.imread(image_path)

        if len(img_annots) == 0:
            continue
        try:
            detections = sv.Detections.from_coco_annotations(coco_annotation=img_annots)
            h, w, _ = img.shape
            h_factor = new_h / h
            w_factor = new_w / w
            for box, _, _, class_id, _ in detections:
                xmin = int(box[0] * w_factor)
                ymin = int(box[1] * h_factor)

                xmax = int(box[2] * w_factor)
                ymax = int(box[3] * h_factor)

                if (xmax - xmin) < new_w/100 or (ymax-ymin) < new_h/100:
                    print('skipping...')
                    continue

                output += (f"{img_info['file_name']},{new_w},{new_h},{id2label[class_id]},{int(box[0] * w_factor)},"
                      f"{int(box[1] * h_factor)},{int(box[2] * w_factor)},{int(box[3] * h_factor)}\n")
        except Exception as exc:
            import traceback
            traceback.print_tb(exc.__traceback__, limit=1, file=sys.stdout)
            exc_type, exc_value, exc_tb = sys.exc_info()
            tb = traceback.TracebackException(exc_type, exc_value, exc_tb)
            print(''.join(tb.format_exception_only()))

    with open(output_csv, "w") as fd:
        fd.write(output)