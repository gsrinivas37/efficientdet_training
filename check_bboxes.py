import csv
import os

import numpy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image


def get_bbox_aspect_ratio(csv_file):
    aspect_ratios = dict()
    all = []
    with open(csv_file, mode='r') as file:
        csvFile = csv.reader(file)
        for lines in csvFile:
            if 'filename' in lines:
                continue
            img_path, _, _, cls, xmin, ymin, xmax, ymax = lines
            if cls not in aspect_ratios:
                aspect_ratios[cls] = []
            asp_ratio = (float(xmax) - float(xmin)) / (float(ymax) - float(ymin))
            all.append(asp_ratio)
            aspect_ratios[cls].append(asp_ratio)
    plt.clf()
    plt.hist(numpy.array(all), range=[0, 3])
    plt.title('All')
    if not os.path.exists('hist'):
        os.mkdir('hist')
    plt.savefig('hist/all.png')
    for keys in aspect_ratios:
        plt.clf()
        plt.hist(numpy.array(aspect_ratios[keys]), range=[0, 3])
        plt.title(keys)
        plt.savefig(f'hist/{keys}.png')


def check_bboxes(csv_file, img_dir, out_dir, max_cnt=50):
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    # opening the CSV file
    with open(csv_file, mode='r') as file:
        csvFile = csv.reader(file)
        # displaying the contents of the CSV file
        i = 0
        prev_image = None
        boxes = []
        for lines in csvFile:
            if 'filename' in lines:
                continue
            img_path, _, _, cls, xmin, ymin, xmax, ymax = lines
            if prev_image is not None and img_path == prev_image:
                boxes.append((int(xmin), int(ymin), int(xmax), int(ymax), cls))
            else:
                if len(boxes) > 0:
                    i = i + 1
                    save_image_with_bbox(os.path.join(img_dir, prev_image), boxes, out_dir)
                boxes = []
                try:
                    boxes.append((int(xmin), int(ymin), int(xmax), int(ymax), cls))
                except ValueError:
                    print(lines)

            print(lines)
            prev_image = img_path
            if i > max_cnt:
                break


def save_image_with_bbox(image_path, boxes, out_dir):
    x = np.array(Image.open(image_path), dtype=np.uint8)

    # Create figure and axes
    fig, ax = plt.subplots(1)

    # Display the image
    ax.imshow(x)

    # Create a Rectangle patch
    for xmin, ymin, xmax, ymax, cls in boxes:
        rect = patches.Rectangle((xmin, ymin), (xmax - xmin), (ymax - ymin), linewidth=1,
                                 edgecolor='r', facecolor="none")
        # Add the patch to the Axes
        ax.add_patch(rect)
    path = os.path.join(out_dir, os.path.split(image_path)[-1])
    plt.savefig(path)
