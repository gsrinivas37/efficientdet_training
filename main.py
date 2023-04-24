import os.path

from resize_images import resize_images
from coco_to_csv import coco_to_csv
from check_bboxes import *
from generate_tfrecord import generate_tf_records


def generate_tf_records_from_coco(SRC_DIR, IMG_SIZE=768):
    if not os.path.exists('data'):
        os.mkdir('data')

    resize_images(os.path.join(SRC_DIR, 'train'), 'data/train', IMG_SIZE, IMG_SIZE)
    resize_images(os.path.join(SRC_DIR, 'test'), 'data/test', IMG_SIZE, IMG_SIZE)

    coco_to_csv(os.path.join(SRC_DIR, 'test'), IMG_SIZE, IMG_SIZE, 'data/test.csv')
    coco_to_csv(os.path.join(SRC_DIR, 'train'), IMG_SIZE, IMG_SIZE, 'data/train.csv')

    check_bboxes('data/train.csv', 'data/train', 'tmp_bbox', max_cnt=200)
    get_bbox_aspect_ratio('data/train.csv')

    generate_tf_records('data/train.csv', 'data/train.tfrecord', 'data/train')
    generate_tf_records('data/test.csv', 'data/test.tfrecord', 'data/test')


SRC_DIR = '/Users/gsrinivas37/Downloads/Pollinators-17_COCO_fullsize_aug_null_nf'
IMG_SIZE = 768
generate_tf_records_from_coco(SRC_DIR, IMG_SIZE)
