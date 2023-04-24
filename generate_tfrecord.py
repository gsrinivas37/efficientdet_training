# based on https://github.com/datitran/raccoon_dataset/blob/master/generate_tfrecord.py

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import hashlib
import os
import io

import PIL
import pandas as pd

from tensorflow.python.framework.versions import VERSION

if VERSION >= "2.0.0a0":
    import tensorflow.compat.v1 as tf
else:
    import tensorflow as tf

from PIL import Image
from object_detection.utils import dataset_util
from collections import namedtuple, OrderedDict

''' 
*************************************************************************
Make sure to edit this method to match the labels you made with labelImg!
*************************************************************************
'''

labels = ["Bees-Wasps",
          "Bumble Bee",
          "Butterflies-Moths",
          "Fly",
          "Hummingbird",
          "Inflorescence",
          "Other"]


def class_text_to_int(row_label):
    for idx in range(len(labels)):
        if row_label == labels[idx]:
            return idx + 1
    print("Got unexpected: " + row_label)
    return None


def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tf_example(id, group, path):
    img_raw = open(path, "rb").read()
    key = hashlib.sha256(img_raw).hexdigest()
    filename = os.path.basename(path)
    width, height = PIL.Image.open(path).size

    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(class_text_to_int(row['class']))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        "image/height": tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
        "image/width": tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
        "image/filename": tf.train.Feature(bytes_list=tf.train.BytesList(value=[filename.encode("utf-8")])),
        "image/source_id": tf.train.Feature(bytes_list=tf.train.BytesList(value=[str(id).encode("utf-8")])),
        "image/key/sha256": tf.train.Feature(bytes_list=tf.train.BytesList(value=[key.encode("utf-8")])),
        "image/encoded": tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
        "image/format": tf.train.Feature(bytes_list=tf.train.BytesList(value=["jpg".encode("utf8")])),
        "image/object/bbox/xmin": tf.train.Feature(
            float_list=tf.train.FloatList(value=xmins)),
        "image/object/bbox/xmax": tf.train.Feature(
            float_list=tf.train.FloatList(value=xmaxs)),
        "image/object/bbox/ymin": tf.train.Feature(
            float_list=tf.train.FloatList(value=ymins)),
        "image/object/bbox/ymax": tf.train.Feature(
            float_list=tf.train.FloatList(value=ymaxs)),
        "image/object/class/text": tf.train.Feature(
            bytes_list=tf.train.BytesList(value=["license_plate".encode("utf-8")])),
        "image/object/class/label": tf.train.Feature(int64_list=tf.train.Int64List(value=classes)),
        "image/object/difficult": tf.train.Feature(int64_list=tf.train.Int64List(value=[0])),
        "image/object/truncated": tf.train.Feature(int64_list=tf.train.Int64List(value=[0])),
        "image/object/view": tf.train.Feature(bytes_list=tf.train.BytesList(value=["Unspecified".encode("utf-8")])),
    }))
    return tf_example


def generate_tf_records(csv_input, output_path, image_dir):
    writer = tf.python_io.TFRecordWriter(output_path)
    examples = pd.read_csv(csv_input)
    grouped = split(examples, 'filename')
    i = 0
    for group in grouped:
        path = os.path.join(image_dir, group.filename)
        tf_example = create_tf_example(i, group, path)
        writer.write(tf_example.SerializeToString())
        i = i + 1

    writer.close()
    output_path = os.path.join(os.getcwd(), output_path)
    print('Successfully created the TFRecords: {}'.format(output_path))
