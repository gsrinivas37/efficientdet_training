import numpy as np
import pandas as pd
import os
import cv2
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.utils import shuffle


def get_data(input_dir, img_size):
    dataset_path = os.listdir(input_dir)

    class_labels = []
    for item in dataset_path:
        # Get all the file names
        all_classes = os.listdir(input_dir + '/' + item)

        # Add them to the list
        for room in all_classes:
            class_labels.append((item, str('dataset_path' + '/' + item) + '/' + room))

    # Build a dataframe
    df = pd.DataFrame(data=class_labels, columns=['Labels', 'image'])
    label_count = df['Labels'].value_counts()
    print(label_count)

    images = []
    labels = []

    for i in dataset_path:
        data_path = os.path.join(input_dir, str(i))
        filenames = [i for i in os.listdir(data_path)]

        for f in filenames:
            img = cv2.imread(data_path + '/' + f)
            img = cv2.resize(img, (img_size, img_size))
            images.append(img)
            labels.append(i)

    images = np.array(images)
    images = images.astype('float32') / 255.0

    y = df['Labels'].values
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    y = y.reshape(-1, 1)
    onehot_encoder = OneHotEncoder(sparse=False)
    Y = onehot_encoder.fit_transform(y)

    X, Y = shuffle(images, Y, random_state=1)
    return X, Y

# train_x, train_y = get_data('train_data', 224)
# test_x, test_y = get_data('test_data', 224)
# print(train_x.shape)
# print(train_y.shape)
# print(test_x.shape)
# print(test_y.shape)
