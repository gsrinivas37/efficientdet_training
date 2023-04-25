import os
from resize_images import resize_images

labels = ["Bees-Wasps",
          "Bumble Bee",
          "Butterflies-Moths",
          "Fly",
          "Hummingbird",
          "Inflorescence",
          "Other"]

dir_name = [
    'Bees_Wasp',
    'Bumble_Bee',
    'Butterflies_Moths',
    'Fly',
    'Hummingbird',
    'Other',
    'Other'
]


def generate_eficientnet_data(input_dir, dest_dir, img_size):
    img_dir = os.path.join(input_dir, 'images')
    label_dir = os.path.join(input_dir, 'labels')
    if not os.path.exists(dest_dir):
        resize_images(os.path.join(input_dir, 'images'), dest_dir, img_size, img_size)


    for img_file in os.listdir(img_dir):
        label_file_path = os.path.join(label_dir, img_file[:-4] + ".txt")
        if os.path.exists(label_file_path):
            label_set = set()
            with open(label_file_path) as label_file:
                for line in label_file:
                    fields = line.split()
                    label = int(fields[0])
                    if float(fields[3]) < 0.01 or float(fields[4]) < 0.01:
                        print(f'Skipping..{img_file} {line}')
                        continue
                    if label != 5:
                        label_set.add(label)
            out_dir = dir_name[5]
            if len(label_set) == 1:
                out_dir = dir_name[list(label_set)[0]]
            elif len(label_set) > 1:
                print("Something wrong.. More than two labels..")

            full_dir_path = os.path.join(dest_dir, out_dir)
            if not os.path.exists(full_dir_path):
                os.mkdir(full_dir_path)
            src_file = os.path.join(dest_dir, img_file)
            dst_file = os.path.join(dest_dir, out_dir, img_file)
            os.rename(src_file, dst_file)


        else:
            print(f"doesn't exist: {label_file_path}")

        # break


SRC_TRAIN_DIR = '/Users/gsrinivas37/Downloads/Pollinators-18_YOLO_640x640/train'
SRC_TEST_DIR = '/Users/gsrinivas37/Downloads/Pollinators-18_YOLO_640x640/test'
IMG_SIZE = 600
generate_eficientnet_data(SRC_TEST_DIR, 'train_data', IMG_SIZE)
generate_eficientnet_data(SRC_TEST_DIR, 'test_data', IMG_SIZE)
