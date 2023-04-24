import sys
import os
import cv2

"""
Call is 
resize_images.py <dir> <width> <height>

This routine resizes the images in the source dir to be
"on the similar scale" to the specified dimensions and
then center crops the images as necessary to proeduce the
desired dimensions

This script creates a new directory called <dir>_cropped
For each image in <dir> this script center crops it to the
specified dimensions. 

Only supports .jpg or .png images
"""


def resize_images(src_dir, dest_dir, width, height):
    if not os.path.exists(src_dir):
        print(f"ERROR: {src_dir} does not exist")
        sys.exit(2)
    if not os.path.isdir(src_dir):
        print(f"ERROR: {src_dir} is not a directory")
        sys.exit(3)

    if os.path.exists(dest_dir):
        print(f"ERROR: {dest_dir} already exists")
        sys.exit(4)

    os.mkdir(dest_dir)

    count = 0
    for fn in os.listdir(src_dir):
        if (count + 1) % 10 == 0:
            print(count + 1)
        count += 1

        fp = os.path.join(src_dir, fn)
        if not os.path.isfile(fp):
            continue

        ext = os.path.splitext(fn)[-1]
        if ext not in [".jpg", ".png"]:
            continue

        img = cv2.imread(fp)
        h, w, c = img.shape

        if h < height:
            print(f"ERROR: Height {h} to small to crop to {height} for {fn}")
            continue
        if w < width:
            print(f"ERROR: Widtrh {w} to small to crop to {width} for {fn}")
            continue

        resized = cv2.resize(img, dsize=(width, height), interpolation=cv2.INTER_AREA)
        new_fp = os.path.join(dest_dir, fn)
        cv2.imwrite(new_fp, resized)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Call is sys.argv[0] <src_dir> <dest_dir> <width> <height>")
        sys.exit(1)

    src_dir = sys.argv[1]
    dest_dir = sys.argv[2]
    width = int(sys.argv[2])
    height = int(sys.argv[3])

    resize_images(src_dir, dest_dir, width, height)
