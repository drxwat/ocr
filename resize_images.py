import cv2
import os
import argparse
from tqdm import tqdm
from ocr.helper import resize_image

ap = argparse.ArgumentParser(description='Changes images size with or without saving image ratio.')

ap.add_argument('files_dir', type=str, metavar='FILES_DIR', help='Path to directory with input files')
ap.add_argument('--height', type=int, metavar='HEIGHT', help='Height of new image')
ap.add_argument('--width', type=int, metavar='WIDTH', help='Width of new image')

args = ap.parse_args()

if args.height is None and args.width is None:
    ap.error('at least one of --height or --width required')

files_dir, new_height, new_width = args.files_dir, args.height, args.width

files = os.listdir(files_dir)

for file_name in tqdm(files):
    image_path = '{}{}'.format(files_dir, file_name)
    image = cv2.imread(image_path)

    cv2.imwrite(image_path, resize_image(image, new_height, new_width))
