import cv2
import os
from tqdm import tqdm

new_height = 500
files_dir = '../samples/4/'

files = os.listdir(files_dir)

for file_name in tqdm(files):
    image_path = '{}{}'.format(files_dir, file_name)
    image = cv2.imread(image_path)

    resize_scale = image.shape[0] / new_height
    new_width = int(image.shape[1] / resize_scale)

    resized_image = cv2.resize(image, (new_width, new_height))
    cv2.imwrite(image_path, resized_image)
