import numpy as np
import os
import argparse
from PIL import Image

# todo: Move to arguments
output_file_name = "pixel_data.csv"

ap = argparse.ArgumentParser(description='Processes images from listed directories to csv file. '
                                         'Each directory to separate class.')
ap.add_argument('class_dir', type=str, nargs='+', metavar='CLASS_DIR', help='Path to the directory with class images')
arguments = vars(ap.parse_args())
directories = arguments['class_dir']

with open(output_file_name, mode='wb') as out_file:
    class_n = 0
    for directory in directories:
        images = os.listdir(directory)

        for image_name in images:
            image_path = '{}{}'.format(directory, image_name)

            # reading image
            image = Image.open(image_path)
            image_pixels = np.asarray(image)

            # reshaping as 1d vector
            new_shape = (1, image_pixels.shape[0] * image_pixels.shape[1] * image_pixels.shape[2])
            image_vec = image_pixels.reshape(new_shape)

            # concatenating class label
            image_class = np.array([class_n], dtype=image_vec.dtype).reshape((1, 1))
            data_sample = np.hstack((image_class, image_vec))

            # saving prepared data sample to csv file
            np.savetxt(out_file, data_sample, delimiter=",", fmt='%d')

        class_n += 1
