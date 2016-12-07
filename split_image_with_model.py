import argparse
import cv2
from ocr.detector import image_pyramid, sliding_window_batch
from ocr.helper import save_image_batch
from keras.models import load_model
from datetime import datetime
import ntpath
import numpy as np

pyramid_scale, pyramid_min_width, pyramid_min_height = 0.8, 150, 150
sl_w_step, sl_w_width, sl_w_height = 5, 30, 30

ap = argparse.ArgumentParser(description='Detecting text blocks with sliding window algorithm.')
ap.add_argument('image', help='Path to the image')
ap.add_argument('model', type=str, help='Path to Keras model')
ap.add_argument('output', type=str, help='Output directory (default: .)')
# Pyramid arguments
ap.add_argument('-ps', type=float, default=pyramid_scale, metavar='SCALE',
                help='Pyramid scale rate (default: {})'.format(pyramid_scale))
ap.add_argument('-pw', type=int, default=pyramid_min_width, metavar='P_WIDTH',
                help='Minimum width of scaled image (default: {})'.format(pyramid_min_width))
ap.add_argument('-ph', type=int, default=pyramid_min_height, metavar='P_HEIGHT',
                help='Minimum height of scaled image (default: {})'.format(pyramid_min_height))
# Sliding window arguments
ap.add_argument('-s', type=int, default=sl_w_step, metavar='STEP',
                help='Sliding window step size in pixels (default: {})'.format(sl_w_step))
ap.add_argument('-sw', type=int, default=sl_w_width, metavar='SW_WIDTH',
                help='Sliding window width (default: {})'.format(sl_w_width))
ap.add_argument('-sh', type=int, default=sl_w_height, metavar='SW_HEIGHT',
                help='Sliding window height (default: {})'.format(sl_w_height))

args = ap.parse_args()
# Ugly assigning
image_path, model_path, directory_to_write, pyramid_scale, pyramid_min_width, pyramid_min_height, sl_w_step, sl_w_width, sl_w_height \
    = args.image, args.model, args.output, args.ps, args.pw, args.ph, args.s, args.sw, args.sh

# Preparing image and model
model = load_model(model_path)
print('Model {} loaded.'.format(model_path))

image = cv2.imread(image_path)

# Processing image
output_directory = '{}{}/'.format(directory_to_write, ntpath.basename(image_path))
pyramid_number = 0
batch_size = 2048
for pyramid_image in image_pyramid(image, scale=pyramid_scale, min_size=(pyramid_min_width, pyramid_min_height)):

    # Getting batches of images from image window algorithm
    for window_batch in sliding_window_batch(pyramid_image, sl_w_step, (sl_w_width, sl_w_height), batch_size):
        # Getting prediction from keras model
        predictions = model.predict(window_batch.reshape(batch_size, 3, 30, 30)/255)

        # Computing if only we fount an image with more than 50%
        if bool(np.any(predictions[:, 0] >= 0.9)):
            # Getting filtered predictions
            predictions_by_percent = {
                '99': (predictions[:, 0] >= 0.99),
                '98': (predictions[:, 0] < 0.99) & (predictions[:, 0] >= 0.98),
                '97': (predictions[:, 0] < 0.98) & (predictions[:, 0] >= 0.97),
                '96': (predictions[:, 0] < 0.97) & (predictions[:, 0] >= 0.96),
                '95': (predictions[:, 0] < 0.96) & (predictions[:, 0] >= 0.95),
                '94': (predictions[:, 0] < 0.95) & (predictions[:, 0] >= 0.94),
                '93': (predictions[:, 0] < 0.94) & (predictions[:, 0] >= 0.93),
                '92': (predictions[:, 0] < 0.93) & (predictions[:, 0] >= 0.92),
                '91': (predictions[:, 0] < 0.92) & (predictions[:, 0] >= 0.91),
                '90': (predictions[:, 0] < 0.91) & (predictions[:, 0] >= 0.90),
            }

            # Saving relevant images by corresponding subdirectories
            for percentage_bound in predictions_by_percent:
                if bool(np.any(predictions_by_percent[percentage_bound])) is True:
                    save_image_batch(window_batch[predictions_by_percent[percentage_bound]],
                                     '{}{}/'.format(output_directory, percentage_bound))

    pyramid_number += 1
