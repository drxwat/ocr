import argparse
import cv2
from ocr.detector import image_pyramid, sliding_window_batch
from ocr.helper import save_image_batch
from keras.models import load_model
import ntpath
import numpy as np

pyramid_scale, pyramid_min_width, pyramid_min_height = 0.8, 150, 150
sl_w_step, sl_w_width, sl_w_height = 5, 30, 30

ap = argparse.ArgumentParser(description='Detecting text blocks with sliding window algorithm.')
ap.add_argument('image', help='Path to the image')
ap.add_argument('model', type=str, help='Path to Keras model')
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
image_path, model_path, pyramid_scale, pyramid_min_width, pyramid_min_height, sl_w_step, sl_w_width, sl_w_height \
    = args.image, args.model, args.ps, args.pw, args.ph, args.s, args.sw, args.sh

# Preparing image and model
model = load_model(model_path)
print('Model {} loaded.'.format(model_path))

image = cv2.imread(image_path)

# Processing image
directory_to_write = '../results/'
output_directory = '{}{}/'.format(directory_to_write, ntpath.basename(image_path))
pyramid_number = 0
batch_size = 128
for pyramid_image in image_pyramid(image, scale=pyramid_scale, min_size=(pyramid_min_width, pyramid_min_height)):

    # Getting batches of images from image window algorithm
    for window_batch in sliding_window_batch(pyramid_image, sl_w_step, (sl_w_width, sl_w_height), batch_size):
        # Getting prediction from keras model
        predictions = model.predict(window_batch.reshape(batch_size, 3, 30, 30)/255)

        # Getting filtered predictions
        predictions_by_percent = {
            '90': (predictions[:, 0] >= 0.9),
            '80': (predictions[:, 0] < 0.9) & (predictions[:, 0] >= 0.8),
            '70': (predictions[:, 0] < 0.8) & (predictions[:, 0] >= 0.7),
            '60': (predictions[:, 0] < 0.7) & (predictions[:, 0] >= 0.6),
            '50': (predictions[:, 0] < 0.6) & (predictions[:, 0] >= 0.5)
        }

        # Saving relevant images
        map(lambda predict_percent, predict_name:
            save_image_batch(window_batch[predict_percent], '{}{}'.format(output_directory, predict_name))
            if bool(np.any(predict_percent))
            else None,
            list(predictions_by_percent.values()), predictions_by_percent.keys())

    pyramid_number += 1

    # symbols_image = Image.new('L', (pyramid_image.shape[1], pyramid_image.shape[0]))
    # pixels = symbols_image.load()  # create the pixel map

    # for x, y, win_image in sliding_window(pyramid_image, step_size=sl_w_step, window_size=(sl_w_width, sl_w_height)):
    #     input_image = np.expand_dims(win_image.reshape(3, 30, 30), axis=0)
    #     prediction = model.predict(input_image / 255)
    #
    #     if prediction[0][0] > 0.5:
    #         input_image = np.squeeze(input_image, axis=(0,)).reshape((30, 30, 3))
    #         cv2.imwrite('{}{}_{}_{}_{}'.format(directory_to_write, pyramid_number, x, y, ntpath.basename(image_path)),
    #                     input_image)

    # Draw black/white image for current pyramid
    # pyramid_number += 1
