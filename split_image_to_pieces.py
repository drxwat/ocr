import argparse
import matplotlib.pyplot as plt
import cv2
from ocr.detector import image_pyramid, sliding_window

pyramid_scale, pyramid_min_width, pyramid_min_height = 0.8, 30, 30
sl_w_step, sl_w_width, sl_w_height = 8, 15, 15

ap = argparse.ArgumentParser(description='Splits original image and it rescaled copies with sliding window algorithm.')
ap.add_argument('image', help='Path to the image')
ap.add_argument('output', type=str, default='.', help='Output directory (default: .)')
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
image_path, output_directory, pyramid_scale, pyramid_min_width, pyramid_min_height, sl_w_step, sl_w_width, sl_w_height \
    = args.image, args.output, args.ps, args.pw, args.ph, args.s, args.sw, args.sh

image = cv2.imread(image_path)

# Processing image
pyramid_number = 0
for pyramid_image in image_pyramid(image, min_size=(pyramid_min_width, pyramid_min_height)):
    for x, y, win_image in sliding_window(pyramid_image, step_size=sl_w_step, window_size=(sl_w_width, sl_w_height)):
        prefix = '{}_{}_{}'.format(pyramid_number, x, y)
        # plt.imsave('{}{}{}'.format(output_directory, prefix, ntpath.basename(image_path)), win_image)
    pyramid_number += pyramid_number
