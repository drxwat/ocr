import cv2
import random
import string
import os


def save_image_batch(images, directory, file_type='jpg'):
    if not os.path.exists(directory):
        os.makedirs(directory)

    for i in range(len(images)):
        cv2.imwrite('{}{}.{}'.format(directory, random_string(), file_type), images[i])


def random_string(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))


def resize_image(image, new_height=None, new_width=None):
    """Resized image with or without saving ratio."""

    if new_width is None and new_height is None:
        raise AttributeError('at least one of new_height or new_width required')

    if new_width is None:
        resize_scale = image.shape[0] / new_height
        new_width = int(image.shape[1] / resize_scale)
    elif new_height is None:
        resize_scale = image.shape[1] / new_width
        new_height = int(image.shape[0] / resize_scale)

    return cv2.resize(image, (new_width, new_height))
