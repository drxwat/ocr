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
