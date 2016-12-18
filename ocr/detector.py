import cv2
import numpy as np


def sliding_window(image, step_size=8, window_size=(15, 15)):
    """ Sliding window image generator.  """
    for y in range(0, image.shape[0], step_size):
        for x in range(0, image.shape[1], step_size):
            # windowed_image = image[x: x + window_size[1], y: y + window_size[1]]
            # windowed_image = image[y:window_size[1], x:window_size[0]]
            windowed_image = image[y: y + window_size[0], x: x + window_size[1]]

            if windowed_image.shape[0] < window_size[0] or windowed_image.shape[1] < window_size[1]:
                break

            yield (x, y, windowed_image)


def sliding_window_batch(image, step_size=8, window_size=(15, 15), batch_size=128):
    """ Sliding window thar returns all windows as numpy array"""
    w_win_num = np.floor(((image.shape[1] - window_size[1]) / step_size) - 1)
    h_win_num = np.floor(((image.shape[0] - window_size[0]) / step_size) - 1)

    win_num = w_win_num.astype(int) * h_win_num.astype(int)
    last_batch_num = win_num % batch_size

    windows = np.zeros((batch_size, window_size[0], window_size[1], image.shape[2]))

    i = 0
    for x, y, window_image in sliding_window(image, step_size=step_size, window_size=window_size):

        if i == batch_size:
            yield windows
            i = 0
            windows = np.zeros((batch_size, window_image.shape[0], window_image.shape[1], window_image.shape[2]))

        windows[i] = window_image
        i += 1

    yield windows[:last_batch_num, :, :, :]


def image_pyramid(image, scale=0.8, min_size=(30, 30)):
    """ Pyramid image generator. """
    yield image

    while True:
        # image = rescale(image, scale)
        # image = cv2.pyrDown(image)

        scaled_shape = (int(image.shape[1] * scale), int(image.shape[0] * scale))
        if scaled_shape[1] < min_size[1] or scaled_shape[0] < min_size[0]:
            break

        image = cv2.resize(image, scaled_shape)
        yield image


def is_object(candidate, model=None, shape=None, boundary=0.5):
    """Evaluates classification check"""
    if model is not None:
        prediction = model.predict(candidate.reshape(shape) if shape is not None else candidate)
        if prediction[0][0] >= boundary:
            return True
    return False
