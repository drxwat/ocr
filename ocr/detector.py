from skimage.transform import rescale
import cv2


def sliding_window(image, step_size=8, window_size=(15, 15)):
    """ Sliding window image generator.  """
    for y in range(0, image.shape[0], step_size):
        for x in range(0, image.shape[1], step_size):
            windowed_image = image[x: x + window_size[1], y: y + window_size[1]]

            if windowed_image.shape[0] < window_size[0] or windowed_image.shape[1] < window_size[1]:
                break

            yield (x, y, windowed_image)


def image_pyramid(image, min_size=(30, 30)):
    """ Pyramid image generator. """
    yield image

    while True:
        # image = rescale(image, scale)
        image = cv2.pyrDown(image)

        if image.shape[0] < min_size[0] or image.shape[1] < min_size[1]:
            break

        yield image
