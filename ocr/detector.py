import cv2


def sliding_window(image, step_size=8, window_size=(15, 15)):
    """ Sliding window image generator.  """
    for y in range(0, image.shape[0], step_size):
        for x in range(0, image.shape[1], step_size):
            # windowed_image = image[x: x + window_size[1], y: y + window_size[1]]
            # windowed_image = image[y:window_size[1], x:window_size[0]]
            windowed_image = image[y: y + window_size[1], x: x + window_size[0]]

            if windowed_image.shape[0] < window_size[0] or windowed_image.shape[1] < window_size[1]:
                break

            yield (x, y, windowed_image)


def image_pyramid(image, scale=0.8, min_size=(30, 30)):
    """ Pyramid image generator. """
    yield image

    while True:
        # image = rescale(image, scale)
        # image = cv2.pyrDown(image)

        scaled_shape = (int(image.shape[1] * scale), int(image.shape[0] * scale))
        if scaled_shape[1] < min_size[0] or scaled_shape[0] < min_size[1]:
            break

        image = cv2.resize(image, scaled_shape)
        yield image
