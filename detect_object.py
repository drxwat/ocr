import cv2
import os
import argparse
import uuid
from keras.models import load_model
from ocr.detector import is_object
from ocr.helper import resize_image


output_dir = '.'
classifier = None
classifier_input_shape = (1, 3, 50, 100)

ap = argparse.ArgumentParser(description='Detects and shows objects using cascade classifier (detector) '
                                         'and optional NN classifier to improve result. Can be used to save '
                                         'positive and negative results. If you press 1-num button then displayed '
                                         'image will be marked as positive else it will be marked as negative.'
                                         'In save mode image will be saved in positives sub directory of output '
                                         'directory.')

ap.add_argument('samples_dir', help='Path to the images directory')
ap.add_argument('сс_model', type=str, help='Cascade classifier detector')
# Save mode optional arguments
ap.add_argument('-c', type=str, metavar='CLASSIFIER', help='Path to keras classifier NN. '
                                                           'If detector also gives wrong objects.')
ap.add_argument('-s', action='store_true')
ap.add_argument('-o', type=str, default=output_dir, metavar='OUTPUT_DIR',
                help='Output directory for save mode (default: {})'.format(output_dir))

args = ap.parse_args()
samples_dir, model_path, nn_classifier_path, save_mode, output_dir = \
    args.samples_dir, args.сс_model, args.c, args.s, args.o

# save mod preparations
uuid = uuid.uuid4()
positives_dir = '{}/{}{}/'.format(output_dir, 'pos', uuid)
negatives_dir = '{}/{}{}/'.format(output_dir, 'neg', uuid)
nf_dir = '{}/{}{}/'.format(output_dir, 'nf', uuid)

if save_mode:
    os.mkdir(positives_dir)
    os.mkdir(negatives_dir)
    os.mkdir(nf_dir)

# Loading detector
detector = cv2.CascadeClassifier(model_path)
files = os.listdir(samples_dir)

# Loading additional checker with NN
if nn_classifier_path is not None:
    classifier = load_model(nn_classifier_path)

# Initializing metrics
true_pos, true_neg, false_pos, false_neg = 0, 0, 0, 0
not_detected = 0

for file_name in files:
    image_path = '{}{}'.format(samples_dir, file_name)
    image = cv2.imread(image_path)
    # getting classifier mention =)
    rectangles = detector.detectMultiScale(image, minSize=(10, 10))

    correct_objects_detected_num = 0
    if len(rectangles) > 0:
        # displaying each candidate and saving if in save mode
        for rect in rectangles:
            is_positive = False
            x, y = rect[0:2]
            width, height = rect[2:4]

            detected = image[y: y + height, x: x + width]

            # additional check with model
            resized_detected = resize_image(detected, classifier_input_shape[2], classifier_input_shape[3])
            if is_object(resized_detected, classifier, classifier_input_shape):
                is_positive = True
                rectangle = cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 25)
            else:
                rectangle = cv2.rectangle(image, (x, y), (x + width, y + height), (0, 0, 255), 25)

            cv2.imshow('Detected object', resize_image(rectangle, 700))
            key = cv2.waitKey(0)
            cv2.destroyAllWindows()

            if is_positive:
                if key == 1114033:
                    true_pos += 1
                    correct_objects_detected_num += 1
                else:
                    false_pos += 1
            else:
                if key == 1114033:
                    true_neg += 1
                else:
                    false_neg += 1

            # saving images
            if save_mode:
                object_file_name = '{}_{}_{}'.format(x, y, file_name)
                cv2.imwrite('{}{}'.format(positives_dir if is_positive else negatives_dir, object_file_name), detected)

    # Negative result processing
    if correct_objects_detected_num == 0:
        message = 'No one candidate' if len(rectangles) == 0 else 'Not fount correct object'
        # bordered_image = cv2.rectangle(image, (0, 0), (0 + image.shape[1], 0 + image.shape[0]), (0, 0, 255), 15)
        # cv2.imshow(message, resize_image(bordered_image, 700))
        # key = cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # if key != 1114033:
        #     not_detected += 1
        print('{} {}'.format(message, file_name))
        if save_mode:
            cv2.imwrite('{}{}'.format(nf_dir, file_name), image)


print('Detection result is {}/{}'.format(len(files) - not_detected, len(files)))
print('TP {} | FP {}'.format(true_pos, false_pos))
print('TN {} | FN {}'.format(true_neg, false_neg))
