import cv2
import os
import argparse
import uuid

output_dir = '.'

ap = argparse.ArgumentParser(description='Detects and shows objects using cascade classifier. Can be used to save'
                                         'positive and negative results. If you press 1-num button then displayed'
                                         'image will be marked as positive else it will be marked as negative.'
                                         'In save mode image will be saved in positives sub directory of output '
                                         'directory.')

ap.add_argument('samples_dir', help='Path to the images directory')
ap.add_argument('model', type=str, help='Cascade classifier model')
# Save mode optional arguments
ap.add_argument('-s', action='store_true')
ap.add_argument('-o', type=str, default=output_dir, metavar='OUTPUT_DIR',
                help='Output directory for save mode (default: {})'.format(output_dir))

args = ap.parse_args()
samples_dir, model_path, save_mode, output_dir = args.samples_dir, args.model, args.s, args.o

# save mod preparations
positives_dir = '{}/{}{}/'.format(output_dir, 'pos', uuid.uuid4())
negatives_dir = '{}/{}{}/'.format(output_dir, 'neg', uuid.uuid4())

if save_mode:
    os.mkdir(positives_dir)
    os.mkdir(negatives_dir)

# Loading classifier
classifier = cv2.CascadeClassifier(model_path)
files = os.listdir(samples_dir)

not_detected = 0
for file_name in files:
    image_path = '{}{}'.format(samples_dir, file_name)
    image = cv2.imread(image_path)
    # getting classifier mention =)
    rectangles = classifier.detectMultiScale(image, minSize=(10, 10))

    if len(rectangles) > 0:
        objects_detected_num = 0
        # displaying each candidate and saving if in save mode
        for rect in rectangles:
            is_positive = False
            x, y = rect[0:2]
            width, height = rect[2:4]

            detected = image[y: y + height, x: x + width]
            cv2.imshow('Detected object', detected)
            key = cv2.waitKey(0)

            if key == 1114033:
                objects_detected_num += 1
                is_positive = True

            # saving images
            if save_mode:
                object_file_name = '{}_{}_{}'.format(x, y, file_name)
                cv2.imwrite('{}{}'.format(positives_dir if is_positive else negatives_dir, object_file_name), detected)

            cv2.destroyAllWindows()

        if objects_detected_num == 0:
            print('Object not detected in {}'.format(file_name))
            not_detected += 1

    else:
        not_detected += 1
        print('No one candidates for {}'.format(file_name))


print('Detection result is {}/{}'.format(len(files) - not_detected, len(files)))
