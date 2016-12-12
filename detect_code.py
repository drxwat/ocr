import cv2
import os

# todo: Move to cli args
files_dir = '../samples/51/'
cascade_path = './models/cascade_150.xml'


classifier = cv2.CascadeClassifier(cascade_path)
files = os.listdir(files_dir)

not_detected = 0

for file_name in files:
    image_path = '{}{}'.format(files_dir, file_name)
    image = cv2.imread(image_path)
    rectangles = classifier.detectMultiScale(image, minSize=(10, 10))

    if len(rectangles) > 0:
        detected_bar_code = False
        for rect in rectangles:
            x, y = rect[0:2]
            width, height = rect[2:4]

            detected = image[y: y + height, x: x + width]
            cv2.imshow('Detected object', detected)
            key = cv2.waitKey(0)

            if key == 1114033:
                detected_bar_code = True
            # 1114033 - 1
            # 1114032 - 0
            cv2.destroyAllWindows()

        if not detected_bar_code:
            print('Found wrong for {}'.format(file_name))
            not_detected += 1

    else:
        not_detected += 1
        print('Nothing found for {}'.format(file_name))


print('Detection result {}/{}'.format(not_detected, len(files)))
