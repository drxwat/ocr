import cv2
import os

# image_path = '../samples/DSC_0231.jpg'
# image_path = '../samples/DSC_0184.jpg'
# image_path = '../samples/2016-04-19 16-38-34.JPG'

files_dir = '../samples/'
cascade_path = './models/cascade_2.xml'

classifier = cv2.CascadeClassifier(cascade_path)
files = os.listdir(files_dir)

for file_name in files:
    image_path = '{}{}'.format(files_dir, file_name)
    image = cv2.imread(image_path)
    rectangles = classifier.detectMultiScale(image, minSize=(10, 10))

    if len(rectangles) > 0:
        for rect in rectangles:
            x, y = rect[0:2]
            width, height = rect[2:4]

            detected = image[y: y + height, x: x + width]
            cv2.imshow('Detected object', detected)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    else:
        print('Nothing found for {}'.format(file_name))
