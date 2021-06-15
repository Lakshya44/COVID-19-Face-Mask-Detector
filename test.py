import cv2
import numpy as np
from keras.models import model_from_json
from keras.preprocessing import image
import os

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

model = model_from_json(open('detect.json', 'r').read())
model.load_weights('detect.h5')

classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

labels_dict = {0: 'without_mask', 1: 'with_mask'}
color_dict = {0: (0, 0, 255), 1: (0, 255, 0)}

size = 4
cam = cv2.VideoCapture(0)

classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

while True:

    (_, image) = cam.read()
    image = cv2.flip(image, 1, 1)

    mini = cv2.resize(image, (image.shape[1] // size, image.shape[0] // size))

    faces = classifier.detectMultiScale(mini)

    for f in faces:
        x, y, w, h = [i*size for i in f]
        roi_img = image[y:y+h, x:x+w]
        resized = cv2.resize(roi_img, (150, 150))
        normalized = resized/255.0
        reshaped = np.reshape(normalized, (1, 150, 150, 3))
        reshaped = np.vstack([reshaped])
        result = model.predict(reshaped)

        label = np.argmax(result, axis=1)[0]

        cv2.rectangle(image, (x, y), (x+w, y+h), color_dict[label], 2)
        cv2.rectangle(image, (x, y-40), (x+w, y), color_dict[label], -1)
        cv2.putText(image, labels_dict[label], (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow('check', image)

    if cv2.waitKey(10) & 0xFF == 27:
        break

cam.release()
cv2.destroyAllWindows()
